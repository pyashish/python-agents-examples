"""Text-only tests: no audio, but real LLM calls via LiveKit Inference.

Run (from this directory):
    pytest -v test_agent.py
"""

import asyncio

import pytest
from livekit.agents import AgentSession, inference

from agent import (
    FRONT_DESK_LLM,
    BillingAgent,
    CallState,
    FrontDeskAgent,
    VerifyIdentityTask,
)

pytestmark = pytest.mark.asyncio


async def _start() -> tuple[AgentSession, CallState]:
    state = CallState()
    session = AgentSession[CallState](
        llm=inference.LLM(model=FRONT_DESK_LLM), userdata=state
    )
    await session.start(FrontDeskAgent())
    # Let ConsentTask take over before the first user turn.
    await asyncio.sleep(0.5)
    return session, state


async def _until(cond, timeout: float = 5.0) -> None:
    """Tool bodies can finish a moment after session.run() returns."""
    for _ in range(int(timeout / 0.1)):
        if cond():
            return
        await asyncio.sleep(0.1)
    raise AssertionError("condition not met")


async def test_task_returns_control_to_same_agent():
    session, state = await _start()
    async with session:
        result = await session.run(user_input="Yes, recording is fine.")
        result.expect.contains_function_call(name="consent_given")
        assert state.recording_ok is True
        assert isinstance(session.current_agent, FrontDeskAgent)

        result = await session.run(user_input="I'd like to book a cleaning.")
        result.expect.contains_function_call(name="book_appointment")
        await session.run(user_input="It's for Sam.")
        await session.run(user_input="Tuesday at 10 AM works.")

        await _until(lambda: state.appointment is not None)
        assert state.patient_name == "Sam"
        assert state.appointment == "Tuesday at 10 AM"
        # The booking tasks finished and FrontDesk still owns the call.
        assert isinstance(session.current_agent, FrontDeskAgent)


async def test_handoff_moves_control_to_billing():
    session, state = await _start()
    async with session:
        await session.run(user_input="Sure, go ahead and record.")

        result = await session.run(user_input="How much do I owe on my account?")
        result.expect.contains_agent_handoff(new_agent_type=BillingAgent)
        # Billing's on_enter started a task, so the task is in control for now.
        await _until(lambda: isinstance(session.current_agent, VerifyIdentityTask))

        result = await session.run(user_input="April twelfth, nineteen ninety.")
        result.expect.contains_function_call(name="check_date_of_birth")
        await _until(lambda: state.verified_dob is not None)
        assert state.verified_dob == "1990-04-12"
        # Task done: control is back with the agent that started it.
        assert isinstance(session.current_agent, BillingAgent)

        result = await session.run(user_input="So what's my balance?")
        result.expect.contains_function_call(name="get_balance")
        # Billing is still in control: nothing handed it back.
        assert isinstance(session.current_agent, BillingAgent)

        result = await session.run(user_input="Thanks. Can I also book a checkup?")
        result.expect.contains_agent_handoff(new_agent_type=FrontDeskAgent)

    print("\n".join(state.events))
