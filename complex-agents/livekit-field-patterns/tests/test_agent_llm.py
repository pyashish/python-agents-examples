"""Turn-level behavior with a real model. Needs LiveKit Inference credentials.

Skipped unless LIVEKIT_API_KEY and LIVEKIT_API_SECRET are set (for example in .env.local).
The end_call tool is mocked so nothing tries to delete a room.
"""

import os

import pytest
from dotenv import load_dotenv
from livekit.agents import AgentSession, inference, mock_tools

from agent.phone_agent import PhoneAgent
from agent.state import CallState
from shared.samples import sample_inbound_config

load_dotenv(".env.local")

pytestmark = pytest.mark.skipif(
    not (os.getenv("LIVEKIT_API_KEY") and os.getenv("LIVEKIT_API_SECRET")),
    reason="needs LiveKit Inference credentials",
)


async def test_agent_looks_up_the_invoice_the_caller_names():
    config = sample_inbound_config()
    async with (
        inference.LLM(model=config.llm_model) as model,
        AgentSession[CallState](llm=model, userdata=CallState(config=config)) as session,
    ):
        await session.start(PhoneAgent(config, greet_on_enter=False))
        result = await session.run(user_input="Hi, can you check invoice INV-20417 for me?")
        result.expect.contains_function_call(name="lookup_invoice", arguments={"invoice_number": "INV-20417"})


async def test_agent_ends_the_call_when_the_caller_is_done():
    config = sample_inbound_config()
    ended: list[bool] = []

    async def fake_end_call(said_goodbye: bool = False) -> None:
        ended.append(said_goodbye)

    async with (
        inference.LLM(model=config.llm_model) as model,
        AgentSession[CallState](llm=model, userdata=CallState(config=config)) as session,
    ):
        with mock_tools(PhoneAgent, {"end_call": fake_end_call}):
            await session.start(PhoneAgent(config, greet_on_enter=False))
            result = await session.run(user_input="That's everything, thanks. Bye!")
            result.expect.contains_function_call(name="end_call")
