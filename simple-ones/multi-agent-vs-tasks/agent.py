"""Multi-agent handoffs vs AgentTasks, side by side, on one phone call.

A dental clinic line. Both patterns run in the same call so you can hear the
difference:

  TASKS (control comes back to the caller of the task)
    - ConsentTask        runs from FrontDesk.on_enter, returns bool
    - book_appointment   runs a TaskGroup (patient name -> preferred time),
                         returns typed results to FrontDesk, which keeps talking
    - VerifyIdentityTask runs inside Billing before any account data is read

  AGENTS (control moves and stays until someone hands it back)
    - FrontDeskAgent     triage, scheduling, cheap model, no account access
    - BillingAgent       own voice, own model, own tools (balance, payment link)
                         and only it can read account data

Run locally (from this directory):
    lk agent console agent.py     # talk to it with your mic
    python agent.py console       # same, without the lk CLI
    lk agent dev agent.py         # register with LiveKit for SIP / playground

Point a dispatch rule at agent_name: multi-agent-vs-tasks
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    AgentTask,
    JobContext,
    JobProcess,
    RunContext,
    TurnHandlingOptions,
    cli,
    function_tool,
    inference,
    room_io,
)
from livekit.agents.beta.tools import EndCallTool
from livekit.agents.beta.workflows import TaskCompletedEvent, TaskGroup
from livekit.plugins import noise_cancellation, silero

_AGENT_DIR = Path(__file__).resolve().parent
load_dotenv(_AGENT_DIR.parents[1] / ".env")
load_dotenv(_AGENT_DIR / ".env", override=True)

logger = logging.getLogger("multi-agent-vs-tasks")
logger.setLevel(logging.INFO)

AGENT_NAME = "multi-agent-vs-tasks"


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Set {name} in simple-ones/multi-agent-vs-tasks/.env")
    return value


STT_MODEL = _required("STT_MODEL")
TTS_MODEL = _required("TTS_MODEL")
# Two voices so a handoff is audible. Tasks keep the voice of the agent
# that started them.
FRONT_DESK_VOICE = _required("FRONT_DESK_VOICE")
BILLING_VOICE = _required("BILLING_VOICE")
FRONT_DESK_LLM = _required("FRONT_DESK_LLM")
BILLING_LLM = _required("BILLING_LLM")

VOICE_STYLE = (
    "You are on a phone call. Keep replies to one or two short sentences. "
    "No lists, markdown, or emojis. Say numbers and times the way people speak them."
)

TURN_HANDLING = TurnHandlingOptions(
    interruption={
        "mode": "adaptive",
        "min_duration": 0.8,
        "min_words": 2,
        "resume_false_interruption": True,
        "false_interruption_timeout": 1.0,
    },
    preemptive_generation={"enabled": True},
)

# Fake account store. Code owns this, not the LLM.
ACCOUNTS = {
    "1990-04-12": {"name": "Sam", "balance": 145.50, "last_visit": "cleaning on March 3rd"},
}


@dataclass
class CallState:
    """Shared across every agent and task in the session."""

    recording_ok: bool = False
    patient_name: str | None = None
    appointment: str | None = None
    verified_dob: str | None = None
    events: list[str] = field(default_factory=list)

    def log(self, event: str) -> None:
        self.events.append(event)
        logger.info("[flow] %s", event)


def _end_call_tool() -> EndCallTool:
    return EndCallTool(
        extra_description="Call this when the caller says goodbye or is finished.",
        end_instructions="Say a brief goodbye and thank them for calling.",
        delete_room=True,
    )


# --------------------------------------------------------------------------
# TASKS: each one does a single job, returns a typed result, then gives
# control straight back to whoever awaited it.
# --------------------------------------------------------------------------


class ConsentTask(AgentTask[bool]):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                f"{VOICE_STYLE} Your only job is to get a clear yes or no to call "
                "recording. Do not help with anything else yet."
            )
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions=(
                "Say you are Bright Smile Dental. Ask if it is okay to record the "
                "call for quality. They can say no."
            )
        )

    @function_tool
    async def consent_given(self) -> None:
        """The caller agreed to recording."""
        self.complete(True)

    @function_tool
    async def consent_denied(self) -> None:
        """The caller declined recording."""
        self.complete(False)


@dataclass
class NameResult:
    name: str


class CollectNameTask(AgentTask[NameResult]):
    def __init__(self) -> None:
        super().__init__(
            instructions=f"{VOICE_STYLE} Get the patient's first name for the booking."
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(instructions="Ask whose name the appointment is for.")

    @function_tool
    async def record_name(self, name: str) -> None:
        """Save the patient's first name.

        Args:
            name: The patient's first name.
        """
        self.complete(NameResult(name=name))


@dataclass
class TimeResult:
    slot: str


class CollectTimeTask(AgentTask[TimeResult]):
    OPEN_SLOTS = ["Tuesday at 10 AM", "Tuesday at 3 PM", "Thursday at 9 AM"]

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                f"{VOICE_STYLE} Help the caller pick one of these open slots: "
                f"{', '.join(self.OPEN_SLOTS)}. Only these slots exist."
            )
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(instructions="Offer the open slots and ask which works.")

    @function_tool
    async def record_slot(self, slot: str) -> str | None:
        """Save the slot the caller picked.

        Args:
            slot: One of the open slots, exactly as offered.
        """
        if slot not in self.OPEN_SLOTS:
            return f"That slot is not open. Offer only: {', '.join(self.OPEN_SLOTS)}."
        self.complete(TimeResult(slot=slot))
        return None


@dataclass
class VerifiedResult:
    dob: str | None


class VerifyIdentityTask(AgentTask[VerifiedResult]):
    """Gate for account data. The account lookup happens in code."""

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                f"{VOICE_STYLE} Verify the caller by date of birth before any "
                "account details are shared. Never reveal account details yourself."
            )
        )
        self._attempts = 0

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions="Say you need to verify them first and ask for their date of birth."
        )

    @function_tool
    async def check_date_of_birth(self, year: int, month: int, day: int) -> str | None:
        """Check the caller's date of birth against our records.

        Args:
            year: Four-digit year.
            month: Month number, 1-12.
            day: Day of the month.
        """
        dob = f"{year:04d}-{month:02d}-{day:02d}"
        if dob in ACCOUNTS:
            self.complete(VerifiedResult(dob=dob))
            return None
        self._attempts += 1
        if self._attempts >= 2:
            self.complete(VerifiedResult(dob=None))
            return None
        return "No match. Ask them to repeat their date of birth once more."


# --------------------------------------------------------------------------
# AGENTS: each one owns the conversation while active. Returning another
# agent from a tool hands off; nothing comes back until a tool hands back.
# --------------------------------------------------------------------------


class FrontDeskAgent(Agent):
    def __init__(self, *, greet: bool = True, chat_ctx=None) -> None:
        self._greet = greet
        super().__init__(
            instructions=(
                f"{VOICE_STYLE} You are the front desk at Bright Smile Dental. "
                "You book appointments with book_appointment. "
                "You cannot see bills, balances, or payments. For any of those, "
                "use transfer_to_billing. "
                "When the caller says bye, use end_call."
            ),
            llm=inference.LLM(model=FRONT_DESK_LLM),
            tts=inference.TTS(model=TTS_MODEL, voice=FRONT_DESK_VOICE),
            tools=_end_call_tool().tools,
            chat_ctx=chat_ctx,
        )

    async def on_enter(self) -> None:
        state: CallState = self.session.userdata
        state.log("AGENT  FrontDesk is now in control")
        if not self._greet:
            # Coming back from Billing: no consent again, just pick up.
            self.session.generate_reply(
                instructions="Say you are back at the front desk and ask what else they need."
            )
            return

        # TASK from on_enter: a deterministic step before the agent talks freely.
        state.log("TASK   ConsentTask started")
        state.recording_ok = await ConsentTask()
        state.log(f"TASK   ConsentTask returned {state.recording_ok} -> back to FrontDesk")
        self.session.generate_reply(
            instructions="Thank them and ask how you can help today. Do not greet again."
        )

    @function_tool
    async def book_appointment(self, context: RunContext[CallState]) -> str:
        """Book a dental appointment. Use when the caller wants to schedule a visit."""
        state = context.userdata

        async def on_step(ev: TaskCompletedEvent) -> None:
            state.log(f"TASK   {ev.task_id} returned {ev.result}")

        # TASK GROUP from a tool: ordered steps, the caller can go back and
        # correct an earlier one ("actually it's for my son, Leo").
        group = TaskGroup(chat_ctx=self.chat_ctx, on_task_completed=on_step)
        group.add(lambda: CollectNameTask(), id="name", description="Patient's first name")
        group.add(lambda: CollectTimeTask(), id="time", description="Appointment slot")
        state.log("TASK   booking TaskGroup started")
        results = (await group).task_results

        state.patient_name = results["name"].name
        state.appointment = results["time"].slot
        state.log("TASK   booking TaskGroup done -> back to FrontDesk")
        # The return value goes to FrontDesk's LLM, which keeps the call.
        return (
            f"Booked {state.patient_name} for {state.appointment}. "
            "Confirm it in one sentence and ask if there is anything else."
        )

    @function_tool
    async def transfer_to_billing(self, context: RunContext[CallState]):
        """Transfer to billing for balances, bills, insurance, or payments."""
        context.userdata.log("HANDOFF FrontDesk -> Billing")
        # Pass a short note, not the whole transcript. Billing gets its own prompt.
        return BillingAgent(), "Transferring you to billing."


class BillingAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                f"{VOICE_STYLE} You are the billing desk at Bright Smile Dental. "
                "You answer questions about balances and payments using your tools only. "
                "For anything that is not billing, use transfer_to_front_desk. "
                "When the caller says bye, use end_call."
            ),
            # Different model and voice: the caller hears a new person.
            llm=inference.LLM(model=BILLING_LLM),
            tts=inference.TTS(model=TTS_MODEL, voice=BILLING_VOICE),
            tools=_end_call_tool().tools,
        )

    async def on_enter(self) -> None:
        state: CallState = self.session.userdata
        state.log("AGENT  Billing is now in control")
        if state.verified_dob:
            self.session.generate_reply(instructions="Say you are billing and ask how you can help.")
            return
        # TASK inside the second agent: verify before any account access.
        state.log("TASK   VerifyIdentityTask started")
        result = await VerifyIdentityTask()
        state.verified_dob = result.dob
        state.log(f"TASK   VerifyIdentityTask returned {result.dob} -> back to Billing")
        if result.dob:
            name = ACCOUNTS[result.dob]["name"]
            self.session.generate_reply(
                instructions=f"Thank {name}, say they are verified, and ask what billing question they have."
            )
        else:
            self.session.generate_reply(
                instructions="Say you could not verify them, so you can't share account details, "
                "and offer to send them back to the front desk."
            )

    @function_tool
    async def get_balance(self, context: RunContext[CallState]) -> str:
        """Read the caller's current balance and last visit."""
        dob = context.userdata.verified_dob
        if not dob:
            return "Caller is not verified. Do not share any account details."
        acct = ACCOUNTS[dob]
        return f"Balance ${acct['balance']:.2f}, last visit was a {acct['last_visit']}."

    @function_tool
    async def send_payment_link(self, context: RunContext[CallState]) -> str:
        """Text the caller a secure payment link."""
        if not context.userdata.verified_dob:
            return "Caller is not verified. Do not send a link."
        context.userdata.log("ACTION payment link sent (fake)")
        return "Payment link sent by text."

    @function_tool
    async def transfer_to_front_desk(self, context: RunContext[CallState]):
        """Transfer back to the front desk for scheduling or general questions."""
        context.userdata.log("HANDOFF Billing -> FrontDesk")
        return FrontDeskAgent(greet=False), "Sending you back to the front desk."


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name=AGENT_NAME)
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    state = CallState()

    session = AgentSession[CallState](
        userdata=state,
        stt=inference.STT(model=STT_MODEL),
        # Session defaults; each agent overrides llm and tts.
        llm=inference.LLM(model=FRONT_DESK_LLM),
        tts=inference.TTS(model=TTS_MODEL, voice=FRONT_DESK_VOICE),
        vad=ctx.proc.userdata["vad"],
        turn_handling=TURN_HANDLING,
    )

    async def _print_flow() -> None:
        logger.info("call flow:\n  %s", "\n  ".join(state.events))

    ctx.add_shutdown_callback(_print_flow)

    await session.start(
        agent=FrontDeskAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: (
                    noise_cancellation.BVCTelephony()
                    if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                    else noise_cancellation.BVC()
                ),
            ),
        ),
    )
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
