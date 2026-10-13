"""Dual-agent demo: Amma and Pappa with different Fish voices.

Run:
    python pappa_amma_agent.py console
"""

from pathlib import Path

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    UserStateChangedEvent,
    cli,
    function_tool,
    inference,
)
from livekit.agents.beta.tools import EndCallTool
from livekit.plugins import silero

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

# Fish Audio voice IDs
PAPPA_VOICE = "v_qboHnRFyFP5A"
AMMA_VOICE = "v_gZLcvnAfh3SW"

TTS_MODEL = "fishaudio/s2.1-pro"


def _end_call_tool() -> EndCallTool:
    # Shared goodbye / hang-up tool for both agents
    return EndCallTool(
        extra_description="Call this when the user says bye, goodbye, or is done talking.",
        end_instructions="Say a brief warm goodbye.",
        delete_room=True,
    )


class AmmaAgent(Agent):
    def __init__(self) -> None:
        end_call = _end_call_tool()
        super().__init__(
            instructions=(
                "You are Amma — a warm, caring mother figure. Keep replies short and conversational. "
                "If the user asks for Pappa, or wants to talk to dad/father, use transfer_to_pappa. "
                "When the user says bye or is finished, use the end_call tool."
            ),
            # Amma's voice (overrides session TTS while this agent is active)
            tts=inference.TTS(model=TTS_MODEL, voice=AMMA_VOICE),
            tools=end_call.tools,
        )

    async def on_enter(self):
        self.session.generate_reply(
            instructions="Greet the user briefly as Amma and offer help. Mention they can ask for Pappa anytime."
        )

    @function_tool
    async def transfer_to_pappa(self):
        """Hand the conversation over to Pappa (different voice)."""
        # Memory: returning a new Agent() starts with a fresh chat_ctx (separate
        # LLM memory per agent). session.history still has the full session log.
        # To SHARE Amma's conversation with Pappa instead:
        #   return PappaAgent(chat_ctx=self.chat_ctx.copy(exclude_instructions=True))
        # To keep SEPARATE long-lived memories, reuse instances from session.userdata
        # instead of constructing PappaAgent() each handoff.
        return PappaAgent()


class PappaAgent(Agent):
    def __init__(self) -> None:
        end_call = _end_call_tool()
        super().__init__(
            instructions=(
                "You are Pappa — a calm, supportive father figure. Keep replies short and conversational. "
                "If the user asks for Amma, or wants to talk to mom/mother, use transfer_to_amma. "
                "When the user says bye or is finished, use the end_call tool."
            ),
            # Pappa's voice
            tts=inference.TTS(model=TTS_MODEL, voice=PAPPA_VOICE),
            tools=end_call.tools,
        )

    async def on_enter(self):
        self.session.generate_reply(
            instructions="Greet the user briefly as Pappa. Mention they can ask for Amma anytime."
        )

    @function_tool
    async def transfer_to_amma(self):
        """Hand the conversation over to Amma (different voice)."""
        # Memory: same as transfer_to_pappa — fresh chat_ctx by default.
        # To share history: return AmmaAgent(chat_ctx=self.chat_ctx.copy(exclude_instructions=True))
        return AmmaAgent()


server = AgentServer()


def prewarm(proc: JobProcess):
    # Load VAD once per process for faster session start
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt=inference.STT(model="deepgram/nova-3-general"),
        llm=inference.LLM(model="openai/gpt-4.1-mini"),
        # Fallback TTS; each agent sets its own voice via tts=
        tts=inference.TTS(model=TTS_MODEL, voice=AMMA_VOICE),
        vad=ctx.proc.userdata["vad"],
        # After 10s of mutual silence, user state becomes "away"
        user_away_timeout=10.0,
    )

    # Inactivity path: shut down when the user goes away (no speech for 10s)
    @session.on("user_state_changed")
    def on_user_state_changed(ev: UserStateChangedEvent):
        if ev.new_state == "away":
            session.shutdown()

    # Start with Amma; user can ask to talk to Pappa
    await session.start(agent=AmmaAgent(), room=ctx.room)
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
