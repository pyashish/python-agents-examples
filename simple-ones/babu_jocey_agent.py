"""Dual-agent demo: Babu and Jocey, both 6-year-old Fish voices.

Run:
    python babu_jocey_agent.py console
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

# Fish Audio voice IDs (6-year-old voices) — fill these in
BABU_VOICE = "v_wKzDk8wtHBba"
JOCEY_VOICE = "v_eUUvxDLoaHss"

TTS_MODEL = "fishaudio/s2.1-pro"

KID_STYLE = (
    "You are 6 years old. Talk the way a 6-year-old talks: short sentences, "
    "easy words, and examples a little kid would use. "
    "Explain anything the way a 6-year-old would explain it. "
    "If something is too hard, grown-up, or complicated, say you don't know. "
    "Do not pretend to understand big or complex things. "
    "Keep replies short and conversational."
)


def _end_call_tool() -> EndCallTool:
    # Shared goodbye / hang-up tool for both agents
    return EndCallTool(
        extra_description="Call this when the user says bye, goodbye, or is done talking.",
        end_instructions="Say a brief goodbye the way a 6-year-old would.",
        delete_room=True,
    )


class BabuAgent(Agent):
    def __init__(self) -> None:
        end_call = _end_call_tool()
        super().__init__(
            instructions=(
                f"You are Babu, a 6-year-old boy. {KID_STYLE} "
                "If the user asks for Jocey, use transfer_to_jocey. "
                "When the user says bye or is finished, use the end_call tool."
            ),
            # Babu's voice (overrides session TTS while this agent is active)
            tts=inference.TTS(model=TTS_MODEL, voice=BABU_VOICE),
            tools=end_call.tools,
        )

    async def on_enter(self):
        self.session.generate_reply(
            instructions=(
                "Say hi briefly as Babu, like a 6-year-old. "
                "Mention they can ask for Jocey anytime."
            )
        )

    @function_tool
    async def transfer_to_jocey(self):
        """Hand the conversation over to Jocey (different voice)."""
        # Memory: returning a new Agent() starts with a fresh chat_ctx (separate
        # LLM memory per agent). session.history still has the full session log.
        # To SHARE Babu's conversation with Jocey instead:
        #   return JoceyAgent(chat_ctx=self.chat_ctx.copy(exclude_instructions=True))
        # To keep SEPARATE long-lived memories, reuse instances from session.userdata
        # instead of constructing JoceyAgent() each handoff.
        return JoceyAgent()


class JoceyAgent(Agent):
    def __init__(self) -> None:
        end_call = _end_call_tool()
        super().__init__(
            instructions=(
                f"You are Jocey, a 6-year-old girl. {KID_STYLE} "
                "If the user asks for Babu, use transfer_to_babu. "
                "When the user says bye or is finished, use the end_call tool."
            ),
            # Jocey's voice
            tts=inference.TTS(model=TTS_MODEL, voice=JOCEY_VOICE),
            tools=end_call.tools,
        )

    async def on_enter(self):
        self.session.generate_reply(
            instructions=(
                "Say hi briefly as Jocey, like a 6-year-old. "
                "Mention they can ask for Babu anytime."
            )
        )

    @function_tool
    async def transfer_to_babu(self):
        """Hand the conversation over to Babu (different voice)."""
        # Memory: same as transfer_to_jocey — fresh chat_ctx by default.
        # To share history: return BabuAgent(chat_ctx=self.chat_ctx.copy(exclude_instructions=True))
        return BabuAgent()


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
        tts=inference.TTS(model=TTS_MODEL, voice=BABU_VOICE),
        vad=ctx.proc.userdata["vad"],
        # After 10s of mutual silence, user state becomes "away"
        user_away_timeout=10.0,
    )

    # Inactivity path: shut down when the user goes away (no speech for 10s)
    @session.on("user_state_changed")
    def on_user_state_changed(ev: UserStateChangedEvent):
        if ev.new_state == "away":
            session.shutdown()

    # Start with Babu; user can ask to talk to Jocey
    await session.start(agent=BabuAgent(), room=ctx.room)
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
