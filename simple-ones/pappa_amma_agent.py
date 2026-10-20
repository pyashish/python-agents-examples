"""Dual-agent demo: Amma and Pappa with different Fish voices.

Run:
    python pappa_amma_agent.py console
"""

import os
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

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)


def _required(name: str) -> str:
    value = os.getenv(name, "").strip().strip('"').strip("'")
    if not value:
        raise RuntimeError(f"Set {name} in {_ENV_PATH}")
    return value


# Fish Audio voice IDs and session models (see repo .env)
PAPPA_VOICE = _required("PAPPA_VOICE")
AMMA_VOICE = _required("AMMA_VOICE")
TTS_MODEL = _required("TTS_MODEL")
STT_MODEL = _required("STT_MODEL")
LLM_MODEL = _required("LLM_MODEL")
USER_AWAY_TIMEOUT = float(_required("USER_AWAY_TIMEOUT"))


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
        stt=inference.STT(model=STT_MODEL),
        llm=inference.LLM(model=LLM_MODEL),
        # Fallback TTS; each agent sets its own voice via tts=
        tts=inference.TTS(model=TTS_MODEL, voice=AMMA_VOICE),
        vad=ctx.proc.userdata["vad"],
        # After this many seconds of mutual silence, user state becomes "away"
        user_away_timeout=USER_AWAY_TIMEOUT,
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
