"""Speech-to-speech voice agent (LiveKit Inference realtime models).

Run in the browser or console:
    python agent.py console
    python agent.py dev

Switch models with environment variables (see README.md).
"""

from __future__ import annotations

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
    inference,
)
from livekit.agents.beta.tools import EndCallTool
from livekit.plugins import openai, silero

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")

# --- Model selection (override in .env or shell) ---
# "realtime" = speech-to-speech | "cascade" = STT + LLM + TTS | "duplex" = full-duplex GPT-Live
VOICE_PIPELINE = os.getenv("VOICE_PIPELINE", "realtime").strip().lower()

# inference = LiveKit Cloud (no OPENAI_API_KEY) | openai = OpenAI Realtime API plugin
REALTIME_BACKEND = os.getenv("REALTIME_BACKEND", "inference").strip().lower()

REALTIME_MODEL = os.getenv("REALTIME_MODEL", "openai/gpt-realtime")
REALTIME_VOICE = os.getenv("REALTIME_VOICE", "").strip() or None

# Used when REALTIME_BACKEND=openai (unprefixed OpenAI model ids)
OPENAI_REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime")
OPENAI_REALTIME_VOICE = os.getenv("OPENAI_REALTIME_VOICE", "marin")

DUPLEX_MODEL = os.getenv("DUPLEX_MODEL", "openai/gpt-live-1")
DUPLEX_VOICE = os.getenv("DUPLEX_VOICE", "").strip() or None


class Assistant(Agent):
    def __init__(self) -> None:
        end_call = EndCallTool(
            extra_description="Call when the user says bye, goodbye, or is done talking.",
            end_instructions="Say a brief goodbye.",
            delete_room=True,
        )
        super().__init__(
            instructions=(
                "You are a helpful voice assistant in a speech-to-speech demo. "
                "Keep replies short and natural — this is a voice call, not a essay. "
                "When the user is finished, use end_call."
            ),
            tools=end_call.tools,
        )

    async def on_enter(self):
        self.session.generate_reply(
            instructions="Greet the user briefly and invite them to try the realtime voice model."
        )


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


def _build_realtime_llm():
    voice = REALTIME_VOICE or None

    if REALTIME_BACKEND == "openai":
        plugin_voice = voice or OPENAI_REALTIME_VOICE
        return openai.realtime.RealtimeModel(
            model=OPENAI_REALTIME_MODEL,
            voice=plugin_voice,
        )

    if REALTIME_BACKEND != "inference":
        raise ValueError(
            f"Unknown REALTIME_BACKEND={REALTIME_BACKEND!r}; use inference or openai"
        )

    rt_kwargs: dict = {"model": REALTIME_MODEL}
    if voice:
        rt_kwargs["voice"] = voice
    return inference.RealtimeModel(**rt_kwargs)


def _build_session(ctx: JobContext) -> AgentSession:
    vad = ctx.proc.userdata["vad"]

    if VOICE_PIPELINE == "cascade":
        return AgentSession(
            stt=inference.STT(model="deepgram/nova-3-general"),
            llm=inference.LLM(model="openai/gpt-4.1-mini"),
            tts=inference.TTS(
                model="fishaudio/s2.1-pro",
                voice="v_gZLcvnAfh3SW",
            ),
            vad=vad,
            user_away_timeout=15.0,
        )

    if VOICE_PIPELINE == "duplex":
        kwargs: dict = {"model": DUPLEX_MODEL}
        if DUPLEX_VOICE:
            kwargs["voice"] = DUPLEX_VOICE
        return AgentSession(
            llm=inference.DuplexModel(**kwargs),
            vad=vad,
            user_away_timeout=15.0,
        )

    if VOICE_PIPELINE != "realtime":
        raise ValueError(
            f"Unknown VOICE_PIPELINE={VOICE_PIPELINE!r}; use realtime, cascade, or duplex"
        )

    return AgentSession(
        llm=_build_realtime_llm(),
        vad=vad,
        user_away_timeout=15.0,
    )


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = _build_session(ctx)

    @session.on("user_state_changed")
    def on_user_state_changed(ev: UserStateChangedEvent):
        if ev.new_state == "away":
            session.shutdown()

    await session.start(agent=Assistant(), room=ctx.room)
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
