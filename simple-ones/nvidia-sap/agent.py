"""Simple voice agent using NVIDIA NIM Parakeet (STT) and Magpie (TTS)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    inference,
)
from livekit.plugins import nvidia, silero

HERE = Path(__file__).parent
load_dotenv(HERE / ".env.livekit")
load_dotenv(HERE / ".env.nvidia")

logger = logging.getLogger("nvidia-sap")
server = AgentServer()


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a friendly voice assistant. Keep replies short and "
                "conversational, one or two sentences. Do not use markdown, lists, "
                "or emojis, since everything you say is spoken aloud."
            )
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="Greet the user briefly and ask how you can help."
        )


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


def speech_models() -> tuple:
    """STACK=nvidia (default), livekit (Deepgram + Cartesia) or assemblyai-gradium; everything
    else in the session stays identical."""
    stack = os.getenv("STACK", "nvidia")
    if stack == "assemblyai-gradium":
        return (
            inference.STT(model="assemblyai/universal-3-6-pro", language="en"),
            inference.TTS(model="gradium/default", voice="4SZHfMpw-p46Ywgs", language="en"),
        )
    if stack == "livekit":
        return (
            inference.STT(model="deepgram/nova-3", language="en"),
            inference.TTS(
                model="cartesia/sonic-3",
                voice=os.getenv("CARTESIA_VOICE", "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
            ),
        )
    return (
        nvidia.STT(
            model=os.getenv(
                "NVIDIA_STT_MODEL",
                "parakeet-1.1b-en-US-asr-streaming-silero-vad-sortformer",
            ),
            language_code="en-US",
        ),
        nvidia.TTS(
            voice=os.getenv("NVIDIA_TTS_VOICE", "Magpie-Multilingual.EN-US.Leo"),
            language_code="en-US",
        ),
    )


@server.rtc_session(agent_name="nvidia-sap")
async def entrypoint(ctx: JobContext) -> None:
    stt, tts = speech_models()
    session = AgentSession(
        stt=stt,
        llm=inference.LLM(model="openai/gpt-4.1-mini"),
        tts=tts,
        vad=ctx.proc.userdata["vad"],
    )

    await session.start(agent=Assistant(), room=ctx.room)
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
