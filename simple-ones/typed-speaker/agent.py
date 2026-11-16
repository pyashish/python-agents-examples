"""Run a local voice session controlled from a separate operator terminal."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    UserInputTranscribedEvent,
    cli,
    inference,
    room_io,
)

HERE = Path(__file__).resolve().parent
load_dotenv(HERE.parents[1] / ".env")

server = AgentServer()


class Speaker(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="Transcribe the user while a human operator supplies replies."
        )


class OperatorBridge:
    """Exchange transcripts and typed replies over a localhost TCP socket."""

    def __init__(self, session: AgentSession) -> None:
        self._session = session
        self._writer: asyncio.StreamWriter | None = None

    async def handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        previous = self._writer
        if previous is not None:
            previous.close()

        self._writer = writer
        self._send({"type": "status", "text": "Connected to local voice agent"})

        try:
            while line := await reader.readline():
                message: dict[str, Any] = json.loads(line)
                reply = str(message.get("text", "")).strip()
                if message.get("type") == "reply" and reply:
                    await self._speak(reply)
        except (ConnectionError, json.JSONDecodeError):
            pass
        finally:
            if self._writer is writer:
                self._writer = None
            writer.close()
            await writer.wait_closed()

    def publish_transcript(self, text: str) -> None:
        self._send({"type": "transcript", "text": text})

    async def _speak(self, text: str) -> None:
        self._send({"type": "status", "text": f"Speaking: {text}"})
        print(f"Operator reply: {text}")

        try:
            speech = self._session.say(
                text,
                allow_interruptions=False,
            )
            await speech.wait_for_playout()
            if error := speech.exception():
                raise error
        except Exception as error:
            print(f"Speech failed: {error!r}")
            self._send(
                {
                    "type": "error",
                    "text": f"Speech failed: {error}",
                }
            )
            return

        self._send({"type": "status", "text": "Finished speaking"})

    def _send(self, message: dict[str, str]) -> None:
        if self._writer is None or self._writer.is_closing():
            return
        self._writer.write((json.dumps(message) + "\n").encode())


@server.rtc_session(agent_name="typed-speaker")
async def entrypoint(ctx: JobContext) -> None:
    voice = os.getenv("PAPPA_VOICE")
    if not voice:
        raise RuntimeError("PAPPA_VOICE must be set")

    session = AgentSession(
        stt=inference.STT(
            model="deepgram/nova-3-general",
            language="en",
        ),
        tts=inference.TTS(
            model="cartesia/sonic-3",
            voice=voice,
        ),
    )

    operator = OperatorBridge(session)

    @session.on("user_input_transcribed")
    def on_user_input_transcribed(event: UserInputTranscribedEvent) -> None:
        if event.is_final and event.transcript.strip():
            operator.publish_transcript(event.transcript.strip())

    await session.start(
        agent=Speaker(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_output=True,
            text_input=False,
        ),
    )
    await ctx.connect()

    host = os.getenv("OPERATOR_HOST", "127.0.0.1")
    port = int(os.getenv("OPERATOR_PORT", "8765"))
    operator_server = await asyncio.start_server(
        operator.handle_connection,
        host,
        port,
    )
    print(f"Local operator socket listening on {host}:{port}")

    async with operator_server:
        await operator_server.serve_forever()


if __name__ == "__main__":
    cli.run_app(server)
