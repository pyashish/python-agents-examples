"""Minimal LiveKit voice agent for console mode.

Run:
    python agent.py console
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
    inference,
)
from livekit.agents.beta.tools import EndCallTool
from livekit.plugins import silero

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")


class Assistant(Agent):
    def __init__(self) -> None:
        # LLM tool: when the user says bye / is done, end the session gracefully
        end_call = EndCallTool(
            extra_description="Call this when the user says bye, goodbye, or is done talking.",
            end_instructions="Say a brief goodbye.",
            delete_room=True,  # also tear down the room for real LiveKit sessions
        )
        super().__init__(
            instructions=(
                "You are a helpful voice assistant. Keep replies short and conversational. "
                "When the user says bye, goodbye, or indicates they are finished, "
                "use the end_call tool to end the session."
            ),
            tools=end_call.tools,
        )

    async def on_enter(self):
        self.session.generate_reply(instructions="Greet the user briefly and offer help.")


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
        tts=inference.TTS(
            model="fishaudio/s2.1-pro",
            voice="v_gZLcvnAfh3SW",
        ),
        vad=ctx.proc.userdata["vad"],
        # After 10s of mutual silence, user state becomes "away"
        user_away_timeout=10.0,
    )

    # Inactivity path: shut down when the user goes away (no speech for 10s)
    @session.on("user_state_changed")
    def on_user_state_changed(ev: UserStateChangedEvent):
        if ev.new_state == "away":
            session.shutdown()

    await session.start(agent=Assistant(), room=ctx.room)
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
