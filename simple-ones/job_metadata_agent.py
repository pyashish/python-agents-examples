"""Agent that learns the user's first name from conversation and uses it.

Flow:
  1. Ask for the user's first name (or pick it up when they say it).
  2. Save it with the record_first_name tool.
  3. Update instructions so every later turn addresses them by name.

Optional seed from job metadata (dispatch) if present — see:
https://docs.livekit.io/agents/server/job/#metadata

Run:
    python job_metadata_agent.py console
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    cli,
    function_tool,
    inference,
)
from livekit.plugins import silero

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

logger = logging.getLogger("job-metadata-agent")
logger.setLevel(logging.INFO)

AGENT_NAME = "personal-assistant"


@dataclass
class UserInfo:
    # Filled from conversation (or seeded from job metadata)
    first_name: str | None = None


def parse_first_name_from_job(metadata_raw: str | None) -> str | None:
    # Optional: seed from dispatch metadata if the client already knows the name
    if not metadata_raw:
        return None
    try:
        data = json.loads(metadata_raw)
    except json.JSONDecodeError:
        logger.warning("job metadata is not valid JSON: %r", metadata_raw)
        return None

    first_name = (
        data.get("first_name")
        or data.get("user_first_name")
        or data.get("user_name")
    )
    if not first_name:
        return None
    return str(first_name).strip() or None


def instructions_for(first_name: str | None) -> str:
    if first_name:
        return (
            f"You are a helpful voice assistant talking to {first_name}. "
            f"Always address the user as {first_name} naturally in conversation "
            f"(greetings, answers, and follow-ups). Keep replies short. "
            f"Do not invent other personal details about {first_name}."
        )
    return (
        "You are a helpful voice assistant. Keep replies short. "
        "You do not know the user's first name yet. "
        "Ask for their first name early, then call record_first_name as soon as "
        "they tell you (or if they already said it). "
        "After the name is recorded, use it in every reply."
    )


class PersonalAssistant(Agent):
    def __init__(self, first_name: str | None = None) -> None:
        super().__init__(instructions=instructions_for(first_name))

    async def on_enter(self):
        info: UserInfo = self.session.userdata
        if info.first_name:
            self.session.generate_reply(
                instructions=f"Greet {info.first_name} by name and offer help."
            )
        else:
            # Get the name from conversation first
            self.session.generate_reply(
                instructions="Greet the user briefly and ask for their first name."
            )

    @function_tool
    async def record_first_name(self, context: RunContext[UserInfo], first_name: str):
        """Save the user's first name from the conversation so you can address them by it.

        Call this as soon as the user tells you their first name.
        """
        name = first_name.strip()
        if not name:
            return "That did not look like a valid first name. Ask again."

        # Persist for the rest of the session
        context.userdata.first_name = name
        # Keep instructions in sync so later turns keep using the name
        await self.update_instructions(instructions_for(name))
        logger.info("recorded first_name from conversation: %s", name)
        return f"Saved. Address the user as {name} from now on."


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name=AGENT_NAME)
async def entrypoint(ctx: JobContext):
    # Optional seed from job metadata; conversation can still override via the tool
    seeded_name = parse_first_name_from_job(ctx.job.metadata)
    logger.info("job metadata=%r seeded_first_name=%s", ctx.job.metadata, seeded_name)

    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession[UserInfo](
        userdata=UserInfo(first_name=seeded_name),
        stt=inference.STT(model="deepgram/nova-3-general"),
        llm=inference.LLM(model="openai/gpt-4.1-mini"),
        tts=inference.TTS(
            model="fishaudio/s2.1-pro",
            voice="v_qboHnRFyFP5A",
        ),
        vad=ctx.proc.userdata["vad"],
    )

    await session.start(
        agent=PersonalAssistant(first_name=seeded_name),
        room=ctx.room,
    )
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
