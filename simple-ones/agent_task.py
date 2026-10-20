"""AgentTask demo: chain several tasks and carry the chat forward.

Run:
    python agent_task.py console

Each task returns a typed result. When it completes, its user and assistant
turns are merged back onto this agent. The next task receives a copy of that
history, plus a short note of facts already collected.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    AgentTask,
    JobContext,
    JobProcess,
    UserStateChangedEvent,
    cli,
    function_tool,
    inference,
)
from livekit.agents.beta.tools import EndCallTool
from livekit.agents.llm import ChatContext
from livekit.plugins import silero

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

# print() stays in the job process and never shows in `lk agent console`.
# Logging records are forwarded to the console, which is already at DEBUG.
logger = logging.getLogger("agent-task")
logger.setLevel(logging.DEBUG)


def conversation_so_far(agent: Agent, *, note: str) -> ChatContext:
    """Copy the spoken history for the next task.

    Drops this agent's system prompt, tool calls, and handoff markers so the
    next task follows its own instructions. `note` is a developer message
    added after the copy, so the facts survive that filter.
    """
    chat_ctx = agent.chat_ctx.copy(
        exclude_instructions=True,
        exclude_function_call=True,
        exclude_handoff=True,
        exclude_empty_message=True,
    )
    chat_ctx.add_message(role="developer", content=note)
    logger.debug(
        "forwarding chat to next task note=%r items=%s",
        note,
        [
            {
                "type": item.type,
                "role": getattr(item, "role", None),
                "content": getattr(item, "content", None),
            }
            for item in chat_ctx.items
        ],
    )
    return chat_ctx


@dataclass
class NameResult:
    name: str


@dataclass
class ColorResult:
    name: str
    color: str


@dataclass
class CityResult:
    name: str
    color: str
    city: str


class CollectNameTask(AgentTask[NameResult]):
    """Ask for a first name and return it."""

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "Collect the user's first name. "
                "When they give it, call record_name. Do not ask anything else."
            )
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="Ask for their first name in one short sentence."
        )

    @function_tool
    async def record_name(self, name: str) -> None:
        """Save the user's first name once they have said it.

        Args:
            name: The user's first name.
        """
        self.complete(NameResult(name=name))


class CollectColorTask(AgentTask[ColorResult]):
    """Ask for a favorite color, using the name and the prior turns."""

    def __init__(self, name: str, chat_ctx=None) -> None:
        self._name = name
        super().__init__(
            instructions=(
                f"The user's name is {name}. The chat history is the conversation so far. "
                "Address them by name and ask for their favorite color. "
                "If they already named a color, record that instead of asking again. "
                "When you have it, call record_color."
            ),
            chat_ctx=chat_ctx,
        )

    async def on_enter(self) -> None:
        logger.debug("Full chat context in CollectColorTask.on_enter:")
        for idx, item in enumerate(self.chat_ctx.items):
            logger.debug(f"{idx}: type={getattr(item, 'type', None)}, role={getattr(item, 'role', None)}, content={getattr(item, 'content', None)}")
   
        await self.session.generate_reply(
            instructions=f"Address them as {self._name} and ask their favorite color."
        )

    @function_tool
    async def record_color(self, color: str) -> None:
        """Save the user's favorite color.

        Args:
            color: The color they named.
        """
        self.complete(ColorResult(name=self._name, color=color))


class CollectCityTask(AgentTask[CityResult]):
    """Ask for a city, with the name, color, and full prior chat."""

    def __init__(self, name: str, color: str, chat_ctx=None) -> None:
        self._name = name
        self._color = color
        super().__init__(
            instructions=(
                f"The user's name is {name} and their favorite color is {color}. "
                "The chat history includes everything they have said so far. "
                "Ask which city they live in. If they already named a city, record it. "
                "Do not ask for their name or color again. When you have the city, call record_city."
            ),
            chat_ctx=chat_ctx,
        )

    async def on_enter(self) -> None:
        logger.debug("Full chat context in CollectCityTask.on_enter:")
        for idx, item in enumerate(self.chat_ctx.items):
            logger.debug(f"{idx}: type={getattr(item, 'type', None)}, role={getattr(item, 'role', None)}, content={getattr(item, 'content', None)}")
   
        await self.session.generate_reply(
            instructions=(
                f"Address them as {self._name} and ask which city they live in. "
                f"You already know their favorite color is {self._color}."
            )
        )

    @function_tool
    async def record_city(self, city: str) -> None:
        """Save the city the user lives in.

        Args:
            city: The city they named.
        """
        self.complete(CityResult(name=self._name, color=self._color, city=city))


class Assistant(Agent):
    def __init__(self) -> None:
        end_call = EndCallTool(
            extra_description="Call this when the user says bye, goodbye, or is done talking.",
            end_instructions="Say a brief goodbye.",
            delete_room=True,
        )
        super().__init__(
            instructions=(
                "You are a helpful voice assistant. Keep replies short and conversational. "
                "When the user says bye, goodbye, or indicates they are finished, "
                "use the end_call tool to end the session."
            ),
            tools=end_call.tools,
        )

    async def on_enter(self) -> None:
        # complete() merges that task's user/assistant turns into self.chat_ctx
        # before the next await, so each copy below is the history so far.
        name_result = await CollectNameTask()
        logger.debug("name task result: %s", name_result)
        logger.debug("chat context from previous task: %s", conversation_so_far(self, note=f"Collected so far: name is {name_result.name}."))
        color_result = await CollectColorTask(
            name=name_result.name,
            chat_ctx=conversation_so_far(
                self,
                note=f"Collected so far: name is {name_result.name}.",
            ),
        )
        logger.debug("color task result: %s", color_result)

        city_result = await CollectCityTask(
            name=color_result.name,
            color=color_result.color,
            chat_ctx=conversation_so_far(
                self,
                note=(
                    f"Collected so far: name is {color_result.name}, "
                    f"favorite color is {color_result.color}."
                ),
            ),
        )
        logger.debug("city task result: %s", city_result)

        await self.session.generate_reply(
            instructions=(
                f"Their name is {city_result.name}, favorite color is {city_result.color}, "
                f"and they live in {city_result.city}. "
                "Confirm all three in one short sentence, then offer more help."
            )
        )


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt=inference.STT(model="deepgram/nova-3-general"),
        llm=inference.LLM(model="openai/gpt-4.1-mini"),
        tts=inference.TTS(
            model="fishaudio/s2.1-pro",
        ),
        vad=ctx.proc.userdata["vad"],
        user_away_timeout=30.0,
    )

    @session.on("user_state_changed")
    def on_user_state_changed(ev: UserStateChangedEvent):
        if ev.new_state == "away":
            session.shutdown()

    await session.start(agent=Assistant(), room=ctx.room)
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
