"""Math voice agent with a three-strike off-topic observer."""

from __future__ import annotations

import asyncio
import json
import logging

from dotenv import load_dotenv
from livekit import api
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    ConversationItemAddedEvent,
    JobContext,
    JobProcess,
    cli,
    inference,
    llm,
)
from livekit.plugins import silero

load_dotenv()

logger = logging.getLogger("math-offscript-observer")
server = AgentServer()


class TopicObserver:
    """Evaluate every user turn sequentially and enforce a three-strike policy."""

    def __init__(
        self,
        session: AgentSession,
        decision_model: llm.DecisionModel,
        job_context: JobContext,
    ) -> None:
        self._session = session
        self._decision_model = decision_model
        self._job_context = job_context
        self._off_topic_count = 0
        self._call_ended = False
        self._turn_queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None
        self._setup_listener()

    def _setup_listener(self) -> None:
        @self._session.on("conversation_item_added")
        def on_conversation_item_added(event: ConversationItemAddedEvent) -> None:
            # Handoffs and tool events can also be committed to the conversation.
            if (
                self._call_ended
                or not isinstance(event.item, llm.ChatMessage)
                or event.item.role != "user"
            ):
                return

            text = event.item.text_content
            if not text or not text.strip():
                return

            self._turn_queue.put_nowait(text.strip())
            if self._worker is None or self._worker.done():
                self._worker = asyncio.create_task(self._process_turns())

    async def _process_turns(self) -> None:
        while not self._turn_queue.empty() and not self._call_ended:
            text = await self._turn_queue.get()
            try:
                await self._evaluate_turn(text)
            except Exception:
                # A model or network failure must not count as a user violation.
                logger.exception("[TOPIC OBSERVER] evaluation failed")
            finally:
                self._turn_queue.task_done()

    async def _evaluate_turn(self, text: str) -> None:
        response = await self._decision_model.evaluate(
            input=text,
            decisions={
                "topic": llm.Choice(
                    prompt=(
                        "Classify the latest user turn for a math-only assistant. "
                        "Choose on_topic for a math question, a response that advances the "
                        "current math problem, or a brief greeting, thanks, goodbye, repeat, "
                        "or clarification request. Choose off_topic for any substantive "
                        "request or discussion unrelated to mathematics."
                    ),
                    options={
                        "on_topic": "Allowed math content or brief conversation control.",
                        "off_topic": "Substantive content unrelated to mathematics.",
                    },
                )
            },
        )

        if error := response.errors.get("topic"):
            logger.error("[TOPIC OBSERVER] decision error: %s", error)
            return

        result = response.results["topic"]
        if not isinstance(result, llm.ChoiceResult):
            logger.error("[TOPIC OBSERVER] unexpected result kind: %s", result.kind)
            return

        confidence = (
            result.probabilities.get(result.value)
            if result.probabilities is not None
            else None
        )
        if result.value == "off_topic":
            self._off_topic_count += 1

        debug_trace = {
            "input": {
                "user_turn": text,
                "decision": "topic",
            },
            "response": response.model_dump(mode="json"),
            "policy_state": {
                "selected": result.value,
                "confidence": confidence,
                "off_topic_count": self._off_topic_count,
                "strikes_remaining": max(0, 3 - self._off_topic_count),
            },
        }
        logger.debug(
            "[TOPIC OBSERVER] DecisionModel trace:\n%s",
            json.dumps(debug_trace, indent=2, sort_keys=True),
        )

        if result.value == "off_topic":
            await self._handle_violation()

    async def _handle_violation(self) -> None:
        await self._interrupt_current_reply()
        await self._inject_policy_update()

        if self._off_topic_count == 1:
            warning = self._session.say(
                "This agent is only for math. Please ask a math question. "
                "This is your first warning.",
                allow_interruptions=False,
            )
            await warning.wait_for_playout()
            return

        if self._off_topic_count == 2:
            warning = self._session.say(
                "Please stay on the topic of math. This is your second and final warning. "
                "Another off-topic request will end the call.",
                allow_interruptions=False,
            )
            await warning.wait_for_playout()
            return

        self._call_ended = True
        closing_speech = self._session.say(
            "You have made three off-topic requests, so this call will now end. Goodbye.",
            allow_interruptions=False,
        )
        await closing_speech.wait_for_playout()
        await self._disconnect_everyone()

    async def _interrupt_current_reply(self) -> None:
        try:
            await self._session.interrupt(force=True)
        except RuntimeError:
            # There may be no generated reply to interrupt yet.
            pass

    async def _inject_policy_update(self) -> None:
        current_agent = self._session.current_agent
        chat_ctx = current_agent.chat_ctx.copy()
        chat_ctx.add_message(
            role="system",
            content=(
                f"[OBSERVER: OFF-TOPIC STRIKE {self._off_topic_count}/3] "
                "The user made a substantive non-math request. Continue to answer only "
                "mathematics questions. The observer owns warnings and call termination."
            ),
        )
        await current_agent.update_chat_ctx(chat_ctx)

    async def _disconnect_everyone(self) -> None:
        try:
            # Deleting the room disconnects the caller too; session shutdown alone does not.
            await self._job_context.api.room.delete_room(
                api.DeleteRoomRequest(room=self._job_context.room.name)
            )
        except Exception:
            logger.exception("[TOPIC OBSERVER] failed to delete room")
        finally:
            self._session.shutdown(drain=False)


class MathAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a focused math tutor. Solve mathematics questions clearly and "
                "concisely, showing useful steps. Do not answer substantive non-math "
                "requests. The background observer independently tracks off-topic turns, "
                "delivers warnings, and may end the call."
            )
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions=(
                "Greet the user briefly, explain that you are a math-only tutor, and ask "
                "what math problem they would like help with."
            )
        )


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name="math-offscript-observer")
async def entrypoint(ctx: JobContext) -> None:
    session = AgentSession(
        stt=inference.STT(model="deepgram/nova-3", language="en"),
        llm=inference.LLM(model="openai/gpt-4.1-mini"),
        tts=inference.TTS(
            model="deepgram/aura-2",
            voice="athena",
            language="en",
            fallback=[
                {
                    "model": "cartesia/sonic-3",
                    "voice": "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
                }
            ],
        ),
        vad=ctx.proc.userdata["vad"],
    )

    TopicObserver(
        session=session,
        decision_model=inference.DecisionModel("livekit/clef-omni"),
        job_context=ctx,
    )

    await session.start(agent=MathAgent(), room=ctx.room)
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
