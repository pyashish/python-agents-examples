"""Outbound-style reminder agent with a voicemail detection observer."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Literal

from dotenv import load_dotenv
from livekit import api
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    UserStateChangedEvent,
    cli,
    inference,
    llm,
)
from livekit.plugins import silero

load_dotenv()

logger = logging.getLogger("voicemail-detection-observer")
server = AgentServer()

MACHINE_CONFIDENCE_THRESHOLD = 0.7
MAX_DETECTION_TURNS = 2
NO_SPEECH_TIMEOUT = 10.0

VOICEMAIL_MESSAGE = (
    "Hi, this is Ava from Bright Dental calling to remind you about your upcoming "
    "appointment. Please call us back at your convenience to confirm. Thank you, "
    "and have a great day."
)

TurnVerdict = Literal["human", "machine", "pending"]


class VoicemailObserver:
    """Classify who answered from the opening turns, then stop listening."""

    def __init__(
        self,
        session: AgentSession,
        decision_model: llm.DecisionModel,
        job_context: JobContext,
    ) -> None:
        self._session = session
        self._decision_model = decision_model
        self._job_context = job_context
        self._greeting_turns: list[str] = []
        self._category: str | None = None
        self._no_speech_timer: asyncio.Task[None] | None = None
        self._machine_task: asyncio.Task[None] | None = None
        self._setup_listener()

    def _setup_listener(self) -> None:
        @self._session.on("user_state_changed")
        def on_user_state_changed(event: UserStateChangedEvent) -> None:
            if event.new_state == "speaking" and self._no_speech_timer is not None:
                self._no_speech_timer.cancel()
                self._no_speech_timer = None

    def start_no_speech_timer(self) -> None:
        self._no_speech_timer = asyncio.create_task(self._wait_for_speech())

    async def _wait_for_speech(self) -> None:
        await asyncio.sleep(NO_SPEECH_TIMEOUT)
        if self._category is not None:
            return
        # Leave the verdict open: a late greeting can still be a voicemail.
        logger.info("[VOICEMAIL OBSERVER] no speech detected, checking for a person")
        self._no_speech_timer = None
        await self._session.generate_reply(
            instructions="Nobody has spoken yet. Politely check whether anyone is there."
        )

    async def classify_turn(self, text: str) -> TurnVerdict:
        """Return whether the agent may reply to this turn."""
        if self._category is not None:
            return "human" if self._category in ("human", "uncertain") else "machine"

        self._greeting_turns.append(text.strip())
        greeting = "\n".join(self._greeting_turns)
        try:
            category, greeting_complete = await self._evaluate_greeting(greeting)
        except Exception:
            # A model or network failure must never hang up on a possible human.
            logger.exception("[VOICEMAIL OBSERVER] evaluation failed")
            category, greeting_complete = "uncertain", True

        if category in ("human", "uncertain"):
            self._resolve(category)
            return "human"

        out_of_turns = len(self._greeting_turns) >= MAX_DETECTION_TURNS
        if category == "voicemail" and not greeting_complete and not out_of_turns:
            # Wait for the rest of the greeting instead of talking over it.
            return "pending"

        self._resolve(category)
        self._machine_task = asyncio.create_task(self._handle_machine(category))
        return "machine"

    async def _evaluate_greeting(self, greeting: str) -> tuple[str, bool]:
        response = await self._decision_model.evaluate(
            input=f"CALLEE AUDIO TRANSCRIPT:\n{greeting}",
            decisions={
                "answer_type": llm.Choice(
                    prompt=(
                        "An outbound phone call was just answered. Classify what answered "
                        "based on the transcript of the callee's opening audio. Voicemail "
                        "greetings usually name a person or number, say they are unavailable, "
                        "and ask the caller to leave a message after the tone. IVR menus list "
                        "options such as 'press 1 for'. Unavailable lines say the mailbox is "
                        "full, not set up, or the number cannot be reached. A human typically "
                        "gives a short greeting such as 'Hello?' or 'This is Sam' and waits."
                    ),
                    options={
                        "human": "A live person answered.",
                        "voicemail": "A voicemail greeting where a message can be left.",
                        "ivr": "An automated phone menu or IVR system.",
                        "unavailable": "Mailbox full, not set up, or the line is unreachable.",
                        "uncertain": "Not enough evidence to classify confidently.",
                    },
                ),
                "machine_probability": llm.Probability(
                    prompt=(
                        "The callee's opening audio is a recorded or automated message "
                        "(voicemail, IVR menu, or carrier announcement) rather than a live person."
                    )
                ),
                "greeting_completeness": llm.Score(
                    prompt=(
                        "If this is a recorded greeting, score whether it appears finished, "
                        "for example ending with 'leave a message', 'after the tone', or a beep."
                    ),
                    levels=[
                        "The greeting is cut off or still in progress.",
                        "The greeting sounds finished.",
                    ],
                ),
            },
        )

        if error := response.errors.get("answer_type"):
            logger.error("[VOICEMAIL OBSERVER] decision error: %s", error)
            return "uncertain", True

        result = response.results["answer_type"]
        if not isinstance(result, llm.ChoiceResult):
            logger.error("[VOICEMAIL OBSERVER] unexpected result kind: %s", result.kind)
            return "uncertain", True

        confidence = (
            result.probabilities.get(result.value)
            if result.probabilities is not None
            else None
        )
        machine_probability = response.results.get("machine_probability")
        if confidence is None and isinstance(
            machine_probability, llm.ProbabilityResult
        ):
            confidence = machine_probability.value

        completeness = response.results.get("greeting_completeness")
        greeting_complete = (
            completeness.value >= 0.5
            if isinstance(completeness, llm.ScoreResult)
            else True
        )

        category = result.value
        is_machine = category in ("voicemail", "ivr", "unavailable")
        if is_machine and (
            confidence is None or confidence < MACHINE_CONFIDENCE_THRESHOLD
        ):
            category = "uncertain"

        debug_trace = {
            "input": {
                "decision": "answer_type",
                "greeting_transcript": greeting,
                "greeting_turns": len(self._greeting_turns),
            },
            "response": response.model_dump(mode="json"),
            "policy_state": {
                "selected": result.value,
                "confidence": confidence,
                "threshold": MACHINE_CONFIDENCE_THRESHOLD,
                "final_category": category,
                "greeting_complete": greeting_complete,
            },
        }
        logger.debug(
            "[VOICEMAIL OBSERVER] DecisionModel trace:\n%s",
            json.dumps(debug_trace, indent=2, sort_keys=True),
        )
        return category, greeting_complete

    def _resolve(self, category: str) -> None:
        if self._no_speech_timer is not None:
            self._no_speech_timer.cancel()
            self._no_speech_timer = None
        if self._category is None:
            logger.info("[VOICEMAIL OBSERVER] verdict: %s", category)
            self._category = category

    async def _handle_machine(self, category: str) -> None:
        if category == "voicemail":
            message = self._session.say(VOICEMAIL_MESSAGE, allow_interruptions=False)
            await message.wait_for_playout()
        else:
            logger.info("[VOICEMAIL OBSERVER] %s detected, hanging up", category)
        await self._disconnect_everyone()

    async def _disconnect_everyone(self) -> None:
        try:
            # Deleting the room disconnects the callee too; session shutdown alone does not.
            await self._job_context.api.room.delete_room(
                api.DeleteRoomRequest(room=self._job_context.room.name)
            )
        except Exception:
            logger.exception("[VOICEMAIL OBSERVER] failed to delete room")
        finally:
            self._session.shutdown(drain=False)


class ReminderAgent(Agent):
    def __init__(self, observer: VoicemailObserver) -> None:
        super().__init__(
            instructions=(
                "You are Ava, a friendly assistant calling on behalf of Bright Dental to "
                "remind the patient about an upcoming appointment. When the person answers, "
                "introduce yourself, mention the appointment reminder, and ask whether they "
                "can confirm. Keep replies short and natural for a phone call."
            )
        )
        self._observer = observer

    async def on_enter(self) -> None:
        # On an outbound call the callee speaks first; stay silent until classified.
        self._observer.start_no_speech_timer()

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        text = new_message.text_content
        if not text or not text.strip():
            return
        if await self._observer.classify_turn(text) != "human":
            raise llm.StopResponse()


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name="voicemail-detection-observer")
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

    observer = VoicemailObserver(
        session=session,
        decision_model=inference.DecisionModel("livekit/clef-omni"),
        job_context=ctx,
    )

    await session.start(agent=ReminderAgent(observer), room=ctx.room)
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
