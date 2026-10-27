"""Inbound SIP agent with a keypad language menu.

Callers change language in either of two ways:

- Press a key. The room ``sip_dtmf_received`` handler switches immediately.
- Ask out loud. The model calls ``switch_language``.

Both paths retarget Deepgram Nova-3 and Cartesia Sonic, then update the
instructions so later replies stay in the chosen language.

Menu:
    1 English
    2 Spanish
    3 Hindi
    4 Tamil
    0 or * repeats the menu

Run locally (from this directory):
    lk agent console sip_dtmf_agent.py
    lk agent dev sip_dtmf_agent.py

Point a LiveKit Phone Number dispatch rule at agent_name: sip-dtmf-agent.
The Agent Console DTMF pane can simulate keypad presses.
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    TurnHandlingOptions,
    cli,
    function_tool,
    inference,
    room_io,
)
from livekit.agents.beta.tools import EndCallTool
from livekit.plugins import noise_cancellation, silero

_AGENT_DIR = Path(__file__).resolve().parent
load_dotenv(_AGENT_DIR.parents[1] / ".env")
load_dotenv(_AGENT_DIR / ".env", override=True)

logger = logging.getLogger("sip-dtmf-agent")
logger.setLevel(logging.INFO)

AGENT_NAME = "sip-dtmf-agent"
STT_MODEL = "deepgram/nova-3"

LanguageName = Literal["english", "spanish", "hindi", "tamil"]


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Set {name} in simple-ones/sip-agent/.env")
    return value


# Coughs and room noise: require 0.8s of speech and 2 real words before
# yielding, and resume quickly if STT hears nothing.
TURN_HANDLING = TurnHandlingOptions(
    interruption={
        "mode": "adaptive",
        "min_duration": 0.8,
        "min_words": 2,
        "resume_false_interruption": True,
        "false_interruption_timeout": 1.0,
    },
    preemptive_generation={"enabled": True},
)


def expressive_options() -> dict:
    # Cartesia Sonic can vary emotion, pace, volume, and pauses. It does not
    # produce laughs or sighs, so those sounds stay off.
    return {
        "speech_steering": {
            "disfluencies": True,
            "pace": "normal",
        },
        "tts_instructions_append": _required("TTS_DELIVERY"),
    }


@dataclass(frozen=True)
class Language:
    code: str
    name: str
    greeting: str
    menu: str


# Digits are the keypad. Codes are what Deepgram Nova-3 and Cartesia Sonic-3
# both accept (en, es, hi, ta).
LANGUAGES: dict[str, Language] = {
    "1": Language(
        code="en",
        name="English",
        greeting="Okay! I'm talking in English now. What do you wanna talk about?",
        menu=(
            "Press 1 for English, 2 for Spanish, 3 for Hindi, or 4 for Tamil. "
            "Or just tell me."
        ),
    ),
    "2": Language(
        code="es",
        name="Spanish",
        greeting="¡Listo! Ahora hablo español. ¿De qué quieres hablar?",
        menu="Pulsa 1 para inglés, 2 para español, 3 para hindi, o 4 para tamil.",
    ),
    "3": Language(
        code="hi",
        name="Hindi",
        greeting="अच्छा! अब मैं हिंदी में बात करूँगा। क्या बात करें?",
        menu="अंग्रेज़ी के लिए 1, स्पेनिश के लिए 2, हिंदी के लिए 3, या तमिल के लिए 4 दबाओ।",
    ),
    "4": Language(
        code="ta",
        name="Tamil",
        greeting="சரி! இப்போ நான் தமிழ்ல பேசுறேன். என்ன பேசலாம்?",
        menu="இங்கிலீஷ்க்கு 1, ஸ்பானிஷ்க்கு 2, ஹிந்திக்கு 3, தமிழுக்கு 4 அழுத்து.",
    ),
}
BY_CODE = {lang.code: lang for lang in LANGUAGES.values()}
BY_NAME = {lang.name.lower(): lang for lang in LANGUAGES.values()}
MENU_DIGITS = {"0", "*"}


class SipDtmfAssistant(Agent):
    def __init__(self) -> None:
        end_call = EndCallTool(
            extra_description=(
                "Call this when the caller says goodbye, asks to hang up, "
                "or the conversation is finished."
            ),
            end_instructions="Say a brief goodbye in the current language and thank them for calling.",
            delete_room=True,
        )
        super().__init__(
            instructions=self._language_instructions(BY_CODE["en"]),
            tools=end_call.tools,
        )
        self._language = BY_CODE["en"]
        self._switch_lock = asyncio.Lock()

    def _language_instructions(self, language: Language) -> str:
        return (
            f"{_required('AGENT_IDENTITY')} {_required('AGENT_PROFILE')} "
            f"{_required('AGENT_STYLE')} "
            f"You are speaking {language.name}. Reply only in {language.name}. "
            "If the caller asks to change language, names English, Spanish, Hindi, "
            "or Tamil, or says a menu number, call switch_language. "
            "Keypad: 1 English, 2 Spanish, 3 Hindi, 4 Tamil. "
            "0 or star repeats the menu. "
            "When the user says bye or is finished, use the end_call tool."
        )

    async def on_enter(self):
        self.session.generate_reply(
            instructions=(
                "Greet the caller briefly as Babu and tell them the language menu: "
                "press 1 for English, 2 for Spanish, 3 for Hindi, or 4 for Tamil. "
                "They can also just say the language."
            )
        )

    async def on_dtmf(self, digit: str) -> None:
        """Keypad fast path. Does not wait for the model."""
        try:
            language = LANGUAGES.get(digit)
            if language is None:
                if digit not in MENU_DIGITS:
                    logger.info("ignoring DTMF digit=%s", digit)
                await self.session.interrupt()
                self.session.say(self._language.menu, allow_interruptions=True)
                return

            await self._switch(language, speak=True)
        except Exception:
            logger.exception("DTMF handler failed digit=%s", digit)

    @function_tool
    async def switch_language(self, language: LanguageName) -> str:
        """Switch the conversation language when the caller asks.

        Use english, spanish, hindi, or tamil. Map spoken numbers too:
        1 English, 2 Spanish, 3 Hindi, 4 Tamil.
        """
        selected = BY_NAME[language]
        return await self._switch(selected, speak=False)

    async def _switch(self, language: Language, *, speak: bool) -> str:
        async with self._switch_lock:
            if language.code == self._language.code:
                logger.info("already speaking %s", language.name)
                return f"Already speaking {language.name}. Stay in {language.name}."

            logger.info("switching language %s -> %s", self._language.code, language.code)
            await self.session.interrupt()

            # A new STT instance reconnects Nova-3 with the language. A mid-call
            # update_options on the open Nova stream is not applied.
            self.update_options(stt=inference.STT(model=STT_MODEL, language=language.code))
            if self.session.tts is not None:
                self.session.tts.update_options(language=language.code)

            self._language = language
            await self.update_instructions(self._language_instructions(language))

        if speak:
            self.session.say(language.greeting, allow_interruptions=True)
            return f"Now speaking {language.name}."

        return (
            f"Now speaking {language.name}. "
            f"Greet the caller in one short {language.name} sentence, then continue."
        )


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name=AGENT_NAME)
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info("sip dtmf agent joining room=%s", ctx.room.name)

    session = AgentSession(
        stt=inference.STT(model=STT_MODEL, language="en"),
        llm=inference.LLM(model="openai/gpt-4.1-mini"),
        tts=inference.TTS(
            model=_required("TTS_MODEL"),
            voice=_required("TTS_VOICE"),
            language="en",
        ),
        vad=ctx.proc.userdata["vad"],
        turn_handling=TURN_HANDLING,
        expressive=expressive_options(),
    )

    agent = SipDtmfAssistant()

    await session.start(
        agent=agent,
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=noise_cancellation.BVCTelephony(),
            ),
        ),
    )

    dtmf_tasks: set[asyncio.Task[None]] = set()

    @ctx.room.on("sip_dtmf_received")
    def on_dtmf(ev: rtc.SipDTMF) -> None:
        logger.info("DTMF digit=%s code=%s", ev.digit, ev.code)
        task = asyncio.create_task(agent.on_dtmf(ev.digit))
        dtmf_tasks.add(task)
        task.add_done_callback(dtmf_tasks.discard)

    await ctx.connect()

    def _log_sip_participant(participant: rtc.RemoteParticipant) -> None:
        if participant.kind != rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
            return
        attrs = participant.attributes or {}
        logger.info(
            "SIP participant identity=%s call_id=%s status=%s phone=%s trunk=%s",
            participant.identity,
            attrs.get("sip.callID"),
            attrs.get("sip.callStatus"),
            attrs.get("sip.phoneNumber"),
            attrs.get("sip.trunkPhoneNumber"),
        )

    for participant in ctx.room.remote_participants.values():
        _log_sip_participant(participant)

    @ctx.room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant):
        _log_sip_participant(participant)


if __name__ == "__main__":
    cli.run_app(server)
