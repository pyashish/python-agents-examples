"""Inbound SIP / telephony voice agent for LiveKit Cloud.

Answers phone calls routed via a LiveKit dispatch rule. No special SIP
dialing code is required for inbound — the SIP participant joins the room
and this agent greets them.

Run locally (from this directory):
    lk agent console sip_agent.py
    lk agent dev sip_agent.py

Deploy:
    lk agent deploy   # from this directory (simple-ones/sip-agent/)

Point a LiveKit Phone Number dispatch rule at agent_name: sip-agent
"""

import logging
import os
from pathlib import Path

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
    inference,
    room_io,
)
from livekit.agents.beta.tools import EndCallTool
from livekit.plugins import noise_cancellation, silero

_AGENT_DIR = Path(__file__).resolve().parent
load_dotenv(_AGENT_DIR.parents[1] / ".env")
load_dotenv(_AGENT_DIR / ".env", override=True)

logger = logging.getLogger("sip-agent")
logger.setLevel(logging.INFO)

AGENT_NAME = "sip-agent"


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


class SipAssistant(Agent):
    def __init__(self) -> None:
        end_call = EndCallTool(
            extra_description=(
                "Call this when the caller says goodbye, asks to hang up, "
                "or the conversation is finished."
            ),
            end_instructions="Say a brief goodbye and thank them for calling.",
            delete_room=True,
        )
        super().__init__(
            instructions=(
                f"{_required('AGENT_IDENTITY')} {_required('AGENT_PROFILE')} "
                f"{_required('AGENT_STYLE')} "
                "If the user asks for Jocey, use transfer_to_jocey. "
                "When the user says bye or is finished, use the end_call tool."
            ),
            tools=end_call.tools,
        )

    async def on_enter(self):
        self.session.generate_reply(
            instructions="Greet the caller briefly and ask how you can help."
        )


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name=AGENT_NAME)
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info("sip agent joining room=%s", ctx.room.name)

    session = AgentSession(
        stt=inference.STT(model="deepgram/nova-3-general"),
        llm=inference.LLM(model="openai/gpt-4.1-mini"),
        tts=inference.TTS(
            model=_required("TTS_MODEL"),
            voice=_required("TTS_VOICE"),
        ),
        vad=ctx.proc.userdata["vad"],
        turn_handling=TURN_HANDLING,
        expressive=expressive_options(),
    )

    @session.on("agent_false_interruption")
    def _on_false_interruption(ev) -> None:
        logger.info("false interruption, resumed=%s", ev.resumed)

    @session.on("overlapping_speech")
    def _on_overlap(ev) -> None:
        logger.info(
            "overlap is_interruption=%s probability=%.2f duration=%.2fs",
            ev.is_interruption,
            ev.probability,
            ev.total_duration,
        )

    await session.start(
        agent=SipAssistant(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                # Telephony-tuned noise cancellation for phone callers
                noise_cancellation=noise_cancellation.BVCTelephony(),
            ),
        ),
    )
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
