"""Outbound SIP / telephony voice agent for LiveKit Cloud.

Unlike inbound sip-agent (which waits for a caller), this agent is dispatched
with a phone number in job metadata, then dials that number via an outbound
SIP trunk using CreateSIPParticipant.

Run locally (from this directory):
    lk agent console sip_outbound_agent.py
    lk agent dev sip_outbound_agent.py

Place a call (with the agent running in another terminal):
    python make_call.py +15551234567

Deploy:
    lk agent create   # first time, from this directory
    lk agent deploy

Requires SIP_OUTBOUND_TRUNK_ID in .env (LiveKit Phone Numbers are inbound-only).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from livekit import api, rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    inference,
    room_io,
)
from livekit.agents.beta.tools import EndCallTool
from livekit.plugins import noise_cancellation, silero

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

logger = logging.getLogger("sip-outbound-agent")
logger.setLevel(logging.INFO)

AGENT_NAME = "sip-outbound-agent"


def parse_phone_number(metadata_raw: str | None) -> str | None:
    """Accept plain E.164 string or JSON {"phone_number": "+1..."}."""
    if not metadata_raw:
        return None
    raw = metadata_raw.strip()
    if not raw:
        return None
    if raw.startswith("{"):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("job metadata is not valid JSON: %r", raw)
            return None
        phone = data.get("phone_number") or data.get("phone")
        return str(phone).strip() if phone else None
    return raw


class OutboundAssistant(Agent):
    def __init__(self) -> None:
        end_call = EndCallTool(
            extra_description=(
                "Call this when the callee says goodbye, asks to hang up, "
                "or the conversation is finished."
            ),
            end_instructions="Say a brief goodbye and thank them for their time.",
            delete_room=True,
        )
        super().__init__(
            instructions=(
                "You are a helpful phone assistant placing an outbound call. "
                "Keep replies short and clear for a phone call — avoid long "
                "lists and markdown. Introduce yourself briefly, help with "
                "their question, and when they are done use the end_call tool "
                "to hang up."
            ),
            tools=end_call.tools,
        )


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name=AGENT_NAME)
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    phone_number = parse_phone_number(ctx.job.metadata)
    outbound_trunk_id = os.getenv("SIP_OUTBOUND_TRUNK_ID")

    if not phone_number:
        logger.error("missing phone_number in job metadata; shutting down")
        ctx.shutdown("missing phone_number")
        return

    if not outbound_trunk_id or not outbound_trunk_id.startswith("ST_"):
        logger.error("SIP_OUTBOUND_TRUNK_ID is missing or invalid")
        ctx.shutdown("missing outbound trunk")
        return

    participant_identity = f"sip_{phone_number}"
    logger.info(
        "outbound agent joining room=%s dialing=%s trunk=%s",
        ctx.room.name,
        phone_number,
        outbound_trunk_id,
    )

    session = AgentSession(
        stt=inference.STT(model="deepgram/nova-3-general"),
        llm=inference.LLM(model="openai/gpt-4.1-mini"),
        tts=inference.TTS(
            model="cartesia/sonic-3",
            voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        ),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    await session.start(
        agent=OutboundAssistant(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=noise_cancellation.BVCTelephony(),
            ),
        ),
    )
    await ctx.connect()

    # Dial the callee; block until they answer (or the call fails).
    try:
        await ctx.api.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                room_name=ctx.room.name,
                sip_trunk_id=outbound_trunk_id,
                sip_call_to=phone_number,
                participant_identity=participant_identity,
                wait_until_answered=True,
                krisp_enabled=True,
            ),
            timeout=60,
        )
    except (api.SipCallError, asyncio.TimeoutError) as e:
        logger.info("call was not answered: %s", e)
        ctx.shutdown("call not answered")
        return

    participant = ctx.room.remote_participants.get(participant_identity)
    if participant is None:
        # Race: call may drop between answer and join.
        participant = await ctx.wait_for_participant(identity=participant_identity)

    attrs = participant.attributes or {}
    logger.info(
        "SIP participant connected identity=%s call_id=%s status=%s phone=%s",
        participant.identity,
        attrs.get("sip.callID"),
        attrs.get("sip.callStatus"),
        attrs.get("sip.phoneNumber"),
    )

    @ctx.room.on("participant_disconnected")
    def on_participant_disconnected(p: rtc.RemoteParticipant):
        if p.identity == participant_identity:
            logger.info("callee hung up; shutting down")
            ctx.shutdown("callee disconnected")

    # Greet only after the callee answers (not while ringing).
    session.generate_reply(
        instructions=(
            "The person just answered the phone. Introduce yourself briefly "
            "and ask how you can help."
        )
    )


if __name__ == "__main__":
    cli.run_app(server)
