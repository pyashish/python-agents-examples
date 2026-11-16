"""Outbound job (Stuut pattern, services/livekit-service/.../jobs/outbound.py).

The backend only booked a dispatch. This job joins an empty room, starts the session,
then dials the callee over an outbound SIP trunk. Answering-machine detection (AMD)
listens to whoever picks up and decides the agent's first move.
"""

from __future__ import annotations

import logging

from google.protobuf.duration_pb2 import Duration
from livekit import api
from livekit.agents import AMD, AgentSession, AMDCategory, JobContext, room_io

from shared.contract import CALLEE_IDENTITY, CallConfig

from .after_call import make_after_call
from .control_channel import attach_control_channel
from .controllers import attach_close_reason, attach_silence_watchdog, start_max_duration_timer
from .phone_agent import PhoneAgent
from .pipeline import build_session
from .recording import start_recording
from .state import CallState
from .telephony import hang_up

logger = logging.getLogger("field-patterns.outbound")

# LiveKit caps ringing_timeout at 80 s.
RINGING_TIMEOUT_S = 60


async def run_outbound(ctx: JobContext, config: CallConfig) -> None:
    # ---------------------------------------------------------------- BEFORE
    await ctx.connect()
    call = CallState(config=config, caller_number=config.to_number)
    session = build_session(config, vad_model=ctx.proc.userdata["vad"], userdata=call)

    # ---------------------------------------------------------------- DURING (wired before start)
    attach_silence_watchdog(session, call)
    attach_close_reason(session, call)
    stop_control = attach_control_channel(ctx.room, session, call)
    max_duration = start_max_duration_timer(call)

    # ---------------------------------------------------------------- AFTER
    ctx.add_shutdown_callback(make_after_call(session, call, cleanup=[stop_control, max_duration.cancel]))

    # ---------------------------------------------------------------- START (silent: listen first)
    agent = PhoneAgent(config, greet_on_enter=False)
    console = ctx.is_fake_job() or config.mock
    # Listen only to the callee, so AMD and STT never hear anyone else.
    callee_only = {} if console else {"participant_identity": CALLEE_IDENTITY}
    room_options = room_io.RoomOptions(
        # Keep the room when the agent leaves: a warm transfer needs it to outlive the
        # agent (Stuut). Hang-ups delete it explicitly instead (telephony.hang_up).
        delete_room_on_close=False,
        **callee_only,
    )
    await session.start(agent=agent, room=ctx.room, room_options=room_options)
    call.recording_path = await start_recording(ctx, config)

    if console:
        # No phone to dial in console mode or a mock run: act as if a person answered.
        call.answered_by = "console"
        agent.greet()
        return

    # ---------------------------------------------------------------- DIAL + AMD
    # Open AMD before dialing so the callee's first words are not missed.
    async with AMD(session, llm=config.llm_model, participant_identity=CALLEE_IDENTITY, ivr_detection=False) as detector:
        try:
            await ctx.api.sip.create_sip_participant(
                api.CreateSIPParticipantRequest(
                    sip_trunk_id=config.sip_trunk_id or "",
                    sip_call_to=config.to_number or "",
                    sip_number=config.from_number or "",  # caller ID; must be allowed on the trunk
                    room_name=ctx.room.name,
                    participant_identity=CALLEE_IDENTITY,
                    wait_until_answered=True,
                    ringing_timeout=Duration(seconds=RINGING_TIMEOUT_S),
                    max_call_duration=Duration(seconds=config.max_duration_s),
                )
            )
        except api.TwirpError as e:
            # Busy, no answer, invalid number, carrier block. Stuut maps SIP codes
            # (404, 410, 484, 604) to "dial failed" and Twilio 32203 to "blocked".
            call.end_with(f"dial-failed:{e.metadata.get('sip_status_code', e.code)}")
            ctx.shutdown(reason=call.ended_reason or "dial-failed")
            return

        # wait_until_answered returns before the participant is always visible in the room.
        await ctx.wait_for_participant(identity=CALLEE_IDENTITY)
        verdict = await detector.execute()
        call.answered_by = verdict.category.value
        logger.info("amd verdict", extra={"category": verdict.category.value, "reason": verdict.reason})
        # Act inside the AMD context, where AMD still holds back the session's auto-reply.
        await act_on_amd_verdict(verdict.category, session, agent, call)


async def act_on_amd_verdict(
    category: AMDCategory, session: AgentSession[CallState], agent: PhoneAgent, call: CallState
) -> None:
    if category in (AMDCategory.HUMAN, AMDCategory.UNCERTAIN):
        agent.greet()
    elif category == AMDCategory.MACHINE_VM:
        if call.config.voicemail_message:
            await session.say(call.config.voicemail_message, allow_interruptions=False)
        call.end_with("left-voicemail")
        await hang_up("left-voicemail")
    elif category == AMDCategory.MACHINE_UNAVAILABLE:
        call.end_with("mailbox-unavailable")
        await hang_up("mailbox-unavailable")
    elif category == AMDCategory.MACHINE_IVR:
        # Stay quiet and let the model navigate with send_dtmf. The menu's first read is
        # already in the transcript, so ask for the reply AMD held back (Stuut).
        call.in_phone_menu = True
        session.generate_reply()
