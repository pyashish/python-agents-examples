"""Inbound job (Strada pattern, apps/voice-agent/src/main.ts).

By the time this runs, the backend has already answered Twilio and bridged the call
into the room. The agent never dials: it waits for the participant named "caller",
then talks first.
"""

from __future__ import annotations

from livekit.agents import JobContext, room_io

from shared.contract import CALLER_IDENTITY, CallConfig

from .after_call import make_after_call
from .control_channel import attach_control_channel
from .controllers import attach_close_reason, attach_silence_watchdog, start_max_duration_timer
from .phone_agent import PhoneAgent
from .pipeline import build_session
from .recording import start_recording
from .state import CallState


async def run_inbound(ctx: JobContext, config: CallConfig) -> None:
    # ---------------------------------------------------------------- BEFORE
    await ctx.connect()
    call = CallState(config=config)

    console = ctx.is_fake_job()
    if not console:
        # The phone leg is already in the room or about to join. Its attributes were
        # set by the backend in connect_twilio_call (backend/before_call.py).
        caller = await ctx.wait_for_participant(identity=CALLER_IDENTITY)
        call.caller_number = caller.attributes.get("sip.phoneNumber")

    session = build_session(config, vad_model=ctx.proc.userdata["vad"], userdata=call)

    # ---------------------------------------------------------------- DURING (wired before start)
    attach_silence_watchdog(session, call)
    attach_close_reason(session, call)
    stop_control = attach_control_channel(ctx.room, session, call)
    max_duration = start_max_duration_timer(call)

    # ---------------------------------------------------------------- AFTER
    ctx.add_shutdown_callback(make_after_call(session, call, cleanup=[stop_control, max_duration.cancel]))

    # ---------------------------------------------------------------- START
    # Listen only to the phone leg; console mode has no participant to point at.
    caller_only = {} if console else {"participant_identity": CALLER_IDENTITY}
    room_options = room_io.RoomOptions(
        # When the session closes, delete the room so the phone leg drops too.
        delete_room_on_close=True,
        **caller_only,
    )
    await session.start(agent=PhoneAgent(config, greet_on_enter=True), room=ctx.room, room_options=room_options)
    call.recording_path = await start_recording(ctx, config)
