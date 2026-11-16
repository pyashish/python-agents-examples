"""DURING the call: small controllers that subscribe to AgentSession events.

Both customers keep call behavior out of the prompt and in code like this:
Strada has an inactivity watchdog and end-call phrases; Stuut has re-engagement,
IVR pickup checks and transcription recovery. Attach them before session.start()
so no event is missed.

Session events used here: "user_state_changed", "close".
"""

from __future__ import annotations

import asyncio
import logging

from livekit.agents import AgentSession, CloseEvent, UserStateChangedEvent

from .state import CallState
from .telephony import hang_up

logger = logging.getLogger("field-patterns.controllers")


def attach_silence_watchdog(session: AgentSession[CallState], call: CallState) -> None:
    """Re-prompt a silent caller once, then hang up.

    user_away_timeout (set in pipeline.py) turns the user "away" after
    silence_reprompt_s. Any later state change means the caller is back.
    """
    pending: asyncio.Task | None = None

    async def hang_up_if_still_silent() -> None:
        await asyncio.sleep(call.config.silence_hangup_s)
        call.end_with("silence-timeout")
        await hang_up("silence-timeout")

    def cancel() -> None:
        nonlocal pending
        if pending is not None:
            pending.cancel()
            pending = None

    @session.on("user_state_changed")
    def _on_user_state(ev: UserStateChangedEvent) -> None:
        nonlocal pending
        cancel()
        # Phone menus go quiet on purpose; don't talk over them (Stuut keeps in_ivr for this).
        if ev.new_state != "away" or call.in_phone_menu:
            return
        session.generate_reply(instructions="The caller has gone quiet. Briefly ask if they are still there.")
        pending = asyncio.create_task(hang_up_if_still_silent())

    @session.on("close")
    def _on_close(_ev: CloseEvent) -> None:
        cancel()


def attach_close_reason(session: AgentSession[CallState], call: CallState) -> None:
    """If nothing more specific ended the call, record the session's own close reason."""

    @session.on("close")
    def _on_close(ev: CloseEvent) -> None:
        call.end_with(f"session-closed:{getattr(ev.reason, 'value', ev.reason)}")


def start_max_duration_timer(call: CallState) -> asyncio.Task:
    """Hang up when the call reaches max_duration_s.

    Cancel this task from the shutdown callback; never await it in the entrypoint.
    Stuut found that awaiting it held every job open until the 15 s shutdown grace.
    """

    async def timer() -> None:
        await asyncio.sleep(call.config.max_duration_s)
        call.end_with("max-duration")
        await hang_up("max-duration")

    return asyncio.create_task(timer())
