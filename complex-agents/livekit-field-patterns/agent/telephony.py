"""Phone-call actions that need the job context: hanging up and pressing keys."""

from __future__ import annotations

import logging

from livekit.agents import get_job_context

logger = logging.getLogger("field-patterns.telephony")

DTMF_CODES = {"*": 10, "#": 11}


async def hang_up(reason: str) -> None:
    """End the call by deleting the room.

    session.shutdown() alone stops the agent but leaves the phone leg connected (and
    billed). Deleting the room drops every participant. In console mode there is no
    real room, so we shut the job down instead.
    """
    job_ctx = get_job_context()
    logger.info("hanging up", extra={"reason": reason})
    if job_ctx.is_fake_job():
        job_ctx.shutdown(reason=reason)
        return
    await job_ctx.delete_room()


async def press_keys(digits: str) -> None:
    """Send DTMF tones into the call (Strada: build-tools.ts dtmfTool).

    The SDK also ships livekit.agents.beta.tools.send_dtmf_events, which Stuut uses.
    """
    room = get_job_context().room
    for digit in digits:
        code = DTMF_CODES.get(digit, int(digit) if digit.isdigit() else -1)
        await room.local_participant.publish_dtmf(code=code, digit=digit)
