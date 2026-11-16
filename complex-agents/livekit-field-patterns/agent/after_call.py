"""AFTER the call, agent side: build the end-of-call report and deliver it.

This runs as a JobContext shutdown callback, the one place guaranteed to run when the
room closes. It has a deadline: the worker kills the job process
`shutdown_process_timeout` seconds after shutdown starts (60 s in agent/main.py), so
retries must finish well inside that. Stuut raised the limit to 75 s for this reason.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable, Iterable

import aiohttp
from livekit.agents import AgentSession

from shared.contract import REPORT_SIGNATURE_HEADER, CallReport, TranscriptLine, sign_body

from .state import CallState

logger = logging.getLogger("field-patterns.after_call")

ATTEMPTS = 3
REQUEST_TIMEOUT_S = 10


def build_call_report(call: CallState, history_items: Iterable) -> CallReport:
    transcript = [
        TranscriptLine(role=item.role, text=item.text_content)
        for item in history_items
        if getattr(item, "type", None) == "message" and item.role in ("user", "assistant") and item.text_content
    ]
    return CallReport(
        call_id=call.config.call_id,
        direction=call.config.direction,
        ended_reason=call.ended_reason or "unknown",
        answered_by=call.answered_by,
        duration_s=call.duration_s(),
        recording_path=call.recording_path,
        transcript=transcript,
    )


async def deliver_report(report: CallReport, *, url: str | None, mock: bool) -> bool:
    """POST the signed report with bounded retries. Returns True when the backend accepted it."""
    body = report.model_dump_json().encode()
    if mock or not url:
        logger.info("call report not sent (mock or no report_url)", extra={"report": report.model_dump()})
        return False

    headers = {
        "content-type": "application/json",
        REPORT_SIGNATURE_HEADER: sign_body(body, os.environ["CALL_REPORT_SECRET"]),
    }
    async with aiohttp.ClientSession() as http:
        for attempt in range(ATTEMPTS):
            try:
                async with http.post(
                    url, data=body, headers=headers, timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S)
                ) as resp:
                    if resp.status < 300:
                        return True
                    if resp.status < 500:
                        # A 4xx won't succeed on retry.
                        logger.error("call report rejected", extra={"status": resp.status})
                        return False
            except (aiohttp.ClientError, TimeoutError):
                logger.warning("call report delivery failed", extra={"attempt": attempt + 1})
            await asyncio.sleep(2**attempt)
    logger.error("call report not delivered after retries", extra={"call_id": report.call_id})
    return False


def make_after_call(
    session: AgentSession[CallState],
    call: CallState,
    *,
    cleanup: list[Callable[[], object]],
) -> Callable[[str], Awaitable[None]]:
    """Return the shutdown callback for this call."""

    async def after_call(shutdown_reason: str) -> None:
        for undo in cleanup:
            undo()
        call.end_with(shutdown_reason)
        report = build_call_report(call, session.history.items)
        await deliver_report(report, url=call.config.report_url, mock=call.config.mock)

    return after_call
