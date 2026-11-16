"""Per-call state, stored as AgentSession.userdata so tools and controllers can reach it.

Stuut does the same with AgentSession[SessionInfo]; Strada keeps a CallContext object.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from shared.contract import CallConfig


@dataclass
class CallState:
    config: CallConfig
    started_at: float = field(default_factory=time.monotonic)
    caller_number: str | None = None
    answered_by: str | None = None  # AMD verdict on outbound calls
    ended_reason: str | None = None  # first writer wins; see end_with()
    in_phone_menu: bool = False  # keeps the silence watchdog quiet while navigating an IVR
    recording_path: str | None = None
    llm_first_token_s: list[float] = field(default_factory=list)

    def end_with(self, reason: str) -> None:
        """Record why the call ended, unless an earlier, more specific reason is already set."""
        if self.ended_reason is None:
            self.ended_reason = reason

    def duration_s(self) -> float:
        return round(time.monotonic() - self.started_at, 1)
