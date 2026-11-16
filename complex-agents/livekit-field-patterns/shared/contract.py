"""The contract between the customer's backend and the agent worker.

Both customers keep this in one shared package so the two sides cannot drift:
- Strada: packages/voice-gateway (zod schemas for control commands and handoffs)
- Stuut:  src/libs/stuut-livekit (pydantic OutboundCallRequestSchema, SessionInfo)

Everything the agent knows about a call arrives as CallConfig JSON in the dispatch
metadata. Mid-call commands and handoffs use the shapes below, and the end-of-call
report comes back as a signed CallReport.
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter

# The name the worker registers under. A dispatch must use the same name.
# Strada: "inbound-generic". Stuut: "stuut-agent".
AGENT_NAME = "field-patterns-agent"

# Identity the inbound phone leg joins the room with (Strada uses "caller").
CALLER_IDENTITY = "caller"

# Identity the outbound callee joins with. Stuut uses "user-<E.164>", which puts
# the phone number into every log line; a fixed identity avoids that.
CALLEE_IDENTITY = "callee"

# Data-packet topic for backend -> agent commands (Strada: "strada.control").
CONTROL_TOPIC = "field.control"

# Header carrying the HMAC of the end-of-call report body.
REPORT_SIGNATURE_HEADER = "X-Call-Signature"


class CallConfig(BaseModel):
    """Everything the agent needs for one call. Built by the backend, sent as dispatch metadata."""

    call_id: str
    direction: Literal["inbound", "outbound"]
    instructions: str
    greeting: str | None = None
    voicemail_message: str | None = None

    # Outbound dialing (Stuut). The trunk is a LiveKit outbound SIP trunk ID.
    to_number: str | None = None
    from_number: str | None = None
    sip_trunk_id: str | None = None

    # Models, as LiveKit Inference model strings. Each has a backup.
    stt_model: str = "deepgram/nova-3"
    stt_fallback_model: str = "assemblyai/universal-streaming"
    llm_model: str = "openai/gpt-4.1-mini"
    llm_backup_model: str = "openai/gpt-4.1"
    tts_model: str = "cartesia/sonic-3"
    tts_voice: str = "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
    language: str = "en"

    # Call policy.
    max_duration_s: int = 600
    silence_reprompt_s: float = 10.0
    silence_hangup_s: float = 20.0
    pronunciations: dict[str, str] = Field(default_factory=dict)

    # Where results go. With mock=True nothing leaves the worker.
    report_url: str | None = None
    tool_server_url: str | None = None
    mock: bool = False


def room_name_for(config: CallConfig) -> str:
    """Room name = call ID, so LiveKit logs, recordings and webhooks join back to the call.

    Strada names inbound rooms with the bare call ID. Stuut prefixes outbound rooms
    with "outbound_" and routes jobs by that prefix.
    """
    if config.direction == "inbound":
        return config.call_id
    return f"outbound_{config.call_id}"


# ---------------------------------------------------------------------------
# Backend -> agent control commands (sent as data packets on CONTROL_TOPIC).
# Strada keeps VAPI's Live Call Control shapes; these are a trimmed version.
# ---------------------------------------------------------------------------


class SayCommand(BaseModel):
    type: Literal["say"] = "say"
    # Bounded so a compromised channel cannot stream unbounded speech into a call.
    text: str = Field(min_length=1, max_length=1000)
    end_call_after: bool = False


class AddMessageCommand(BaseModel):
    type: Literal["add_message"] = "add_message"
    role: Literal["system", "user", "assistant"]
    content: str = Field(max_length=4000)
    reply: bool = False


class EndCallCommand(BaseModel):
    type: Literal["end_call"] = "end_call"


ControlCommand = Annotated[
    SayCommand | AddMessageCommand | EndCallCommand,
    Field(discriminator="type"),
]
CONTROL_COMMAND = TypeAdapter(ControlCommand)


class Handoff(BaseModel):
    """Switch the call to a new set of instructions mid-call.

    Travels in room metadata as {"handoff": {...}}, not as a data packet: a full
    assistant config can exceed the ~15 KiB data-packet limit (Strada's reason).
    The id makes it idempotent, since metadata can be re-delivered.
    """

    id: str
    instructions: str
    greeting: str | None = None
    keep_history: bool = True


class RoomMetadata(BaseModel):
    handoff: Handoff | None = None


# ---------------------------------------------------------------------------
# Agent -> backend end-of-call report.
# ---------------------------------------------------------------------------


class TranscriptLine(BaseModel):
    role: str
    text: str


class CallReport(BaseModel):
    call_id: str
    direction: Literal["inbound", "outbound"]
    ended_reason: str
    answered_by: str | None = None
    duration_s: float
    recording_path: str | None = None
    transcript: list[TranscriptLine] = Field(default_factory=list)


def sign_body(body: bytes, secret: str) -> str:
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def verify_body(body: bytes, signature: str, secret: str) -> bool:
    return hmac.compare_digest(sign_body(body, secret), signature)
