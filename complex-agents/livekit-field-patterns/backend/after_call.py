"""AFTER the call: results come back to the backend on two separate paths.

1. The agent's own end-of-call report, POSTed from its shutdown callback and signed
   with a shared secret. It carries what only the agent knows: why the call ended,
   who answered (human, voicemail, phone menu) and the transcript.
   Strada: VAPI-format "end-of-call-report". Stuut: signed "call_ended" webhook.

2. LiveKit's server webhooks (room_finished, egress_ended, ...), signed by LiveKit.
   They arrive even if the agent crashed, and egress_ended is when the recording file
   is ready. Stuut feeds these into Inngest and reconciles them with the agent report.
"""

from __future__ import annotations

from livekit import api

from shared.contract import CallReport, verify_body


class InvalidSignature(Exception):
    pass


def receive_call_report(body: bytes, signature: str, secret: str) -> CallReport:
    """Verify and parse the agent's report. Reject anything not signed with our secret."""
    if not verify_body(body, signature, secret):
        raise InvalidSignature("call report signature does not match")
    return CallReport.model_validate_json(body)


def receive_livekit_webhook(body: str, authorization: str, receiver: api.WebhookReceiver) -> str:
    """Verify a LiveKit server webhook (receiver = api.WebhookReceiver(api.TokenVerifier()))."""
    event = receiver.receive(body, authorization)
    return route_livekit_event(event)


def route_livekit_event(event: api.WebhookEvent) -> str:
    """Decide what each LiveKit event means for our call record. Returns a description."""
    if event.event == "room_finished":
        return f"call {event.room.name} finished; close it out if the agent report never arrived"
    if event.event == "egress_ended":
        info = event.egress_info
        return f"recording {info.egress_id} for call {info.room_name} is ready; attach it"
    if event.event == "participant_left":
        return f"{event.participant.identity} left call {event.room.name}"
    return f"ignored {event.event}"
