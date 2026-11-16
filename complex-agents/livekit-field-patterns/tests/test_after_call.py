"""AFTER the call: the agent's report and LiveKit's webhooks."""

import pytest
from livekit import api
from livekit.agents import llm

from agent.after_call import build_call_report, deliver_report
from agent.state import CallState
from backend import after_call
from shared.contract import CallReport, sign_body
from shared.samples import sample_inbound_config

SECRET = "test-secret"


def make_report() -> CallReport:
    history = llm.ChatContext.empty()
    history.add_message(role="system", content="instructions are not part of the transcript")
    history.add_message(role="assistant", content="Thanks for calling. How can I help?")
    history.add_message(role="user", content="Just checking an invoice.")
    call = CallState(config=sample_inbound_config())
    call.end_with("agent-ended-call")
    call.end_with("session-closed:job_shutdown")  # later, less specific reasons don't overwrite
    return build_call_report(call, history.items)


def test_report_keeps_only_the_conversation_and_the_first_end_reason():
    report = make_report()

    assert report.ended_reason == "agent-ended-call"
    assert [line.role for line in report.transcript] == ["assistant", "user"]


def test_backend_accepts_a_signed_report_and_rejects_a_tampered_one():
    report = make_report()
    body = report.model_dump_json().encode()
    signature = sign_body(body, SECRET)

    assert after_call.receive_call_report(body, signature, SECRET) == report
    with pytest.raises(after_call.InvalidSignature):
        after_call.receive_call_report(body.replace(b"agent-ended-call", b"customer-hung-up"), signature, SECRET)
    with pytest.raises(after_call.InvalidSignature):
        after_call.receive_call_report(body, signature, "wrong-secret")


async def test_mock_calls_never_post_the_report():
    assert await deliver_report(make_report(), url="https://backend.example.com/report", mock=True) is False


def test_livekit_webhooks_are_routed_by_event_type():
    egress = api.WebhookEvent(event="egress_ended", egress_info=api.EgressInfo(egress_id="EG_1", room_name="call-1"))
    finished = api.WebhookEvent(event="room_finished", room=api.Room(name="call-1"))
    other = api.WebhookEvent(event="track_published")

    assert "recording EG_1 for call call-1" in after_call.route_livekit_event(egress)
    assert "call call-1 finished" in after_call.route_livekit_event(finished)
    assert after_call.route_livekit_event(other) == "ignored track_published"
