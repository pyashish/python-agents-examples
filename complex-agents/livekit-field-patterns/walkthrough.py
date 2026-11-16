"""A guided tour of one inbound and one outbound call, BEFORE -> DURING -> AFTER.

Runs offline: the backend talks to a recording fake of the LiveKit API, and the agent
steps that need a live room are described rather than run. Every function called
here is the real one from backend/ or agent/.

    uv run walkthrough.py
"""

from __future__ import annotations

import asyncio
import json

from livekit import api
from livekit.agents import llm

from agent.after_call import build_call_report
from agent.control_channel import parse_control_packet, parse_handoff
from agent.phone_agent import OPT_OUT, apply_pronunciations
from agent.pipeline import describe_session
from agent.state import CallState
from backend import after_call, before_call, during_call
from backend.fake_livekit import RecordingLiveKitAPI
from shared.contract import CONTROL_TOPIC, CallConfig, room_name_for, sign_body
from shared.samples import SAMPLE_CUSTOMER

SECRET = "walkthrough-secret"


def heading(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def step(who: str, text: str) -> None:
    print(f"\n[{who}] {text}")


def show(label: str, value: object) -> None:
    print(f"    {label:<22} {value}")


async def main() -> None:
    lk = RecordingLiveKitAPI()

    # ------------------------------------------------------------------ BEFORE
    heading("BEFORE THE CALL  (backend/before_call.py, agent/main.py)")

    step("backend · inbound · Strada", "Twilio posts its Voice webhook -> answer_inbound_twilio_call()")
    twiml = await before_call.answer_inbound_twilio_call(
        lk, {"From": "+15555550111", "To": "+15555550100", "CallSid": "CA_example"}, mock=True
    )
    req = lk.last().request
    show("API call", lk.last().name)
    show("room_name", req.room_name)
    show("dispatch agent_name", req.agents[0].agent_name)
    show("participant_identity", req.participant_identity)
    show("participant_attributes", dict(req.participant_attributes))
    show("returns TwiML", twiml)
    inbound = CallConfig.model_validate_json(req.agents[0].metadata)

    step("backend · outbound · Stuut", "book_outbound_call() creates a dispatch and returns at once")
    outbound = await before_call.book_outbound_call(
        lk,
        customer=SAMPLE_CUSTOMER,
        to_number="+15555550123",
        from_number="+15555550100",
        sip_trunk_id="ST_example",
        mock=True,
    )
    req = lk.last().request
    show("API call", lk.last().name)
    show("room", req.room)
    show("agent_name", req.agent_name)
    show("metadata bytes", len(req.metadata))
    show("note", "the phone has not rung; the agent dials from inside its job")

    step("agent · prewarm", "each idle process loads Silero VAD once (agent/main.py prewarm)")
    step("agent · entrypoint", "the job parses job.metadata into CallConfig (agent/main.py load_call_config)")
    for config in (inbound, outbound):
        show(config.direction, f"call_id={config.call_id[:8]}…  room={room_name_for(config)[:20]}…")
    step("agent · build_session", "agent/pipeline.py builds the AgentSession from the config:")
    for line in describe_session(outbound):
        show("", line)

    step("agent · inbound start", "agent/inbound.py run_inbound()")
    show("1", "ctx.wait_for_participant(identity='caller')")
    show("2", "attach controllers + control channel, add the shutdown callback")
    show("3", "session.start(..., RoomOptions(delete_room_on_close=True))")
    show("4", f"PhoneAgent.on_enter() speaks the greeting: {inbound.greeting!r}")

    step("agent · outbound start", "agent/outbound.py run_outbound()")
    show("1", "session.start() silently, delete_room_on_close=False")
    show("2", "async with AMD(...): sip.create_sip_participant(wait_until_answered=True)")
    show("3", "verdict human -> greet | voicemail -> leave message, hang up")
    show("", "phone menu -> stay quiet, send_dtmf | full mailbox -> hang up")

    # ------------------------------------------------------------------ DURING
    heading("DURING THE CALL  (backend/during_call.py, agent/control_channel.py, agent/phone_agent.py)")
    room = room_name_for(inbound)

    step("backend", "during_call.say() sends a data packet")
    await during_call.say(lk, room, "One moment while I check that for you.")
    packet = lk.last().request
    show("API call", f"{lk.last().name} topic={packet.topic!r} kind={api.DataPacket.Kind.Name(packet.kind)}")
    command = parse_control_packet(packet.data, sender=None, topic=packet.topic)
    show("agent parses", command)
    show("from a participant", parse_control_packet(packet.data, sender=object(), topic=CONTROL_TOPIC))  # type: ignore[arg-type]
    show("wrong topic", parse_control_packet(packet.data, sender=None, topic="chat"))

    step("backend", "during_call.hand_off() writes room metadata (too big for a data packet)")
    await during_call.hand_off(lk, room, "You are now the claims specialist.", greeting="Hi, I can help with that claim.")
    metadata = lk.last().request.metadata
    show("API call", lk.last().name)
    show("agent parses", parse_handoff(metadata))
    show("then", "session.update_agent(PhoneAgent(new config, chat_ctx=history copy))")

    step("agent · hooks on every turn", "agent/phone_agent.py")
    said = "Please stop calling me."
    show("on_user_turn_completed", f"{said!r} -> opt-out rule matched: {bool(OPT_OUT.search(said))}")
    text = "Northwind here about INV-20417, due October 31."
    show("tts_node", f"{text!r}")
    show("", f"-> {apply_pronunciations(text, outbound.pronunciations)!r}")
    show("llm_node", "records time to first token in CallState.llm_first_token_s")
    show("tools", "lookup_invoice (HTTP tool server or mock), send_dtmf, end_call")

    step("backend", "during_call.end_call() deletes the room, which drops the phone leg")
    await during_call.end_call(lk, room)
    show("API call", f"{lk.last().name} room={lk.last().request.room[:8]}…")

    # ------------------------------------------------------------------ AFTER
    heading("AFTER THE CALL  (agent/after_call.py, backend/after_call.py)")

    step("agent · shutdown callback", "build_call_report() from the session history")
    history = llm.ChatContext.empty()
    history.add_message(role="assistant", content=inbound.greeting or "")
    history.add_message(role="user", content="Can you check invoice INV-20417?")
    history.add_message(role="assistant", content="It's open, $1,240 due October 31. Anything else?")
    history.add_message(role="user", content="No, that's all. Thanks!")
    call = CallState(config=inbound, recording_path=f"recordings/{inbound.call_id}.ogg")
    call.end_with("agent-ended-call")
    report = build_call_report(call, history.items)
    body = report.model_dump_json().encode()
    signature = sign_body(body, SECRET)
    show("ended_reason", report.ended_reason)
    show("transcript lines", len(report.transcript))
    show("signature header", signature[:16] + "…")

    step("backend", "after_call.receive_call_report() verifies the signature")
    accepted = after_call.receive_call_report(body, signature, SECRET)
    show("accepted", f"call {accepted.call_id[:8]}… ended by {accepted.ended_reason}")
    tampered = json.loads(body)
    tampered["ended_reason"] = "customer-ended-call"
    try:
        after_call.receive_call_report(json.dumps(tampered).encode(), signature, SECRET)
    except after_call.InvalidSignature as e:
        show("tampered body", f"rejected: {e}")

    step("LiveKit -> backend", "server webhooks arrive separately (verified with api.WebhookReceiver)")
    for event in (
        api.WebhookEvent(event="egress_ended", egress_info=api.EgressInfo(egress_id="EG_example", room_name=room)),
        api.WebhookEvent(event="room_finished", room=api.Room(name=room)),
    ):
        show(event.event, after_call.route_livekit_event(event))

    print(f"\nRecorded {len(lk.calls)} LiveKit API calls: {', '.join(c.name for c in lk.calls)}\n")


if __name__ == "__main__":
    asyncio.run(main())
