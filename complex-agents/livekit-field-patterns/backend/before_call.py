"""BEFORE the call: the backend decides everything, then hands the call to LiveKit.

Two entry points, one per customer pattern:

- answer_inbound_twilio_call  (Strada, apps/server/src/routes/webhook/livekit-inbound-call.ts)
  Twilio posts its Voice webhook here. We create the call ID, ask LiveKit's Twilio
  connector to bridge the call into a room *with the agent dispatch attached*, and
  answer Twilio with TwiML that streams the call's audio to LiveKit.

- book_outbound_call  (Stuut, libs/stuut-core/.../outreach/call/execution.py)
  We build the call config and create an agent dispatch. LiveKit accepts it in about
  30 ms and we return. The phone does not ring yet: the agent dials from inside its job.

`lkapi` is either `livekit.api.LiveKitAPI()` or backend.fake_livekit.RecordingLiveKitAPI.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from xml.sax.saxutils import quoteattr

from livekit import api

from shared.contract import AGENT_NAME, CALLER_IDENTITY, CallConfig, room_name_for
from shared.samples import Customer, inbound_instructions, outbound_instructions


async def answer_inbound_twilio_call(lkapi, twilio_form: Mapping[str, str], *, mock: bool = False) -> str:
    """Handle Twilio's Voice webhook for a number answered by LiveKit. Returns TwiML.

    A real handler first validates X-Twilio-Signature with the number's auth token and
    looks the number up to pick the assistant; Strada rejects unknown numbers with 403.
    """
    call_id = str(uuid.uuid4())  # Strada uses uuidv7; it doubles as the room name.
    config = CallConfig(
        call_id=call_id,
        direction="inbound",
        instructions=inbound_instructions("Northwind Supply"),
        greeting="Thanks for calling Northwind Supply billing. How can I help?",
        pronunciations={"Northwind": "North wind"},
        report_url="https://backend.example.com/calls/report",
        mock=mock,
    )

    request = api.ConnectTwilioCallRequest(
        twilio_call_direction=api.ConnectTwilioCallRequest.TWILIO_CALL_DIRECTION_INBOUND,
        room_name=room_name_for(config),
        # The dispatch rides on the bridge request: the room and the agent arrive together.
        agents=[api.RoomAgentDispatch(agent_name=AGENT_NAME, metadata=config.model_dump_json())],
        participant_identity=CALLER_IDENTITY,
        # The agent reads these off the participant (Strada: telephony/call-info.ts).
        participant_attributes={
            "sip.phoneNumber": twilio_form["From"],
            "sip.trunkPhoneNumber": twilio_form["To"],
            "twilio.callSid": twilio_form["CallSid"],
        },
    )
    response = await lkapi.connector.connect_twilio_call(request)
    return twiml_connect_stream(response.connect_url)


def twiml_connect_stream(url: str) -> str:
    """<Connect><Stream> hands the call's media to LiveKit for as long as the call lasts."""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f"<Response><Connect><Stream url={quoteattr(url)}/></Connect></Response>"
    )


async def book_outbound_call(
    lkapi,
    *,
    customer: Customer,
    to_number: str,
    from_number: str,
    sip_trunk_id: str,
    mock: bool = False,
) -> CallConfig:
    """Create the dispatch for an outbound call and return its config.

    Stuut marks the call REGISTERED at this point. Whether it was answered, went to
    voicemail or failed to dial is reported later by the agent's webhook.
    """
    config = CallConfig(
        call_id=str(uuid.uuid4()),
        direction="outbound",
        # The backend builds the whole prompt; the agent never sees the raw customer record.
        instructions=outbound_instructions(customer, "Northwind Supply"),
        greeting=f"Hi, this is Northwind Supply calling for {customer.name}. Is now a good time?",
        voicemail_message=(
            f"Hi {customer.name}, this is Northwind Supply about invoice "
            f"{customer.invoice_number}. Please call us back. Thank you."
        ),
        to_number=to_number,
        from_number=from_number,
        sip_trunk_id=sip_trunk_id,
        pronunciations={"Northwind": "North wind", "INV-": "invoice "},
        report_url="https://backend.example.com/calls/report",
        mock=mock,
    )
    await lkapi.agent_dispatch.create_dispatch(
        api.CreateAgentDispatchRequest(
            agent_name=AGENT_NAME,
            room=room_name_for(config),
            metadata=config.model_dump_json(),
        )
    )
    return config
