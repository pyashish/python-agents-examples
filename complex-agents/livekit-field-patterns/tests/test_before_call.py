"""BEFORE the call: what the backend sends to LiveKit."""

from livekit import api

from backend import before_call
from backend.fake_livekit import RecordingLiveKitAPI
from shared.contract import AGENT_NAME, CALLER_IDENTITY, CallConfig
from shared.samples import SAMPLE_CUSTOMER

TWILIO_FORM = {"From": "+15555550111", "To": "+15555550100", "CallSid": "CA_test"}


async def test_inbound_bridges_twilio_call_with_the_dispatch_attached():
    lk = RecordingLiveKitAPI()

    twiml = await before_call.answer_inbound_twilio_call(lk, TWILIO_FORM)

    [call] = lk.calls
    assert call.name == "connector.connect_twilio_call"
    request: api.ConnectTwilioCallRequest = call.request
    assert request.twilio_call_direction == api.ConnectTwilioCallRequest.TWILIO_CALL_DIRECTION_INBOUND
    assert request.participant_identity == CALLER_IDENTITY
    assert request.participant_attributes["sip.phoneNumber"] == "+15555550111"
    assert request.participant_attributes["twilio.callSid"] == "CA_test"

    [dispatch] = request.agents
    assert dispatch.agent_name == AGENT_NAME
    config = CallConfig.model_validate_json(dispatch.metadata)
    assert config.direction == "inbound"
    # Room name and call ID are the same thing for inbound calls.
    assert request.room_name == config.call_id

    assert "<Connect><Stream url=" in twiml
    assert request.room_name in twiml


async def test_outbound_only_books_a_dispatch():
    lk = RecordingLiveKitAPI()

    config = await before_call.book_outbound_call(
        lk,
        customer=SAMPLE_CUSTOMER,
        to_number="+15555550123",
        from_number="+15555550100",
        sip_trunk_id="ST_test",
    )

    # One call, and it is not a SIP dial: the agent dials later from its job.
    [call] = lk.calls
    assert call.name == "agent_dispatch.create_dispatch"
    request: api.CreateAgentDispatchRequest = call.request
    assert request.agent_name == AGENT_NAME
    assert request.room == f"outbound_{config.call_id}"
    assert CallConfig.model_validate_json(request.metadata) == config
    # The backend builds the prompt; the customer's details are in it.
    assert SAMPLE_CUSTOMER.invoice_number in config.instructions
    assert config.voicemail_message
