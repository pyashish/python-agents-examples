"""DURING the call: backend commands and how the agent reads them."""

from livekit import api

from agent.control_channel import parse_control_packet, parse_handoff
from backend import during_call
from backend.fake_livekit import RecordingLiveKitAPI
from shared.contract import CONTROL_TOPIC, AddMessageCommand, EndCallCommand, SayCommand

ROOM = "call-123"


async def test_say_round_trips_through_a_reliable_data_packet():
    lk = RecordingLiveKitAPI()

    await during_call.say(lk, ROOM, "One moment, please.", end_call_after=True)

    request: api.SendDataRequest = lk.last().request
    assert lk.last().name == "room.send_data"
    assert request.room == ROOM
    assert request.topic == CONTROL_TOPIC
    assert request.kind == api.DataPacket.Kind.RELIABLE
    command = parse_control_packet(request.data, sender=None, topic=request.topic)
    assert command == SayCommand(text="One moment, please.", end_call_after=True)


async def test_add_message_and_end_call_parse_to_their_commands():
    lk = RecordingLiveKitAPI()

    await during_call.add_message(lk, ROOM, "The caller is a priority customer.", reply=True)
    await during_call.end_call(lk, ROOM)

    add = lk.calls[0].request
    assert isinstance(parse_control_packet(add.data, sender=None, topic=add.topic), AddMessageCommand)
    assert lk.calls[1].name == "room.delete_room"
    assert lk.calls[1].request.room == ROOM
    assert parse_control_packet(b'{"type": "end_call"}', sender=None, topic=CONTROL_TOPIC) == EndCallCommand()


def test_agent_ignores_packets_that_are_not_server_commands():
    data = SayCommand(text="hello").model_dump_json().encode()

    # Sent by a room participant rather than the server API.
    assert parse_control_packet(data, sender=object(), topic=CONTROL_TOPIC) is None  # type: ignore[arg-type]
    # Wrong topic.
    assert parse_control_packet(data, sender=None, topic="chat") is None
    # Malformed or out of bounds.
    assert parse_control_packet(b"not json", sender=None, topic=CONTROL_TOPIC) is None
    too_long = ('{"type": "say", "text": "' + "x" * 2000 + '"}').encode()
    assert parse_control_packet(too_long, sender=None, topic=CONTROL_TOPIC) is None


async def test_handoff_travels_in_room_metadata():
    lk = RecordingLiveKitAPI()

    sent = await during_call.hand_off(lk, ROOM, "You are the claims specialist.", greeting="Hi!")

    request: api.UpdateRoomMetadataRequest = lk.last().request
    assert lk.last().name == "room.update_room_metadata"
    assert parse_handoff(request.metadata) == sent
    assert parse_handoff("") is None
    assert parse_handoff("not json") is None
    assert parse_handoff('{"something": "else"}') is None
