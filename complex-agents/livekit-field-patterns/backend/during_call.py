"""DURING the call: the backend steers a live call without joining the room.

Strada does this (packages/voice-gateway/src/client.ts); Stuut does not, because its
worker handles everything in-process. Three channels, from lightest to heaviest:

- Data packets (RoomService.SendData) on CONTROL_TOPIC: short commands such as "say".
  Packets sent through the server API have no sender, which is how the agent tells
  them apart from anything a participant publishes.
- Room metadata (RoomService.UpdateRoomMetadata): larger state such as a handoff.
- Deleting the room (RoomService.DeleteRoom): ends the call outright, no agent needed.
"""

from __future__ import annotations

import uuid

from livekit import api

from shared.contract import (
    CONTROL_TOPIC,
    AddMessageCommand,
    EndCallCommand,
    Handoff,
    RoomMetadata,
    SayCommand,
)


async def _send_command(lkapi, room_name: str, command: SayCommand | AddMessageCommand | EndCallCommand) -> None:
    await lkapi.room.send_data(
        api.SendDataRequest(
            room=room_name,
            data=command.model_dump_json().encode(),
            kind=api.DataPacket.Kind.RELIABLE,
            topic=CONTROL_TOPIC,
        )
    )


async def say(lkapi, room_name: str, text: str, *, end_call_after: bool = False) -> None:
    """Make the agent speak a line now, queued after anything it is already saying."""
    await _send_command(lkapi, room_name, SayCommand(text=text, end_call_after=end_call_after))


async def add_message(lkapi, room_name: str, content: str, *, reply: bool = False) -> None:
    """Add a system message to the agent's context, optionally asking it to respond."""
    await _send_command(lkapi, room_name, AddMessageCommand(role="system", content=content, reply=reply))


async def hand_off(lkapi, room_name: str, instructions: str, *, greeting: str | None = None) -> Handoff:
    """Switch the live call to new instructions; the agent calls session.update_agent()."""
    handoff = Handoff(id=str(uuid.uuid4()), instructions=instructions, greeting=greeting)
    await lkapi.room.update_room_metadata(
        api.UpdateRoomMetadataRequest(room=room_name, metadata=RoomMetadata(handoff=handoff).model_dump_json())
    )
    return handoff


async def end_call(lkapi, room_name: str) -> None:
    """Hang up from the backend. Deleting the room drops every participant, phone leg included."""
    await lkapi.room.delete_room(api.DeleteRoomRequest(room=room_name))
