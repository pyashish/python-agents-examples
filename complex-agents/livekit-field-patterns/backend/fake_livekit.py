"""A stand-in for livekit.api.LiveKitAPI that records requests instead of sending them.

Every backend function takes the API client as a parameter, so the walkthrough and
the tests pass this fake and real code passes `api.LiveKitAPI()`. The requests it
records are the real protobuf messages, so what you see is exactly what would be sent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from livekit import api


def _canned_response(service: str, method: str, request: Any) -> Any:
    if method == "connect_twilio_call":
        return api.ConnectTwilioCallResponse(
            connect_url=f"wss://example.livekit.cloud/twilio/{request.room_name}?token=..."
        )
    if method == "create_dispatch":
        return api.AgentDispatch(
            id="AD_mock", agent_name=request.agent_name, room=request.room, metadata=request.metadata
        )
    if method == "update_room_metadata":
        return api.Room(name=request.room, metadata=request.metadata)
    if method == "delete_room":
        return api.DeleteRoomResponse()
    if method == "send_data":
        return api.SendDataResponse()
    return None


@dataclass
class RecordedCall:
    name: str  # "service.method"
    request: Any


@dataclass
class _Service:
    owner: RecordingLiveKitAPI
    service: str

    def __getattr__(self, method: str):
        async def call(request: Any) -> Any:
            self.owner.calls.append(RecordedCall(f"{self.service}.{method}", request))
            return _canned_response(self.service, method, request)

        return call


@dataclass
class RecordingLiveKitAPI:
    calls: list[RecordedCall] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.agent_dispatch = _Service(self, "agent_dispatch")
        self.connector = _Service(self, "connector")
        self.room = _Service(self, "room")
        self.sip = _Service(self, "sip")
        self.egress = _Service(self, "egress")

    async def aclose(self) -> None:
        return None

    def last(self) -> RecordedCall:
        return self.calls[-1]
