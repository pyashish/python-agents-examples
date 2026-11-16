"""DURING the call: receive the backend's commands (Strada: control/commands.ts, agents/handoff.ts).

The backend never joins the room. It sends data packets and updates room metadata
through the server API (backend/during_call.py); the agent listens for both here.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Coroutine
from typing import Any

from livekit import rtc
from livekit.agents import AgentSession
from pydantic import ValidationError

from shared.contract import (
    CONTROL_COMMAND,
    CONTROL_TOPIC,
    AddMessageCommand,
    ControlCommand,
    EndCallCommand,
    Handoff,
    RoomMetadata,
    SayCommand,
)

from .state import CallState
from .telephony import hang_up

logger = logging.getLogger("field-patterns.control")


def parse_control_packet(data: bytes, *, sender: rtc.RemoteParticipant | None, topic: str | None) -> ControlCommand | None:
    """Return the command, or None for anything that is not a valid server command.

    Packets sent with the server API have no sender. Anything a participant publishes
    (including the caller's own client, if it had one) is ignored.
    """
    if sender is not None or topic != CONTROL_TOPIC:
        return None
    try:
        return CONTROL_COMMAND.validate_json(data)
    except ValidationError:
        # Don't log the payload: it can carry phone numbers or free text.
        logger.warning("rejected malformed control command")
        return None


def parse_handoff(metadata: str) -> Handoff | None:
    try:
        return RoomMetadata.model_validate(json.loads(metadata or "{}")).handoff
    except (ValueError, ValidationError):
        return None


async def apply_command(command: ControlCommand, session: AgentSession[CallState], call: CallState) -> None:
    if isinstance(command, SayCommand):
        handle = session.say(command.text)
        if command.end_call_after:
            await handle.wait_for_playout()
            call.end_with("backend-ended-call")
            await hang_up("backend-ended-call")
    elif isinstance(command, AddMessageCommand):
        agent = session.current_agent
        chat_ctx = agent.chat_ctx.copy()
        chat_ctx.add_message(role=command.role, content=command.content)
        await agent.update_chat_ctx(chat_ctx)
        if command.reply:
            session.generate_reply()
    elif isinstance(command, EndCallCommand):
        call.end_with("backend-ended-call")
        await hang_up("backend-ended-call")


def apply_handoff(handoff: Handoff, session: AgentSession[CallState], call: CallState) -> None:
    """Swap in a new agent, keeping the conversation unless told otherwise."""
    from .phone_agent import PhoneAgent  # local import: phone_agent imports nothing from here

    call.config = call.config.model_copy(update={"instructions": handoff.instructions, "greeting": handoff.greeting})
    chat_ctx = session.history.copy(exclude_instructions=True) if handoff.keep_history else None
    session.update_agent(PhoneAgent(call.config, greet_on_enter=handoff.greeting is not None, chat_ctx=chat_ctx))


def attach_control_channel(room: rtc.Room, session: AgentSession[CallState], call: CallState) -> Callable[[], None]:
    """Listen for commands and handoffs. Returns a function that stops listening."""
    tasks: set[asyncio.Task] = set()
    applied_handoffs: set[str] = set()

    def spawn(coro: Coroutine[Any, Any, None]) -> None:
        # Room callbacks are synchronous; keep a reference so the task isn't collected.
        task = asyncio.create_task(coro)
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    def on_data(packet: rtc.DataPacket) -> None:
        command = parse_control_packet(packet.data, sender=packet.participant, topic=packet.topic)
        if command is not None:
            logger.info("control command", extra={"type": command.type})
            spawn(apply_command(command, session, call))

    def on_metadata(_old: str, new: str) -> None:
        handoff = parse_handoff(new)
        # Metadata can be re-delivered; apply each handoff once.
        if handoff is None or handoff.id in applied_handoffs:
            return
        applied_handoffs.add(handoff.id)
        apply_handoff(handoff, session, call)

    room.on("data_received", on_data)
    room.on("room_metadata_changed", on_metadata)

    def detach() -> None:
        room.off("data_received", on_data)
        room.off("room_metadata_changed", on_metadata)

    return detach
