"""Dispatch sip-outbound-agent to place a phone call.

The agent reads the phone number from job metadata and dials via
CreateSIPParticipant (see sip_outbound_agent.py).

Usage:
    python make_call.py +15551234567

Requires LIVEKIT_* credentials and a running worker named sip-outbound-agent
(local `python sip_outbound_agent.py dev` or a Cloud deploy).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import uuid
from pathlib import Path

from dotenv import load_dotenv
from livekit import api

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

logger = logging.getLogger("make-call")
logger.setLevel(logging.INFO)

AGENT_NAME = "sip-outbound-agent"


async def make_call(phone_number: str) -> None:
    room_name = f"outbound-{uuid.uuid4().hex[:10]}"
    metadata = json.dumps({"phone_number": phone_number})

    lkapi = api.LiveKitAPI()
    try:
        logger.info(
            "dispatching agent=%s room=%s phone=%s",
            AGENT_NAME,
            room_name,
            phone_number,
        )
        dispatch = await lkapi.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name=AGENT_NAME,
                room=room_name,
                metadata=metadata,
            )
        )
        logger.info("created dispatch id=%s — agent will dial shortly", dispatch.id)
    finally:
        await lkapi.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Place an outbound call via LiveKit")
    parser.add_argument(
        "phone_number",
        help="E.164 number to dial, e.g. +15551234567",
    )
    args = parser.parse_args()
    asyncio.run(make_call(args.phone_number))


if __name__ == "__main__":
    main()
