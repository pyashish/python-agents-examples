"""Record the call with egress: audio-only room composite to S3 (both customers)."""

from __future__ import annotations

import logging
import os

from livekit import api
from livekit.agents import JobContext

from shared.contract import CallConfig

logger = logging.getLogger("field-patterns.recording")


async def start_recording(ctx: JobContext, config: CallConfig) -> str | None:
    """Start egress and return the file path, or None when recording is off.

    The file is ready when LiveKit sends the egress_ended webhook (backend/after_call.py),
    not when this returns.
    """
    bucket = os.getenv("RECORDING_S3_BUCKET")
    if config.mock or ctx.is_fake_job() or not bucket:
        logger.info("recording skipped (mock, console, or no bucket configured)")
        return None

    path = f"recordings/{config.call_id}.ogg"
    await ctx.api.egress.start_room_composite_egress(
        api.RoomCompositeEgressRequest(
            room_name=ctx.room.name,
            audio_only=True,
            file_outputs=[
                api.EncodedFileOutput(
                    file_type=api.EncodedFileType.OGG,
                    filepath=path,
                    s3=api.S3Upload(
                        bucket=bucket,
                        region=os.getenv("AWS_REGION", "us-east-1"),
                        access_key=os.environ["AWS_ACCESS_KEY_ID"],
                        secret=os.environ["AWS_SECRET_ACCESS_KEY"],
                    ),
                )
            ],
        )
    )
    return path
