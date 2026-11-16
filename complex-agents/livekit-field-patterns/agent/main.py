"""The agent server: one process that registers with LiveKit and runs one job per call.

Run it with `uv run worker.py console` (talk to it locally) or `uv run worker.py dev`
(register with LiveKit Cloud and take real dispatches). See README.md.
"""

from __future__ import annotations

import logging
import os

from livekit.agents import AgentServer, JobContext, JobProcess, cli
from livekit.plugins import silero

from shared.contract import AGENT_NAME, CallConfig
from shared.samples import sample_inbound_config, sample_outbound_config

from .inbound import run_inbound
from .outbound import run_outbound

logger = logging.getLogger("field-patterns")

server = AgentServer(
    # How long shutdown callbacks (the end-of-call report) get before the job process
    # is killed. The default is 10 s; Stuut uses 75 s to allow webhook retries.
    shutdown_process_timeout=60.0,
)


def prewarm(proc: JobProcess) -> None:
    """BEFORE any call: runs once per idle process, so a new call doesn't pay for it."""
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


def load_call_config(ctx: JobContext) -> CallConfig:
    """Everything about the call comes from the dispatch metadata.

    Console mode and `lk agent simulate` have no metadata, so fall back to a sample call
    with mock=True: no dialing, no recording, no webhooks, canned tool results.
    """
    if ctx.job.metadata.strip():
        return CallConfig.model_validate_json(ctx.job.metadata)
    direction = os.getenv("CONSOLE_DIRECTION", "inbound")
    logger.info("no dispatch metadata; using the sample %s call", direction)
    return sample_outbound_config() if direction == "outbound" else sample_inbound_config()


@server.rtc_session(agent_name=AGENT_NAME)
async def entrypoint(ctx: JobContext) -> None:
    config = load_call_config(ctx)
    # Tag every SDK log line for this job with the call ID.
    ctx.log_context_fields["call_id"] = config.call_id
    # Stuut routes by room-name prefix; the direction field says the same thing explicitly.
    if config.direction == "inbound":
        await run_inbound(ctx, config)
    else:
        await run_outbound(ctx, config)


def main() -> None:
    cli.run_app(server)
