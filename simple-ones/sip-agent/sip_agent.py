"""Inbound SIP / telephony voice agent for LiveKit Cloud.

Answers phone calls routed via a LiveKit dispatch rule. No special SIP
dialing code is required for inbound — the SIP participant joins the room
and this agent greets them.

Run locally:
    python sip_agent.py console
    python sip_agent.py dev

Deploy:
    lk agent deploy   # from this directory (simple-ones/sip-agent/)

Point a LiveKit Phone Number dispatch rule at agent_name: sip-agent
"""

import logging
from pathlib import Path

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    cli,
    inference,
)
from livekit.agents.beta.tools import EndCallTool
from livekit.plugins import noise_cancellation, silero

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

logger = logging.getLogger("sip-agent")
logger.setLevel(logging.INFO)

AGENT_NAME = "sip-agent"


class SipAssistant(Agent):
    def __init__(self) -> None:
        end_call = EndCallTool(
            extra_description=(
                "Call this when the caller says goodbye, asks to hang up, "
                "or the conversation is finished."
            ),
            end_instructions="Say a brief goodbye and thank them for calling.",
            delete_room=True,
        )
        super().__init__(
            instructions=(
                "You are a helpful phone assistant. Keep replies short and clear "
                "for a phone call — avoid long lists and markdown. "
                "Greet the caller, help with their question, and when they are "
                "done use the end_call tool to hang up."
            ),
            tools=end_call.tools,
        )

    async def on_enter(self):
        self.session.generate_reply(
            instructions="Greet the caller briefly and ask how you can help."
        )


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name=AGENT_NAME)
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info("sip agent joining room=%s", ctx.room.name)

    session = AgentSession(
        stt=inference.STT(model="deepgram/nova-3-general"),
        llm=inference.LLM(model="openai/gpt-4.1-mini"),
        tts=inference.TTS(
            model="cartesia/sonic-3",
            voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        ),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    await session.start(
        agent=SipAssistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # Telephony-tuned noise cancellation for phone callers
            noise_cancellation=noise_cancellation.BVCTelephony(),
        ),
    )
    await ctx.connect()

    def _log_sip_participant(participant: rtc.RemoteParticipant) -> None:
        if participant.kind != rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
            return
        attrs = participant.attributes or {}
        logger.info(
            "SIP participant identity=%s call_id=%s status=%s phone=%s trunk=%s",
            participant.identity,
            attrs.get("sip.callID"),
            attrs.get("sip.callStatus"),
            attrs.get("sip.phoneNumber"),
            attrs.get("sip.trunkPhoneNumber"),
        )

    for participant in ctx.room.remote_participants.values():
        _log_sip_participant(participant)

    @ctx.room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant):
        _log_sip_participant(participant)


if __name__ == "__main__":
    cli.run_app(server)
