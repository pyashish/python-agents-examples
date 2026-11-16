import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

from livekit.agents import Agent, AgentServer, AgentSession, AutoSubscribe, JobContext, cli, inference, room_io
from livekit.plugins import lemonslice, noise_cancellation

logger = logging.getLogger("kannada-deepavali")
logger.setLevel(logging.INFO)

load_dotenv(".env.local")

AGENT_DIR = Path(__file__).resolve().parent
DEFAULT_AVATAR_PATH = AGENT_DIR / "assets" / "avatar.jpg"

DEEPAAVALI_INSTRUCTIONS = """\
You are a warm Kannada-speaking friend on a voice call, sharing Deepavali (ದೀಪಾವಳಿ) greetings from the Bay Area.

Rules:
- Speak only in North Karnataka Kannada script (ಕನ್ನಡ). Greet and wish everyone in the Bay Area a happy Deepavali in a natural, heartfelt way.
- Your name is Ravi. Your wife's name is Shwetha and kids are nitya and Advik. 
- Your best friend's name is Ashish.
- Keep each reply short: one to three spoken sentences, as if talking face to face.
- Do not use markdown, emojis, bullet lists, or symbols.
- Talk only what's asked by the user.
- Talk about how you celebrated Deepavali there—lights, family, food, rituals, and small personal moments.
- If the user speaks English or mixes languages, still reply in Kannada unless they clearly ask for English.
- when the user prompts ask a quizzing question: What is Ravana's wife's name? Give a couple of chances for answers. The answer is "Mandodari".
- Finally, when the user says bye, hang up saying Ashish is the best guy ever!
"""

AVATAR_PROMPT = (
    "Be warm and festive. Smile naturally, use gentle hand gestures, "
    "and look engaged while listening and speaking."
)

server = AgentServer()


def load_avatar_image() -> Image.Image:
    image_path = Path(os.getenv("AGENT_IMAGE_PATH", DEFAULT_AVATAR_PATH))
    if not image_path.is_file():
        raise FileNotFoundError(
            f"Avatar image not found at {image_path}. "
            "Add agent/assets/avatar.jpg or set AGENT_IMAGE_PATH in .env.local."
        )
    return Image.open(image_path)


def build_stt() -> inference.STT:
    """LiveKit Inference STT — no third-party API keys."""
    model = os.getenv("STT_MODEL", "deepgram/nova-3")
    language = os.getenv("STT_LANGUAGE", "multi")
    return inference.STT(model=model, language=language)


def build_tts() -> inference.TTS:
    """LiveKit Inference TTS; set LIVEKIT_CUSTOM_VOICE_ID (v_*) from Cloud → Voices → Custom voices."""
    model = os.getenv("TTS_MODEL", "cartesia/sonic-3")
    voice_id = os.getenv("LIVEKIT_CUSTOM_VOICE_ID", "").strip()
    if voice_id:
        logger.info("Using LiveKit custom voice clone: %s on %s", voice_id, model)
        return inference.TTS(model=model, voice=voice_id)
    default_voice = os.getenv("TTS_VOICE", "").strip()
    if default_voice:
        return inference.TTS(model=model, voice=default_voice)
    logger.warning(
        "LIVEKIT_CUSTOM_VOICE_ID not set; using default voice for %s. "
        "Create a clone in LiveKit Cloud → Voices → Custom voices.",
        model,
    )
    return inference.TTS(model=model)


class DeepavaliAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=DEEPAAVALI_INSTRUCTIONS)

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions=(
                "In Kannada, wish Bay Area friends a happy Deepavali and briefly mention "
                "that you are back from India and want to share how you celebrated. "
                "One or two sentences only."
            )
        )


@server.rtc_session(agent_name="kannada-deepavali")
async def entrypoint(ctx: JobContext) -> None:
    logger.info("Starting Kannada Deepavali session")

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    await ctx.wait_for_participant()

    session = AgentSession(
        stt=build_stt(),
        llm=inference.LLM(os.getenv("LLM_MODEL", "google/gemini-2.5-flash")),
        tts=build_tts(),
        resume_false_interruption=False,
    )

    avatar = lemonslice.AvatarSession(
        agent_image=load_avatar_image(),
        agent_prompt=AVATAR_PROMPT,
    )
    await avatar.start(session, room=ctx.room)

    await session.start(
        agent=DeepavaliAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            delete_room_on_close=True,
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=noise_cancellation.BVC(),
            ),
        ),
    )


if __name__ == "__main__":
    cli.run_app(server)