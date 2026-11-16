"""Build the AgentSession for one call from its CallConfig.

Both customers build the pipeline per call from metadata, and both wrap every model
in a fallback (Strada: pipeline/*.ts, Stuut: builder.py). The prewarmed VAD comes
from the job process (see agent/main.py prewarm).
"""

from __future__ import annotations

from livekit.agents import AgentSession, inference, llm, vad

from shared.contract import CallConfig

from .state import CallState


def build_session(config: CallConfig, *, vad_model: vad.VAD, userdata: CallState) -> AgentSession[CallState]:
    return AgentSession[CallState](
        userdata=userdata,
        vad=vad_model,
        # LiveKit Inference STT with a built-in fallback model.
        stt=inference.STT(
            model=config.stt_model,
            language=config.language,
            fallback=[config.stt_fallback_model],
        ),
        # Inference LLM has no fallback= option, so wrap two of them in a FallbackAdapter.
        llm=llm.FallbackAdapter(
            [
                inference.LLM(model=config.llm_model),
                inference.LLM(model=config.llm_backup_model),
            ],
            attempt_timeout=5.0,
        ),
        tts=inference.TTS(model=config.tts_model, voice=config.tts_voice, language=config.language),
        turn_handling={
            "turn_detection": inference.TurnDetector(),
            "endpointing": {"min_delay": 0.3},  # Stuut: 0.3 s minimum endpointing
            "interruption": {"mode": "adaptive"},  # Strada: adaptive interruption
            "preemptive_generation": {"enabled": True},  # Stuut: start the LLM before end of turn
        },
        # After this much silence the user state becomes "away" (see controllers.py).
        user_away_timeout=config.silence_reprompt_s,
    )


def describe_session(config: CallConfig) -> list[str]:
    """What build_session will create, without creating it (for the walkthrough)."""
    return [
        f"STT  {config.stt_model} (fallback {config.stt_fallback_model})",
        f"LLM  {config.llm_model} -> {config.llm_backup_model} in a FallbackAdapter",
        f"TTS  {config.tts_model} voice {config.tts_voice[:8]}…",
        "turn detector + 0.3 s min endpointing + adaptive interruption + preemptive generation",
        f"user goes 'away' after {config.silence_reprompt_s:g} s of silence",
    ]
