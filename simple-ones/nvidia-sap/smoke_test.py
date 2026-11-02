"""Synthesize a phrase with Magpie, then transcribe it back with Parakeet."""

import asyncio
from pathlib import Path

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import stt as agents_stt
from livekit.plugins import nvidia

load_dotenv(Path(__file__).parent / ".env.nvidia")


async def main() -> None:
    tts = nvidia.TTS()
    frames = [ev.frame async for ev in tts.synthesize("Hello from NVIDIA Magpie and Parakeet.")]
    print(f"TTS ok: {len(frames)} frames, {sum(f.duration for f in frames):.2f}s audio")

    resampler = rtc.AudioResampler(frames[0].sample_rate, 16000, num_channels=1)
    stream = nvidia.STT().stream()
    for frame in frames:
        for out in resampler.push(frame):
            stream.push_frame(out)
    silence = rtc.AudioFrame(b"\x00\x00" * 1600, 16000, 1, 1600)
    for _ in range(20):
        stream.push_frame(silence)
    stream.end_input()
    async for ev in stream:
        if ev.type == agents_stt.SpeechEventType.FINAL_TRANSCRIPT:
            print("STT ok:", ev.alternatives[0].text)


asyncio.run(main())
