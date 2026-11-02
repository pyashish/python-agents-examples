"""List what the configured NVIDIA functions serve: ASR models per language, TTS voices per language.

    python -m bench.probe [--asr-function ID] [--tts-function ID]

Use this to fill NVIDIA_STT_OVERRIDES / NVIDIA_TTS_VOICES with NVIDIA's best per-language option
before running the language comparison, so the NVIDIA side is not handicapped by configuration.
"""

import argparse

from .common import nvidia_catalog


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr-function", default="1598d209-5e27-4d3c-8079-4751568b1081")
    ap.add_argument("--tts-function", default="877104f7-e885-42b9-8de8-f6e4c6303969")
    a = ap.parse_args()
    for m in nvidia_catalog(a.asr_function, "asr"):
        print(f"ASR  {m.get('language_code', '?'):8} {m.get('type', ''):8} {m['model']}")
    for m in nvidia_catalog(a.tts_function, "tts"):
        voices = [v.split(":")[0] for v in m.get("subvoices", "").split(",") if v]
        print(f"TTS  {m.get('language_code', '?'):8} {m.get('voice_name', '')}: {', '.join(voices)}")


if __name__ == "__main__":
    main()
