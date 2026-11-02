"""Shared pieces for the NVIDIA vs Deepgram/Cartesia benchmark: arms, audio IO, scoring."""

from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from dotenv import load_dotenv

HERE = Path(__file__).parent
ROOT = HERE.parent
load_dotenv(ROOT / ".env.livekit")
load_dotenv(ROOT / ".env.nvidia")

SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000

BCP47 = {
    "en": "en-US",
    "en-GB": "en-GB",
    "en-IN": "en-IN",
    "es": "es-US",
    "fr": "fr-FR",
    "de": "de-DE",
    "it": "it-IT",
    "pt": "pt-BR",
    "hi": "hi-IN",
}

# Per-language NVIDIA overrides: {"es": {"model": "...", "function_id": "..."}}. Without an
# entry the arm sends the en-US Parakeet model with the target language code, exactly what the
# client's agent can do today.
NVIDIA_STT_OVERRIDES: dict[str, dict] = json.loads(os.getenv("NVIDIA_STT_OVERRIDES", "{}"))
NVIDIA_TTS_VOICES: dict[str, str] = {
    "en": os.getenv("NVIDIA_TTS_VOICE", "Magpie-Multilingual.EN-US.Leo"),
    "es": "Magpie-Multilingual.ES-US.Diego",
    "fr": "Magpie-Multilingual.FR-FR.Pascal",
    "de": "Magpie-Multilingual.DE-DE.Jason",
    "it": "Magpie-Multilingual.IT-IT.Isabela",
    "pt": "Magpie-Multilingual.PT-BR.Diego",
    "hi": "Magpie-Multilingual.HI-IN.Leo",
    **json.loads(os.getenv("NVIDIA_TTS_VOICES", "{}")),
}
CARTESIA_VOICE = os.getenv("CARTESIA_VOICE", "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc")
GRADIUM_VOICES = {
    "en": "4SZHfMpw-p46Ywgs",  # Harper
    "fr": "YhIHaAfQ0cQPDV9R",  # Solène
    "de": "aBNlTApBeOlVKa23",  # Lorena
    "es": "iTQW2xFICXk8riV4",  # Vera
    "pt": "uCqxlQCKi8sPHwG2",  # Bianca
}

STT_ARMS = ("nvidia", "deepgram", "deepgram-multi", "assemblyai")
TTS_ARMS = ("nvidia", "cartesia", "gradium")


class UnsupportedLanguage(ValueError):
    pass


def base_lang(lang: str) -> str:
    return lang.split("-")[0]


def make_stt(arm: str, lang: str):
    from livekit.agents import inference
    from livekit.plugins import nvidia

    if arm == "nvidia":
        o = NVIDIA_STT_OVERRIDES.get(lang, {})
        kw = {"function_id": o["function_id"]} if "function_id" in o else {}
        return nvidia.STT(
            model=o.get(
                "model",
                os.getenv(
                    "NVIDIA_STT_MODEL", "parakeet-1.1b-en-US-asr-streaming-silero-vad-sortformer"
                ),
            ),
            # Accents are a test of the en-US model, not a request for a locale it does not serve.
            language_code="en-US" if base_lang(lang) == "en" else BCP47.get(lang, lang),
            **kw,
        )
    if arm == "deepgram":
        return inference.STT(model="deepgram/nova-3", language=base_lang(lang))
    if arm == "deepgram-multi":
        return inference.STT(model="deepgram/nova-3", language="multi")
    if arm == "assemblyai":
        return inference.STT(model="assemblyai/universal-3-6-pro", language=base_lang(lang))
    if arm.startswith("judge:"):
        return inference.STT(model=arm.removeprefix("judge:"), language=base_lang(lang))
    raise ValueError(f"unknown STT arm {arm}")


def make_tts(arm: str, lang: str):
    from livekit.agents import inference
    from livekit.plugins import nvidia

    if arm == "nvidia":
        return nvidia.TTS(voice=NVIDIA_TTS_VOICES.get(lang, NVIDIA_TTS_VOICES["en"]),
                          language_code=BCP47.get(lang, lang))
    if arm == "cartesia":
        return inference.TTS(model="cartesia/sonic-3", voice=CARTESIA_VOICE, language=base_lang(lang))
    if arm == "gradium":
        if base_lang(lang) not in GRADIUM_VOICES:
            raise UnsupportedLanguage(f"gradium has no {lang} voice")
        return inference.TTS(model="gradium/default", voice=GRADIUM_VOICES[base_lang(lang)],
                             language=base_lang(lang))
    raise ValueError(f"unknown TTS arm {arm}")


def nvidia_catalog(function_id: str, kind: str) -> list[dict]:
    """What an NVCF function actually serves. Raises SystemExit on auth failure so a bad key is
    never scored as a model failure."""
    import grpc
    import riva.client
    from riva.client.proto import riva_asr_pb2, riva_tts_pb2

    from livekit.plugins.nvidia import auth

    a = auth.create_riva_auth(api_key=os.getenv("NVIDIA_API_KEY", ""), function_id=function_id,
                              server="grpc.nvcf.nvidia.com:443", use_ssl=True)
    try:
        if kind == "asr":
            r = riva.client.ASRService(a).stub.GetRivaSpeechRecognitionConfig(
                riva_asr_pb2.RivaSpeechRecognitionConfigRequest())
        else:
            r = riva.client.SpeechSynthesisService(a).stub.GetRivaSynthesisConfig(
                riva_tts_pb2.RivaSynthesisConfigRequest())
    except grpc.RpcError as e:
        if e.code() in (grpc.StatusCode.UNAUTHENTICATED, grpc.StatusCode.PERMISSION_DENIED):
            raise SystemExit(f"NVIDIA rejected NVIDIA_API_KEY ({e.code().name}); fix .env.nvidia "
                             "before benchmarking, or every NVIDIA row is an auth error.") from e
        raise
    return [{"model": m.model_name, **dict(m.parameters)} for m in r.model_config]


def preflight(arms: list[str], kind: str) -> None:
    if "nvidia" in arms:
        fid = ("1598d209-5e27-4d3c-8079-4751568b1081" if kind == "asr"
               else "877104f7-e885-42b9-8de8-f6e4c6303969")
        nvidia_catalog(fid, kind)


def read_wav(path: Path) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        from scipy.signal import resample_poly

        g = np.gcd(sr, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // g, sr // g).astype(np.float32)
    return audio


def write_wav(path: Path, audio: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.clip(audio, -1, 1), sr, subtype="PCM_16")


def to_pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()


def frames_from_pcm(pcm: bytes, sample_rate: int) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768


def rms_db(x: np.ndarray) -> float:
    return 20 * np.log10(np.sqrt(np.mean(x**2)) + 1e-12)


_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


_ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
    "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13,
    "fourteenth": 14, "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
    "nineteenth": 19, "twentieth": 20, "thirtieth": 30,
}


def _num_words(n: int) -> str:
    """Under 100 as words; longer numbers digit by digit, since order numbers, IDs and amounts are
    read either way and recognizers format them either way."""
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    return " ".join(_ONES[int(d)] for d in str(n))


def normalize(text: str, lang: str = "en") -> list[str]:
    """Lowercase and strip punctuation; for English, render numbers and ordinals one way."""
    t = unicodedata.normalize("NFKC", text).lower()
    t = t.replace("&", " and ").replace("%", " percent ")
    if base_lang(lang) == "en":
        t = re.sub(r"\b(\d+)(st|nd|rd|th)\b", lambda m: f" {m.group(1)} th ", t)
        t = re.sub(r"\b(" + "|".join(_ORDINALS) + r")\b", lambda m: f" {_ORDINALS[m.group(1)]} th ", t)
        t = re.sub(r"\b([ap])\.?\s?m\.?(?=\W|$)", r" \1m ", t)
        t = re.sub(r"(\d+)", lambda m: " " + _num_words(int(m.group(1))) + " ", t)
    t = re.sub(r"[^\w\s']|_", " ", t)
    t = t.replace("'", "")
    return t.split()


@dataclass
class Alignment:
    sub: int
    dele: int
    ins: int
    n_ref: int
    ref_ops: list[str]  # per reference word: "ok" | "sub" | "del"

    @property
    def wer(self) -> float:
        return (self.sub + self.dele + self.ins) / max(self.n_ref, 1)


def align(ref: list[str], hyp: list[str]) -> Alignment:
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + c)
    i, j, sub, dele, ins = n, m, 0, 0, 0
    ops = ["ok"] * n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            if ref[i - 1] != hyp[j - 1]:
                sub += 1
                ops[i - 1] = "sub"
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            dele += 1
            ops[i - 1] = "del"
            i -= 1
        else:
            ins += 1
            j -= 1
    return Alignment(sub, dele, ins, n, ops)


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row.setdefault("ts", time.time())
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
