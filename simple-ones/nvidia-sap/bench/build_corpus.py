"""Build the STT test corpus: clean, noisy, beep-prefixed, noise-only and multi-turn session clips.

Reference speech is synthesized with Inworld (a vendor in neither arm) so neither stack is favored,
and every clip has an exact transcript. Add real recordings with --real-dir (each `x.wav` beside an
`x.txt` transcript, filename prefixed with its language, e.g. `en_call12_turn3.wav`); real speech
is the stronger evidence and should be used whenever the client can share some.

    python -m bench.build_corpus [--real-dir DIR] [--beep-wav their_beep.wav] [--noise-dir DIR]
"""

from __future__ import annotations

import argparse
import asyncio
import random
from pathlib import Path

import numpy as np

from .common import HERE, SAMPLE_RATE, append_jsonl, read_wav, rms_db, write_wav
from .prompts import CALLER_UTTERANCES, CORPUS_VOICES

CORPUS = HERE / "corpus"
SPEECH_DB = -22.0


def _gain_to(x: np.ndarray, target_db: float) -> np.ndarray:
    return x * 10 ** ((target_db - rms_db(x)) / 20)


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(n)
    f = np.fft.rfft(white)
    f /= np.sqrt(np.maximum(np.arange(len(f)), 1))
    x = np.fft.irfft(f, n)
    return (x / np.max(np.abs(x))).astype(np.float32)


def beep(kind_wav: Path | None) -> np.ndarray:
    if kind_wav:
        return read_wav(kind_wav)
    t = np.arange(int(0.3 * SAMPLE_RATE)) / SAMPLE_RATE
    tone = 0.5 * np.sin(2 * np.pi * 1000 * t) + 0.2 * np.sin(2 * np.pi * 2000 * t)
    env = np.minimum(1, np.minimum(t, t[-1] - t) / 0.01)
    return (tone * env * 0.5).astype(np.float32)


def keyboard(n: int, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros(n, dtype=np.float32)
    pos = 0
    while pos < n - 400:
        click = rng.standard_normal(240) * np.exp(-np.arange(240) / 30)
        out[pos : pos + 240] += click.astype(np.float32) * rng.uniform(0.2, 0.6)
        pos += int(rng.uniform(0.08, 0.35) * SAMPLE_RATE)
    return out


async def synthesize_speech(force: bool) -> list[dict]:
    from livekit.agents import inference, utils

    items = []
    async with utils.http_context.open():
        for lang_tag, tts_lang, voice in CORPUS_VOICES:
            texts = CALLER_UTTERANCES[tts_lang]
            try:
                tts = inference.TTS(model="inworld/inworld-tts-2", voice=voice, language=tts_lang)
            except Exception as e:
                print(f"skip voice {voice}: {e}")
                continue
            for i, text in enumerate(texts):
                path = CORPUS / "speech" / f"{lang_tag}_{voice}_{i:02d}.wav"
                if force or not path.exists():
                    try:
                        frames = [ev.frame async for ev in tts.synthesize(text)]
                    except Exception as e:
                        print(f"  fail {path.name}: {e}")
                        continue
                    sr = frames[0].sample_rate
                    pcm = np.concatenate(
                        [np.frombuffer(f.data, dtype=np.int16) for f in frames]
                    ).astype(np.float32) / 32768
                    if sr != SAMPLE_RATE:
                        from scipy.signal import resample_poly

                        g = np.gcd(sr, SAMPLE_RATE)
                        pcm = resample_poly(pcm, SAMPLE_RATE // g, sr // g).astype(np.float32)
                    write_wav(path, _trim(pcm))
                    print(f"  {path.name}")
                items.append({"path": path, "lang": lang_tag, "voice": voice, "ref": text,
                              "source": "synthetic"})
    return items


def _trim(x: np.ndarray, thresh_db: float = -45) -> np.ndarray:
    frame = 160
    energy = [rms_db(x[i : i + frame]) for i in range(0, len(x) - frame, frame)]
    voiced = [i for i, e in enumerate(energy) if e > thresh_db]
    if not voiced:
        return x
    return x[max(0, voiced[0] * frame - 800) : min(len(x), (voiced[-1] + 1) * frame + 800)]


def real_items(real_dir: Path) -> list[dict]:
    items = []
    for wav in sorted(real_dir.glob("*.wav")):
        txt = wav.with_suffix(".txt")
        if not txt.exists():
            print(f"skip {wav.name}: no transcript")
            continue
        items.append({"path": wav, "lang": wav.stem.split("_")[0], "voice": "real",
                      "ref": txt.read_text().strip(), "source": "real"})
    return items


def build(items: list[dict], beep_wav: Path | None, noise_dir: Path | None, seed: int) -> None:
    rng = np.random.default_rng(seed)
    random.seed(seed)
    manifest = CORPUS / "manifest.jsonl"
    manifest.unlink(missing_ok=True)
    clips = CORPUS / "clips"
    sil = lambda s: np.zeros(int(s * SAMPLE_RATE), dtype=np.float32)  # noqa: E731

    speech = {id(it): _gain_to(read_wav(it["path"]), SPEECH_DB) for it in items}
    en_items = [it for it in items if it["lang"].startswith("en")]

    def babble(n: int, exclude: dict | None = None) -> np.ndarray:
        pool = [it for it in items if it is not exclude]
        out = np.zeros(n, dtype=np.float32)
        for _ in range(6):
            src = speech[id(random.choice(pool))]
            src = np.tile(src, int(np.ceil(n / len(src))) + 1)
            off = rng.integers(0, len(src) - n)
            out += src[off : off + n]
        return out

    noise_beds = {"pink": lambda n, ex: pink_noise(n, rng), "babble": babble}
    if noise_dir:
        for f in sorted(noise_dir.glob("*.wav")):
            bed = read_wav(f)
            noise_beds[f.stem] = lambda n, ex, bed=bed: np.resize(bed, n)

    def emit(cid: str, audio: np.ndarray, **meta) -> None:
        path = clips / f"{cid}.wav"
        write_wav(path, audio)
        append_jsonl(manifest, {"id": cid, "path": str(path.relative_to(HERE)),
                                "duration": len(audio) / SAMPLE_RATE, **meta})

    lead = 0.5
    for it in items:
        s = speech[id(it)]
        base = f"{it['lang']}_{it['voice']}_{Path(it['path']).stem.split('_')[-1]}"
        span = [[lead, lead + len(s) / SAMPLE_RATE]]
        common = {"lang": it["lang"], "voice": it["voice"], "source": it["source"]}
        emit(f"clean_{base}", np.concatenate([sil(lead), s]), kind="clean", ref=it["ref"],
             speech_spans=span, **common)

        if it in en_items:
            for bed in noise_beds:
                for snr in (10, 5, 0):
                    clean = np.concatenate([sil(lead), s])
                    noise = _gain_to(noise_beds[bed](len(clean), it), SPEECH_DB - snr)
                    emit(f"noisy_{bed}{snr}_{base}", clean + noise,
                         kind="noisy", noise=bed, snr=snr, ref=it["ref"], speech_spans=span,
                         **common)

            b = beep(beep_wav)
            audio = np.concatenate([sil(2.0), b, sil(1.0), s])
            onset = 2.0 + len(b) / SAMPLE_RATE + 1.0
            emit(f"beep_{base}", audio, kind="beep", ref=it["ref"],
                 speech_spans=[[onset, onset + len(s) / SAMPLE_RATE]],
                 beep_span=[2.0, 2.0 + len(b) / SAMPLE_RATE], **common)

    n8 = 8 * SAMPLE_RATE
    for k in range(4):
        emit(f"noiseonly_pink_{k}", _gain_to(pink_noise(n8, rng), SPEECH_DB - 10),
             kind="noise_only", noise="pink", ref="", speech_spans=[], lang="en")
        emit(f"noiseonly_babble_{k}", _gain_to(babble(n8), SPEECH_DB - 10),
             kind="noise_only", noise="babble", ref="", speech_spans=[], lang="en")
        emit(f"noiseonly_keyboard_{k}", _gain_to(keyboard(n8, rng), SPEECH_DB - 6),
             kind="noise_only", noise="keyboard", ref="", speech_spans=[], lang="en")
        b = beep(beep_wav)
        emit(f"noiseonly_beep_{k}", np.concatenate([sil(2.0), b, sil(4.0)]),
             kind="noise_only", noise="beep", ref="", speech_spans=[], lang="en",
             beep_span=[2.0, 2.0 + len(b) / SAMPLE_RATE])
        for bed in (noise_dir.glob("*.wav") if noise_dir else []):
            emit(f"noiseonly_{bed.stem}_{k}", _gain_to(noise_beds[bed.stem](n8, None), SPEECH_DB - 10),
                 kind="noise_only", noise=bed.stem, ref="", speech_spans=[], lang="en")

    # Multi-turn sessions: one continuous stream, many utterances, the way a call actually flows.
    by_voice: dict[str, list[dict]] = {}
    for it in en_items:
        by_voice.setdefault(it["voice"], []).append(it)
    for voice, its in by_voice.items():
        for variant, bed, snr in (("quiet", "pink", 25), ("babble10", "babble", 10)):
            order = its[:]
            random.shuffle(order)
            parts, spans, refs, t = [sil(1.0)], [], [], 1.0
            for it in order:
                s = speech[id(it)]
                spans.append([t, t + len(s) / SAMPLE_RATE])
                refs.append(it["ref"])
                gap = random.uniform(1.6, 2.6)
                parts += [s, sil(gap)]
                t += len(s) / SAMPLE_RATE + gap
            audio = np.concatenate(parts)
            audio += _gain_to(noise_beds[bed](len(audio), None), SPEECH_DB - snr)
            emit(f"session_{variant}_{voice}", audio, kind="session", noise=bed, snr=snr,
                 ref=" ".join(refs), utterance_refs=refs, speech_spans=spans, lang="en",
                 voice=voice, source=its[0]["source"])

    print(f"wrote {sum(1 for _ in manifest.open())} clips to {manifest}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-dir", type=Path)
    ap.add_argument("--beep-wav", type=Path, help="the client's actual connect beep, if available")
    ap.add_argument("--noise-dir", type=Path, help="extra noise beds (*.wav): TV, street, call center")
    ap.add_argument("--no-synthetic", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    items = [] if a.no_synthetic else asyncio.run(synthesize_speech(a.force))
    if a.real_dir:
        items += real_items(a.real_dir)
    if not items:
        raise SystemExit("no speech items")
    build(items, a.beep_wav, a.noise_dir, a.seed)


if __name__ == "__main__":
    main()
