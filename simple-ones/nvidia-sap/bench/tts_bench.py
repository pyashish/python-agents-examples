"""Time-to-first-audio, cold start and silent sentence loss for each TTS arm.

Text is pushed word by word at roughly LLM token speed, the way an agent feeds TTS, and time to
first audio is measured from the first word pushed. Each output is transcribed by a judge STT from
a vendor in neither arm and aligned to the input text per sentence: a sentence the audio does not
contain is a "dropped" response, which the NVIDIA plugin produces without raising (it logs the
synthesis error and moves to the next sentence). Audio is saved for a blind listening test.

    python -m bench.tts_bench --arms nvidia,cartesia --repeats 3 --cold-trials 5
"""

from __future__ import annotations

import argparse
import asyncio
import random
import re
import time
import traceback
from pathlib import Path

import numpy as np
from livekit.agents import utils

from .common import (
    HERE,
    SAMPLE_RATE,
    UnsupportedLanguage,
    align,
    append_jsonl,
    make_stt,
    make_tts,
    normalize,
    preflight,
    write_wav,
)
from .prompts import AGENT_REPLIES
from .stt_bench import stream_clip

AUDIO_DIR = HERE / "results" / "tts_audio"


async def synth(tts_obj, text: str, word_interval_s: float) -> dict:
    stream = tts_obj.stream()
    words = text.split(" ")
    t0 = time.perf_counter()
    first = None
    chunks: list[np.ndarray] = []
    sr = None

    async def push() -> None:
        for w in words:
            stream.push_text(w + " ")
            await asyncio.sleep(word_interval_s)
        stream.end_input()

    pusher = asyncio.create_task(push())
    error = None
    try:
        async for ev in stream:
            if first is None:
                first = time.perf_counter() - t0
            sr = ev.frame.sample_rate
            chunks.append(np.frombuffer(ev.frame.data, dtype=np.int16).astype(np.float32) / 32768)
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"
    finally:
        pusher.cancel()
        await asyncio.gather(pusher, return_exceptions=True)
        await stream.aclose()
    total = time.perf_counter() - t0
    audio = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
    return {"ttfb_s": round(first, 3) if first is not None else None, "total_s": round(total, 3),
            "audio_s": round(len(audio) / sr, 3) if sr else 0.0, "sr": sr, "audio": audio,
            "error": error, "text_push_s": round(len(words) * word_interval_s, 3)}


def to_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr == SAMPLE_RATE or not len(audio):
        return audio
    from scipy.signal import resample_poly

    g = np.gcd(sr, SAMPLE_RATE)
    return resample_poly(audio, SAMPLE_RATE // g, sr // g).astype(np.float32)


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?।])\s+", text.strip()) if s]


async def judge(judge_model: str, lang: str, audio16: np.ndarray, ref: str) -> dict:
    stt_obj = make_stt(f"judge:{judge_model}", lang)
    lead = np.zeros(int(1.0 * SAMPLE_RATE), np.float32)
    res = await stream_clip(stt_obj, np.concatenate([lead, audio16]), tail_s=2.0, flush_wait_s=6.0)
    hyp = " ".join(e["text"] for e in res["events"] if e["type"] == "final")
    al = align(normalize(ref, lang), normalize(hyp, lang))
    sents, k = [], 0
    for s in sentences(ref):
        n = len(normalize(s, lang))
        ops = al.ref_ops[k : k + n]
        k += n
        sents.append({"text": s, "deleted_frac": round(ops.count("del") / max(n, 1), 3)})
    return {"judge": judge_model, "judge_hyp": hyp, "judge_error": res["error"],
            "rt_wer": round(al.wer, 4), "rt_errs": al.sub + al.dele + al.ins, "rt_del": al.dele,
            "n_ref": al.n_ref,
            "sentences": sents, "dropped_sentences": sum(s["deleted_frac"] >= 0.8 for s in sents)}


async def run(args) -> None:
    arms = args.arms.split(",")
    preflight(arms, "tts")
    langs = args.langs.split(",") if args.langs else list(AGENT_REPLIES)
    out = HERE / "results" / args.out
    sem = asyncio.Semaphore(args.concurrency)

    async def one(arm: str, lang: str, i: int, text: str, rep: int, tts_obj, cold: bool) -> None:
        async with sem:
            try:
                r = await synth(tts_obj, text, args.word_interval)
                audio16 = to_16k(r.pop("audio"), r["sr"] or SAMPLE_RATE)
                wav = AUDIO_DIR / arm / f"{lang}_{i:02d}_r{rep}{'_cold' if cold else ''}.wav"
                if len(audio16):
                    write_wav(wav, audio16)
                j = await judge(args.judge, lang, audio16, text) if len(audio16) and not args.no_judge else {}
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                r, j, wav = {"error": f"{type(e).__name__}: {e}"}, {}, None
            append_jsonl(out, {"arm": arm, "lang": lang, "prompt": i, "rep": rep, "cold": cold,
                               "text": text, "wav": str(wav.relative_to(HERE)) if wav else None,
                               **r, **j})
            print(f"[{arm:9}] {lang} #{i:02d} r{rep}{' cold' if cold else '     '} "
                  f"ttfb={r.get('ttfb_s')} audio={r.get('audio_s')}s rt_wer={j.get('rt_wer')} "
                  f"dropped={j.get('dropped_sentences')} {r.get('error') or ''}")

    async with utils.http_context.open():
        # Cold start: a brand-new client per trial, one short reply, nothing prewarmed.
        for t in range(args.cold_trials):
            for arm in arms:
                await one(arm, "en", 0, AGENT_REPLIES["en"][0], t, make_tts(arm, "en"), cold=True)

        jobs = []
        for rep in range(args.repeats):
            for lang in langs:
                objs = {}
                for arm in arms:
                    try:
                        objs[arm] = make_tts(arm, lang)
                    except UnsupportedLanguage as e:
                        for i, text in enumerate(AGENT_REPLIES[lang]):
                            append_jsonl(out, {"arm": arm, "lang": lang, "prompt": i, "rep": rep,
                                               "cold": False, "text": text, "wav": None,
                                               "error": f"unsupported: {e}"})
                for i, text in enumerate(AGENT_REPLIES[lang]):
                    for arm in objs:
                        jobs.append(one(arm, lang, i, text, rep, objs[arm], cold=False))
        await asyncio.gather(*jobs)
    print(f"results -> {out}")
    if not args.no_listen:
        write_listening_test(out)


def write_listening_test(results_path: Path) -> None:
    """A blind page: same text from every arm, order randomized; the key is in a separate file."""
    from .common import read_jsonl

    rows = [r for r in read_jsonl(results_path) if r.get("wav") and not r["cold"] and r["rep"] == 0]
    by_prompt: dict = {}
    for r in rows:
        by_prompt.setdefault((r["lang"], r["prompt"]), {})[r["arm"]] = r
    rng = random.Random(11)
    items, key = [], []
    for (lang, i), arms in sorted(by_prompt.items()):
        if len(arms) < 2:
            continue
        group = list(arms.values())
        rng.shuffle(group)
        n = len(items) + 1
        labels = "ABCDEF"[: len(group)]
        key.append(f"{n}: " + " ".join(f"{lb}={g['arm']}" for lb, g in zip(labels, group)))
        items.append(
            f"<div class=q><p><b>{n}.</b> [{lang}] {group[0]['text']}</p>"
            + "".join(f"<p>{lb} <audio controls src='{g['wav'].removeprefix('results/')}'></audio> "
                      f"<label><input type=radio name=q{n} value={lb}>best</label></p>"
                      for lb, g in zip(labels, group))
            + f"<label><input type=radio name=q{n} value=same>no preference</label></div>")
    html = ("<!doctype html><meta charset=utf-8><title>Blind TTS listening test</title>"
            "<style>body{font:15px system-ui;max-width:900px;margin:2em auto}.q{margin:1.2em 0;"
            "padding:.8em;border:1px solid #ddd;border-radius:8px}audio{vertical-align:middle}</style>"
            "<h2>Which sounds more natural and correct for a phone agent?</h2>" + "".join(items) +
            "<button onclick=\"alert([...document.querySelectorAll('input:checked')]"
            ".map(i=>i.name+'='+i.value).join('\\n'))\">Show my answers</button>")
    page = HERE / "results" / "listening_test.html"
    page.write_text(html)
    (HERE / "results" / "listening_test_key.txt").write_text("\n".join(key) + "\n")
    print(f"listening test -> {page} (key in listening_test_key.txt)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="nvidia,cartesia")
    ap.add_argument("--langs")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--cold-trials", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--word-interval", type=float, default=0.03, help="seconds between pushed words")
    ap.add_argument("--judge", default="speechmatics/enhanced",
                    help="round-trip STT from a vendor in neither TTS arm")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--no-listen", action="store_true")
    ap.add_argument("--out", default="tts.jsonl")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
