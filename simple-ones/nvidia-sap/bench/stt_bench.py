"""Stream every corpus clip through each STT arm in real time and record what comes back, when.

Each clip gets a fresh stream, the way each call does, so start-of-stream warm-up is part of every
measurement. Audio is paced at 20 ms per 20 ms frame; endpointing is time-based and dumping a file
at full speed would not reproduce it. After the clip, a silent tail is streamed (default 3 s) and
only then is input closed: a transcript that arrives only after the close is a turn the recognizer
never ended on its own, which in a live call is a stall or a lost turn.

    python -m bench.stt_bench --arms nvidia,deepgram --repeats 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
import traceback

import numpy as np
from livekit import rtc
from livekit.agents import stt as agents_stt
from livekit.agents import utils

from .common import (
    FRAME_SAMPLES,
    HERE,
    SAMPLE_RATE,
    align,
    append_jsonl,
    make_stt,
    normalize,
    preflight,
    read_wav,
    to_pcm16,
)

TEXT_EVENTS = {
    agents_stt.SpeechEventType.INTERIM_TRANSCRIPT: "interim",
    agents_stt.SpeechEventType.FINAL_TRANSCRIPT: "final",
}


async def stream_clip(stt_obj, audio: np.ndarray, tail_s: float, flush_wait_s: float) -> dict:
    events: list[dict] = []
    state = {"flushed": False}
    pcm = to_pcm16(np.concatenate([audio, np.zeros(int(tail_s * SAMPLE_RATE), np.float32)]))
    step = FRAME_SAMPLES * 2
    t_create = time.perf_counter()
    stream = stt_obj.stream()
    t0 = time.perf_counter()

    async def reader() -> None:
        async for ev in stream:
            kind = TEXT_EVENTS.get(ev.type)
            if kind is None:
                continue
            text = ev.alternatives[0].text if ev.alternatives else ""
            if text.strip():
                events.append({"t": round(time.perf_counter() - t0, 3), "type": kind,
                               "text": text, "after_flush": state["flushed"]})

    task = asyncio.create_task(reader())
    error = None
    flush_t = -1.0
    try:
        for i in range(0, len(pcm), step):
            target = t0 + (i // step) * FRAME_SAMPLES / SAMPLE_RATE
            delay = target - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            if task.done():
                break
            chunk = pcm[i : i + step]
            stream.push_frame(rtc.AudioFrame(chunk, SAMPLE_RATE, 1, len(chunk) // 2))
        state["flushed"] = True
        flush_t = time.perf_counter() - t0
        stream.end_input()
        await asyncio.wait_for(asyncio.shield(task), flush_wait_s)
    except TimeoutError:
        pass
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"
    finally:
        if task.done() and task.exception() and error is None:
            exc = task.exception()
            error = f"{type(exc).__name__}: {exc} {exc.__cause__!r}"
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await stream.aclose()
    return {"events": events, "error": error, "flush_t": round(flush_t, 3),
            "setup_s": round(t0 - t_create, 4)}


def score(clip: dict, res: dict) -> dict:
    ev = res["events"]
    finals = [e for e in ev if e["type"] == "final"]
    spans = clip["speech_spans"]
    m: dict = {
        "hyp": " ".join(e["text"] for e in finals),
        "n_text_events": len(ev),
        "finals_after_flush": sum(e["after_flush"] for e in finals),
    }
    if clip["kind"] == "noise_only":
        m["false_interim"] = any(e["type"] == "interim" for e in ev)
        m["false_final"] = bool(finals)
        if "beep_span" in clip:
            m["beep_transcribed"] = bool(ev)
        return m

    ref = normalize(clip["ref"], clip["lang"])
    al = align(ref, normalize(m["hyp"], clip["lang"]))
    m.update(wer=round(al.wer, 4), sub=al.sub, dele=al.dele, ins=al.ins, n_ref=al.n_ref)
    onset, end = spans[0][0], spans[-1][1]
    first = next((e["t"] for e in ev if e["t"] >= onset - 0.2), None)
    m["first_text_after_onset_s"] = round(first - onset, 3) if first is not None else None
    m["text_before_onset"] = [e["text"] for e in ev if e["t"] < onset - 0.2]
    if clip["kind"] == "beep":
        m["beep_transcribed"] = bool(m["text_before_onset"])

    if clip["kind"] != "session":
        on_time = [e["t"] for e in finals if not e["after_flush"]]
        m["endpoint_missed"] = m["finals_after_flush"] > 0 or not finals
        m["final_latency_s"] = (round(max(on_time) - end, 3)
                                if on_time and not m["endpoint_missed"] else None)
        return m

    # Per utterance: dropped = the alignment deletes >= 80% of its words; latency = last final
    # that landed in the utterance's window minus the end of the utterance's speech.
    utts = []
    k = 0
    for idx, (u_ref, (s, e)) in enumerate(zip(clip["utterance_refs"], spans, strict=True)):
        n = len(normalize(u_ref, clip["lang"]))
        ops = al.ref_ops[k : k + n]
        k += n
        nxt = spans[idx + 1][0] if idx + 1 < len(spans) else float("inf")
        win = [f["t"] for f in finals if s < f["t"] <= nxt and not f["after_flush"]]
        utts.append({
            "i": idx, "ref": u_ref, "n": n,
            "deleted_frac": round(ops.count("del") / max(n, 1), 3),
            "dropped": ops.count("del") / max(n, 1) >= 0.8,
            "latency_s": round(max(win) - e, 3) if win else None,
        })
    m["utterances"] = utts
    m["dropped"] = sum(u["dropped"] for u in utts)
    m["n_utterances"] = len(utts)
    return m


async def run(args) -> None:
    clips = [json.loads(line) for line in (HERE / "corpus" / "manifest.jsonl").read_text().splitlines()]
    if args.kinds:
        clips = [c for c in clips if c["kind"] in args.kinds.split(",")]
    if args.langs:
        clips = [c for c in clips if c["lang"] in args.langs.split(",")]
    if args.match:
        clips = [c for c in clips if re.search(args.match, c["id"])]
    if args.limit:
        clips = clips[: args.limit]
    arms = args.arms.split(",")
    preflight(arms, "asr")
    out = HERE / "results" / args.out
    sem = asyncio.Semaphore(args.concurrency)
    stt_cache: dict = {}

    async def one(arm: str, clip: dict, rep: int) -> None:
        async with sem:
            key = (arm, clip["lang"])
            try:
                if key not in stt_cache:
                    stt_cache[key] = make_stt(arm, clip["lang"])
                res = await stream_clip(stt_cache[key], read_wav(HERE / clip["path"]),
                                        args.tail, args.flush_wait)
                metrics = score(clip, res)
            except Exception as e:  # noqa: BLE001
                res, metrics = {"events": [], "error": f"{type(e).__name__}: {e}"}, {}
                traceback.print_exc()
            row = {"arm": arm, "clip": clip["id"], "rep": rep,
                   **{k: clip.get(k) for k in ("kind", "lang", "voice", "noise", "snr", "source", "ref")},
                   "error": res["error"], "setup_s": res.get("setup_s"), **metrics,
                   "events": res["events"]}
            append_jsonl(out, row)
            flag = f"ERR {res['error'][:80]}" if res["error"] else (
                f"wer={metrics.get('wer')}" if "wer" in metrics else
                f"false_final={metrics.get('false_final')}")
            print(f"[{arm:14}] r{rep} {clip['id']:40} {flag}")

    # Clip-major order: every arm sees a clip before the next, so a partial run is still paired.
    jobs = [one(arm, c, r) for r in range(args.repeats) for c in clips for arm in arms]
    async with utils.http_context.open():
        await asyncio.gather(*jobs)
    print(f"results -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="nvidia,deepgram")
    ap.add_argument("--kinds", help="comma list: clean,noisy,beep,noise_only,session")
    ap.add_argument("--langs")
    ap.add_argument("--match", help="regex on clip id")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--tail", type=float, default=3.0, help="seconds of silence streamed after a clip")
    ap.add_argument("--flush-wait", type=float, default=6.0)
    ap.add_argument("--out", default="stt.jsonl")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
