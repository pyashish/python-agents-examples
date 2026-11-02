"""Turn results/stt.jsonl and results/tts.jsonl into a markdown report keyed to the client's issues.

    python -m bench.report [--stt stt.jsonl] [--tts tts.jsonl] [--baseline nvidia]

Every number is printed with its denominator. Paired deltas (same clip, both arms) carry a 95%
bootstrap interval; a delta whose interval crosses zero is reported as not shown.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from statistics import median

import numpy as np

from .common import HERE, read_jsonl


def pct(xs) -> str:
    xs = list(xs)
    return f"{100 * sum(xs) / len(xs):.1f}% ({sum(xs)}/{len(xs)})" if xs else "n/a"


def p(xs, q) -> float | None:
    xs = [x for x in xs if x is not None]
    return float(np.percentile(xs, q)) if xs else None


def fmt_s(x) -> str:
    return "n/a" if x is None else f"{x * 1000:.0f} ms"


def lat(xs) -> str:
    xs = [x for x in xs if x is not None]
    return f"{fmt_s(p(xs, 50))} / {fmt_s(p(xs, 95))} (n={len(xs)})" if xs else "n/a"


def mean_wer(rows) -> str:
    rows = [r for r in rows if r.get("n_ref")]
    if not rows:
        return "n/a"
    errs = sum(r["sub"] + r["dele"] + r["ins"] for r in rows)
    return f"{100 * errs / sum(r['n_ref'] for r in rows):.1f}% (clips={len(rows)})"


def rt_wer(rows) -> str:
    rows = [r for r in rows if r.get("n_ref")]
    if not rows:
        return "n/a"
    errs = sum(r.get("rt_errs", round(r["rt_wer"] * r["n_ref"])) for r in rows)
    return f"{100 * errs / sum(r['n_ref'] for r in rows):.1f}% (n={len(rows)})"


def paired(rows, a, b, key, by=("clip", "rep"), n_boot=4000) -> str:
    """Mean of (b - a) over items both arms completed, with a 95% bootstrap interval."""
    va = {tuple(r[k] for k in by): r.get(key) for r in rows if r["arm"] == a}
    vb = {tuple(r[k] for k in by): r.get(key) for r in rows if r["arm"] == b}
    d = np.array([vb[k] - va[k] for k in va.keys() & vb.keys()
                  if va[k] is not None and vb[k] is not None])
    if len(d) < 5:
        return f"n={len(d)}, too few pairs"
    rng = np.random.default_rng(0)
    boots = rng.choice(d, (n_boot, len(d))).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    verdict = "shown" if lo > 0 or hi < 0 else "NOT shown (interval crosses 0)"
    return f"{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] n={len(d)}: {verdict}"


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def stt_section(rows: list[dict], arms: list[str], base: str) -> list[str]:
    ok = [r for r in rows if not r.get("error")]
    by = lambda arm, **f: [r for r in ok if r["arm"] == arm and all(r.get(k) == v for k, v in f.items())]  # noqa: E731
    en = lambda r: str(r.get("lang", "")).startswith("en")  # noqa: E731
    out = ["## STT"]

    out += ["### Reliability", table(
        ["arm", "streams that errored", "rows"],
        [[a, pct(bool(r.get("error")) for r in rows if r["arm"] == a),
          sum(r["arm"] == a for r in rows)] for a in arms])]
    errs = defaultdict(int)
    for r in rows:
        if r.get("error"):
            errs[(r["arm"], r["error"][:110])] += 1
    if errs:
        out.append("\nErrors seen:\n" + "\n".join(f"- `{a}` x{n}: {e}" for (a, e), n in
                                                 sorted(errs.items(), key=lambda x: -x[1])[:8]))

    out += ["### 1. Latency (clean English, fresh stream per clip)", table(
        ["arm", "end of speech -> final transcript p50 / p95", "speech onset -> first words p50 / p95",
         "turn never endpointed (needed input close)"],
        [[a, lat(r.get("final_latency_s") for r in by(a, kind="clean") if en(r)),
          lat(r.get("first_text_after_onset_s") for r in by(a, kind="clean") if en(r)),
          pct(r.get("endpoint_missed") for r in by(a, kind="clean") if en(r))] for a in arms])]
    def sess(a, noise=None):
        return [u for r in by(a, kind="session") if noise in (None, r.get("noise"))
                for u in r["utterances"]]

    out += ["\nWarm-up and noise inside a call (continuous sessions; end of speech -> final):", table(
        ["arm", "quiet: 1st utterance p50 / p95", "quiet: later utterances p50 / p95",
         "babble 10 dB: all utterances p50 / p95"],
        [[a, lat(u["latency_s"] for u in sess(a, "pink") if u["i"] == 0),
          lat(u["latency_s"] for u in sess(a, "pink") if u["i"] > 0),
          lat(u["latency_s"] for u in sess(a, "babble"))] for a in arms])]
    for a in arms:
        if a != base:
            clean_en = [r for r in ok if r["kind"] == "clean" and en(r)]
            out.append(f"\nPaired final-transcript latency, {a} minus {base} (s): "
                       f"{paired(clean_en, base, a, 'final_latency_s')}")

    beep_rows = lambda a: by(a, kind="beep") + [r for r in by(a, kind="noise_only") if r.get("noise") == "beep"]  # noqa: E731
    out += ["### 2. The 2-second beep", table(
        ["arm", "beep produced text (a phantom user turn)", "WER on the speech after the beep"],
        [[a, pct(r.get("beep_transcribed") for r in beep_rows(a)), mean_wer(by(a, kind="beep"))]
         for a in arms])]

    out += ["### 3. Dropped turns (continuous multi-utterance sessions)", table(
        ["arm", "quiet: utterances dropped (>=80% of words missing)", "babble 10 dB: dropped",
         "session streams that errored"],
        [[a, pct(u["dropped"] for u in sess(a, "pink")), pct(u["dropped"] for u in sess(a, "babble")),
          pct(bool(r.get("error")) for r in rows if r["arm"] == a and r.get("kind") == "session")]
         for a in arms])]
    for a in arms:
        drops = [(u["ref"]) for u in sess(a) if u["dropped"]]
        if drops:
            out.append(f"\n`{a}` dropped: " + "; ".join(f'"{d}"' for d in drops[:8]))

    beds = sorted({r["noise"] for r in ok if r["kind"] == "noise_only" and r.get("noise") != "beep"})
    out += ["### 4. Background noise", "Noise-only clips (no one is speaking):", table(
        ["arm"] + [f"{b}: false final / false partial" for b in beds],
        [[a] + [f"{pct(r['false_final'] for r in by(a, kind='noise_only', noise=b))} / "
                f"{pct(r['false_interim'] for r in by(a, kind='noise_only', noise=b))}" for b in beds]
         for a in arms])]
    conds = sorted({(r["noise"], r["snr"]) for r in ok if r["kind"] == "noisy"}, key=lambda x: (x[0], -x[1]))
    out += ["\nWER with background noise (English):", table(
        ["arm", "clean"] + [f"{n} {s} dB" for n, s in conds],
        [[a, mean_wer([r for r in by(a, kind="clean") if en(r)])] +
         [mean_wer(by(a, kind="noisy", noise=n, snr=s)) for n, s in conds] for a in arms])]
    for a in arms:
        if a != base:
            out.append(f"\nPaired WER, {a} minus {base}, all noisy clips: "
                       f"{paired([r for r in ok if r['kind'] == 'noisy'], base, a, 'wer')}")

    langs = sorted({r["lang"] for r in rows if r.get("kind") == "clean"})
    out += ["### 5. Languages and accents (clean)", table(
        ["arm"] + langs,
        [[a] + [mean_wer(by(a, kind="clean", lang=lg)) + (
            f", errors {pct(bool(r.get('error')) for r in rows if r['arm'] == a and r.get('kind') == 'clean' and r['lang'] == lg)}"
            if any(r.get("error") for r in rows if r["arm"] == a and r.get("lang") == lg) else "")
            for lg in langs] for a in arms])]
    return out


def tts_section(rows: list[dict], arms: list[str], base: str) -> list[str]:
    ok = [r for r in rows if not r.get("error")]
    out = ["## TTS", "### 1. Latency (time to first audio, from the first word pushed)", table(
        ["arm", "cold start p50 / p95", "warm p50 / p95", "real-time factor (synth time / audio)"],
        [[a, lat(r["ttfb_s"] for r in ok if r["arm"] == a and r["cold"]),
          lat(r["ttfb_s"] for r in ok if r["arm"] == a and not r["cold"]),
          f"{median([r['total_s'] / r['audio_s'] for r in ok if r['arm'] == a and r.get('audio_s')] or [0]):.2f}"]
         for a in arms])]
    for a in arms:
        if a != base:
            warm = [r for r in ok if not r["cold"]]
            out.append(f"\nPaired warm TTFB, {a} minus {base} (s): "
                       f"{paired(warm, base, a, 'ttfb_s', by=('lang', 'prompt', 'rep'))}")
    judged = [r for r in ok if "sentences" in r and not r["cold"]]
    out += ["### 3. Dropped or truncated responses", table(
        ["arm", "requests that errored", "requests with no audio",
         "sentences missing from the audio (judge STT)", "round-trip WER (judge STT)"],
        [[a, pct(bool(r.get("error")) for r in rows if r["arm"] == a),
          pct(not r.get("audio_s") for r in rows if r["arm"] == a),
          pct(s["deleted_frac"] >= 0.8 for r in judged if r["arm"] == a for s in r["sentences"]),
          rt_wer(r for r in judged if r["arm"] == a)] for a in arms])]
    langs = sorted({r["lang"] for r in rows})
    out += ["### 5. Languages (round-trip WER per language; errors in brackets)", table(
        ["arm"] + langs,
        [[a] + [rt_wer(r for r in judged if r["arm"] == a and r["lang"] == lg) +
                (f" [{sum(1 for r in rows if r['arm'] == a and r['lang'] == lg and (r.get('error') or not r.get('audio_s')))} failed]"
                 if any(r["arm"] == a and r["lang"] == lg and (r.get("error") or not r.get("audio_s")) for r in rows) else "")
                for lg in langs] for a in arms])]
    out.append("\nListen blind: `results/listening_test.html` (answer key in `listening_test_key.txt`).")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stt", default="stt.jsonl")
    ap.add_argument("--tts", default="tts.jsonl")
    ap.add_argument("--baseline", default="nvidia")
    ap.add_argument("--out", default="REPORT.md")
    a = ap.parse_args()
    stt_rows = read_jsonl(HERE / "results" / a.stt)
    tts_rows = read_jsonl(HERE / "results" / a.tts)
    lines = ["# NVIDIA (Parakeet / Magpie) vs alternative STT / TTS stacks",
             "Latencies are measured from this machine, not from production's region; "
             "compare arms to each other, not to production numbers."]
    order = lambda rows: sorted({r["arm"] for r in rows}, key=lambda x: (x != a.baseline, x))  # noqa: E731
    if stt_rows:
        lines += stt_section(stt_rows, order(stt_rows), a.baseline)
    if tts_rows:
        lines += tts_section(tts_rows, order(tts_rows), a.baseline)
    text = "\n\n".join(lines) + "\n"
    (HERE / "results" / a.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
