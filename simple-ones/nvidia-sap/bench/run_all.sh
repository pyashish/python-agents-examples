#!/usr/bin/env bash
# Full comparison: corpus -> STT -> TTS -> report. REPEATS=3 runs everything three times so the
# report shows the ordering holds run to run. One repeat takes roughly 45 min for STT (real-time
# streaming of ~530 clips per arm at concurrency 3) and 10 min for TTS.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-../venv/bin/python}
REPEATS=${REPEATS:-1}
CONC=${CONC:-3}

$PY -m bench.probe
[ -f bench/corpus/manifest.jsonl ] || $PY -m bench.build_corpus "$@"
$PY -m bench.stt_bench --arms nvidia,deepgram --repeats "$REPEATS" --concurrency "$CONC"
$PY -m bench.tts_bench --arms nvidia,cartesia --repeats "$REPEATS" --cold-trials 5
$PY -m bench.report
