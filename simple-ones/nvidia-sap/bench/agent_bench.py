"""Echo and noise-interruption tests with a real AgentSession in the loop.

The STT-only benchmark cannot see either problem: both happen while the agent is talking. Here a
simulated speaker plays the agent's TTS in real time (with pause/resume, like a room output) and a
simulated microphone hears that audio back after a delay at a chosen level (echo that acoustic echo
cancellation did not remove), mixed with scheduled noise or a real caller barge-in. Everything else
is the SDK default session the client runs, with interruptions decided either by VAD alone or by
LiveKit's adaptive detector.

Outcomes per run, while the agent speaks:
  cut_off       the agent's reply was cleared (it stopped talking for good)
  phantom_turn  a user turn was committed although nobody spoke to the agent
  stutter       the agent paused, heard no words, and resumed (a false interruption recovered)
  clean         nothing happened
For barge-in runs the agent should stop: yield latency = caller onset -> agent paused or stopped.

    python -m bench.agent_bench --stacks nvidia,livekit,assemblyai-gradium --modes vad,adaptive --repeats 2
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import random
import time
import traceback

import numpy as np
from livekit import rtc
from livekit.agents import Agent, AgentSession, inference, utils
from livekit.agents.voice import io as voice_io

from .build_corpus import _gain_to, beep, keyboard, pink_noise
from .common import HERE, SAMPLE_RATE, append_jsonl, normalize, read_wav
from .prompts import AGENT_REPLIES

TICK = 0.02
TICK_SAMPLES = int(SAMPLE_RATE * TICK)
CALLER_DB = -22.0
REPLY = AGENT_REPLIES["en"][6]  # ~12 s, five sentences
GREETING = "Hi, thanks for calling. How can I help you today?"
INJECT_AFTER_S = 4.0  # past the SDK's 3 s AEC warm-up, so interruptions are live

NOISE_FLOOR_DB = -55.0
# Silero triggers on any echo above the noise floor regardless of level, so "good" AEC means the
# residual sits below the floor; model it as fully cancelled.
GOOD_AEC_DB = None
SCENARIOS = {
    # name: (echo dB relative to the agent's own output or None, what is injected). Non-echo
    # scenarios use good AEC so they measure the noise, not the echo.
    "echo_no_aec": (-6.0, None),
    "echo_weak_aec": (-20.0, None),
    "echo_good_aec": (GOOD_AEC_DB, None),
    "beep_at_2s_greeting": (GOOD_AEC_DB, "beep"),
    "noise_babble": (GOOD_AEC_DB, "babble"),
    "noise_babble_loud": (GOOD_AEC_DB, "babble_loud"),
    "noise_keyboard": (GOOD_AEC_DB, "keyboard"),
    "bargein_long": (GOOD_AEC_DB, "caller_long"),
    "bargein_short": (GOOD_AEC_DB, "caller_short"),
}


def make_stack(name: str):
    from livekit.plugins import nvidia

    if name == "nvidia":
        return (nvidia.STT(language_code="en-US"),
                nvidia.TTS(voice="Magpie-Multilingual.EN-US.Leo", language_code="en-US"))
    if name == "livekit":
        return (inference.STT(model="deepgram/nova-3", language="en"),
                inference.TTS(model="cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"))
    if name == "assemblyai-gradium":
        return (inference.STT(model="assemblyai/universal-3-6-pro", language="en"),
                inference.TTS(model="gradium/default", voice="4SZHfMpw-p46Ywgs", language="en"))
    raise ValueError(name)


class Speaker(voice_io.AudioOutput):
    """Plays agent audio one tick at a time, driven by the microphone clock so echo is in sync."""

    def __init__(self, log) -> None:
        super().__init__(label="sim-speaker", capabilities=voice_io.AudioOutputCapabilities(pause=True))
        self._log = log
        self._queue: list[np.ndarray] = []
        self._resamplers: dict[int, rtc.AudioResampler] = {}
        self._pushed = 0.0
        self._played = 0.0
        self._started = False
        self._flushed = False
        self._interrupted = False
        self._paused = False
        self.segment = 0

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)
        if frame.sample_rate != SAMPLE_RATE:
            rs = self._resamplers.setdefault(
                frame.sample_rate, rtc.AudioResampler(frame.sample_rate, SAMPLE_RATE, num_channels=1))
            frames = rs.push(frame)
        else:
            frames = [frame]
        for f in frames:
            self._queue.append(np.frombuffer(f.data, dtype=np.int16).astype(np.float32) / 32768)
        self._pushed += frame.duration

    def flush(self) -> None:
        super().flush()
        if self._pushed:
            self._flushed = True

    def clear_buffer(self) -> None:
        if self._pushed:
            self._interrupted = True
            self._queue.clear()
            self._log("speaker.cleared", f"segment {self.segment} after {self._played:.1f}s")

    def pause(self) -> None:
        super().pause()
        if not self._paused:
            self._log("speaker.paused", f"segment {self.segment} at {self._played:.1f}s")
        self._paused = True

    def resume(self) -> None:
        super().resume()
        if self._paused:
            self._log("speaker.resumed", f"segment {self.segment} at {self._played:.1f}s")
        self._paused = False

    def tick(self) -> np.ndarray:
        out = np.zeros(TICK_SAMPLES, np.float32)
        if self._queue and not self._interrupted and not self._paused:
            if not self._started:
                self._started = True
                self.on_playback_started(created_at=time.time())
                self._log("speaker.started", f"segment {self.segment}")
            buf = np.concatenate(self._queue)
            take = buf[:TICK_SAMPLES]
            out[: len(take)] = take
            rest = buf[TICK_SAMPLES:]
            self._queue = [rest] if len(rest) else []
            self._played += len(take) / SAMPLE_RATE
        if self._flushed and (self._interrupted or not self._queue):
            self._log("speaker.finished", f"segment {self.segment} played {self._played:.1f}s"
                      f"{' INTERRUPTED' if self._interrupted else ''}")
            self.on_playback_finished(playback_position=self._played, interrupted=self._interrupted)
            self._pushed = self._played = 0.0
            self._started = self._flushed = self._interrupted = self._paused = False
            self._queue.clear()
            self.segment += 1
        return out


class Mic(voice_io.AudioInput):
    """Real-time 20 ms frames: delayed, attenuated speaker output plus scheduled overlays."""

    def __init__(self, speaker: Speaker, echo_db: float | None, rng: np.random.Generator,
                 echo_delay_s: float = 0.12) -> None:
        super().__init__(label="sim-mic")
        self._speaker = speaker
        self._gain = 0.0 if echo_db is None else 10 ** (echo_db / 20)
        self._floor = _gain_to(pink_noise(SAMPLE_RATE * 10, rng), NOISE_FLOOR_DB)
        self._delay = int(echo_delay_s / TICK)
        self._history: list[np.ndarray] = []
        self._overlays: list[tuple[int, np.ndarray]] = []
        self.k = 0
        self._t0: float | None = None

    def add_overlay(self, audio: np.ndarray, delay_s: float = 0.0) -> float:
        start = self.k + int(delay_s / TICK)
        self._overlays.append((start, audio))
        return start * TICK

    async def __anext__(self) -> rtc.AudioFrame:
        if self._t0 is None:
            self._t0 = time.perf_counter()
        wait = self._t0 + self.k * TICK - time.perf_counter()
        if wait > 0:
            await asyncio.sleep(wait)
        self._history.append(self._speaker.tick())
        f0 = (self.k * TICK_SAMPLES) % (len(self._floor) - TICK_SAMPLES)
        frame = self._floor[f0 : f0 + TICK_SAMPLES].copy()
        if self.k >= self._delay:
            frame += self._history[self.k - self._delay] * self._gain
        a = self.k * TICK_SAMPLES
        for start, audio in self._overlays:
            off = a - start * TICK_SAMPLES
            if 0 <= off < len(audio):
                seg = audio[off : off + TICK_SAMPLES]
                frame[: len(seg)] += seg
        self.k += 1
        pcm = (np.clip(frame, -1, 1) * 32767).astype(np.int16).tobytes()
        return rtc.AudioFrame(pcm, SAMPLE_RATE, 1, TICK_SAMPLES)


def injected_audio(kind: str, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    speech_dir = HERE / "corpus" / "speech"
    if kind == "beep":
        return beep(None), ""
    if kind in ("babble", "babble_loud"):
        files = sorted(speech_dir.glob("en*_*.wav"))
        n = int(4.0 * SAMPLE_RATE)
        out = np.zeros(n, np.float32)
        for f in random.Random(int(rng.integers(1 << 30))).sample(files, 6):
            s = np.resize(_gain_to(read_wav(f), CALLER_DB), n)
            out += s
        return _gain_to(out, CALLER_DB - (6 if kind == "babble_loud" else 12)), ""
    if kind == "keyboard":
        return _gain_to(keyboard(int(4.0 * SAMPLE_RATE), rng), CALLER_DB - 6), ""
    if kind == "caller_long":
        return _gain_to(read_wav(speech_dir / "en_Edward_12.wav"), CALLER_DB), \
            "Hold on a second, let me grab my account details."
    if kind == "caller_short":
        return _gain_to(read_wav(speech_dir / "en_Edward_02.wav"), CALLER_DB), "No, that's not right."
    raise ValueError(kind)


async def run_one(stack: str, mode: str, scenario: str, rep: int, out) -> None:
    from livekit.plugins import silero

    echo_db, kind = SCENARIOS[scenario]
    rng = np.random.default_rng(rep * 1000 + list(SCENARIOS).index(scenario))
    events: list[dict] = []
    t_start = time.perf_counter()

    def log(kind_: str, detail: str = "") -> None:
        events.append({"t": round(time.perf_counter() - t_start, 3), "ev": kind_, "detail": detail})

    stt, tts = make_stack(stack)
    session = AgentSession(
        stt=stt, tts=tts, llm=inference.LLM(model="openai/gpt-4.1-mini"),
        vad=silero.VAD.load(),
        turn_handling={"interruption": {"mode": mode}},
    )
    speaker = Speaker(log)
    mic = Mic(speaker, echo_db, rng)
    session.input.audio = mic
    session.output.audio = speaker

    @session.on("user_input_transcribed")
    def _stt(ev) -> None:
        if ev.is_final and ev.transcript.strip():
            log("stt.final", ev.transcript)

    @session.on("conversation_item_added")
    def _item(ev) -> None:
        role = getattr(ev.item, "role", None)
        if role == "user":
            log("user_turn", ev.item.text_content or "")
        elif role == "assistant":
            log("agent_turn", ev.item.text_content or "")

    @session.on("agent_false_interruption")
    def _false(ev) -> None:
        log("false_interruption")

    error = None
    onset = None
    ref = ""
    text = GREETING if scenario.startswith("beep") else REPLY
    try:
        await session.start(agent=Agent(instructions=(
            "You are a friendly voice assistant for a support line. Keep replies to one sentence.")))
        log("session.started", f"mode={mode}")
        handle = session.say(text, allow_interruptions=True)
        if kind == "beep":
            # The client's beep: about 2 s into the call, while the greeting plays.
            onset = mic.add_overlay(*injected_audio(kind, rng)[:1], delay_s=2.0 - mic.k * TICK)
            log("inject", f"beep at {onset:.2f}s")
        else:
            started = asyncio.get_running_loop().create_future()
            orig = speaker.on_playback_started

            def _started(*, created_at: float) -> None:
                orig(created_at=created_at)
                if not started.done():
                    started.set_result(mic.k * TICK)

            speaker.on_playback_started = _started  # type: ignore[method-assign]
            await asyncio.wait_for(started, 30)
            if kind:
                audio, ref = injected_audio(kind, rng)
                onset = mic.add_overlay(audio, delay_s=INJECT_AFTER_S)
                log("inject", f"{kind} at mic {onset:.2f}s {ref}")
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(handle.wait_for_playout(), 40)
        await asyncio.sleep(6.0)  # let any phantom turn commit and the agent answer it
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        with contextlib.suppress(Exception):
            await session.aclose()

    # Times above are wall-clock from t_start; the mic clock started at the first frame.
    mic_offset = (mic._t0 - t_start) if mic._t0 else 0.0
    onset_wall = onset + mic_offset if onset is not None else None
    first_seg = [e for e in events if e["detail"].startswith("segment 0")]
    cut_off = any(e["ev"] == "speaker.cleared" for e in first_seg)
    paused = [e for e in first_seg if e["ev"] == "speaker.paused"]
    resumed = any(e["ev"] == "speaker.resumed" for e in first_seg)
    user_turns = [e for e in events if e["ev"] == "user_turn"]
    yield_s = None
    if onset_wall is not None:
        stops = [e["t"] for e in first_seg if e["ev"] in ("speaker.paused", "speaker.cleared")
                 and e["t"] >= onset_wall - 0.1]
        yield_s = round(min(stops) - onset_wall, 3) if stops else None
    if scenario.startswith("bargein"):
        outcome = "yielded" if (cut_off or paused) else "talked_over_caller"
    elif user_turns:
        outcome = "phantom_turn"
    elif cut_off:
        outcome = "cut_off"
    elif paused:
        outcome = "stutter"
    else:
        outcome = "clean"
    agent_words = set(normalize(" ".join([text] + [e["detail"] for e in events if e["ev"] == "agent_turn"])))
    turn_words = [w for e in user_turns for w in normalize(e["detail"])]
    caller_words = set(normalize(ref)) if ref else set()
    echo_words = [w for w in turn_words if w in agent_words and w not in caller_words]
    row = {"stack": stack, "mode": mode, "scenario": scenario, "rep": rep, "echo_db": echo_db,
           "outcome": outcome, "cut_off": cut_off, "paused": len(paused), "resumed": resumed,
           "user_turns": [e["detail"] for e in user_turns], "n_user_turns": len(user_turns),
           "echo_word_frac": round(len(echo_words) / len(turn_words), 2) if turn_words else None,
           "yield_s": yield_s, "error": error, "events": events}
    append_jsonl(out, row)
    print(f"[{stack:18} {mode:8}] {scenario:20} r{rep} -> {outcome:18} "
          f"turns={[t[:40] for t in row['user_turns']]} yield={yield_s} {error or ''}")


async def main_async(args) -> None:
    out = HERE / "results" / args.out
    sem = asyncio.Semaphore(args.concurrency)
    scenarios = args.scenarios.split(",") if args.scenarios else list(SCENARIOS)

    async def guarded(*a) -> None:
        async with sem:
            await run_one(*a, out)

    reps = range(args.first_rep, args.first_rep + args.repeats)
    jobs = [guarded(s, m, sc, r) for r in reps for sc in scenarios
            for m in args.modes.split(",") for s in args.stacks.split(",")]
    from livekit.agents.testing import fake_job_context

    async with utils.http_context.open():
        with fake_job_context():
            await asyncio.gather(*jobs)
    print(f"results -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stacks", default="nvidia,livekit,assemblyai-gradium")
    ap.add_argument("--modes", default="vad,adaptive")
    ap.add_argument("--scenarios")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--first-rep", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out", default="agent.jsonl")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
