# Reminder agent with a voicemail detection observer

This outbound-style appointment-reminder agent uses LiveKit's
[background observer pattern](https://livekit.com/blog/observer-pattern-voice-agent-guardrails)
and `inference.DecisionModel("livekit/clef-omni")` to decide who answered the
call before the agent says anything.

As on a real outbound call, the callee speaks first. The observer:

1. Classifies the opening audio transcript as `human`, `voicemail`, `ivr`,
   `unavailable`, or `uncertain`.
2. Logs a real-time, pretty-printed `[VOICEMAIL OBSERVER] DecisionModel trace`.
3. Acts on a machine verdict only when its confidence is at least `0.7`;
   anything less is treated as `uncertain`, which proceeds as a human.
4. Waits for up to two turns if a voicemail greeting looks unfinished, so the
   agent doesn't talk over it.
5. Stops listening once it has a verdict.

| Verdict | Action |
| --- | --- |
| `human`, `uncertain` | The agent introduces itself and continues normally. |
| `voicemail` | Leaves a fixed reminder message, waits for playout, deletes the room. |
| `ivr`, `unavailable` | Logs the result and deletes the room. |

If nobody speaks for 10 seconds, the agent asks whether anyone is there; the
next turn is still classified, so a late voicemail greeting is caught. Model or
network errors are logged and
treated as `uncertain`, so the agent never hangs up on a possible human because
of a failure.

The agent's `on_user_turn_completed` hook passes each opening turn to the
observer and raises `StopResponse` until the observer confirms a human, so the
main LLM never replies to a voicemail greeting.

## Run

Use LiveKit Cloud credentials in `.env`:

```dotenv
LIVEKIT_URL=wss://...
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
```

Then:

```bash
pip install -r simple-ones/voicemail-detection-observer/requirements.txt
cd simple-ones/voicemail-detection-observer
lk agent console agent.py
```

Test by voice, not with `--text`. Typed turns skip `on_user_turn_completed`, so
the observer never sees them. To script spoken turns, use the debugger in audio
mode:

```bash
lk agent debugger start agent.py --audio
lk agent debugger say "Hi, you've reached Sam. Please leave a message after the tone."
lk agent debugger stop
```

In the console, deleting the room logs a `not_found` error because the console
uses a mock room; the session still shuts down.

## Try it

Play the callee. Each line below is a separate session:

- Human: "Hello?"
- Voicemail: "Hi, you've reached Sam. I can't take your call right now. Please
  leave a message after the tone."
- IVR: "Thank you for calling. For billing, press 1. For support, press 2."
- Unavailable: "The mailbox you are trying to reach is full and cannot accept
  new messages."
- No speech: stay silent for 10 seconds.

## Production outbound calls

For real outbound SIP calls, prefer the built-in
[answering machine detection](https://docs.livekit.io/telephony/features/answering-machine-detection/)
(`livekit.agents.AMD`, available in `livekit-agents` 1.8.10). It combines a
timing heuristic with an LLM classifier, pauses agent speech during detection,
and can hand off to IVR navigation. This example shows the observer approach,
which you can test in the console and tune for your own domain.
