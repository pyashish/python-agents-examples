# Math agent with an off-topic observer

This math-only voice agent uses LiveKit's
[background observer pattern](https://livekit.com/blog/observer-pattern-voice-agent-guardrails)
and `inference.DecisionModel("livekit/clef-omni")`.

For every completed user turn, the observer:

1. Classifies that individual turn as `on_topic` or `off_topic`.
2. Logs a real-time, pretty-printed `[TOPIC OBSERVER] DecisionModel trace`.
3. Counts off-topic turns cumulatively across the call.
4. Gives a spoken warning after strikes one and two.
5. After strike three, speaks a closing message, waits for it to finish, and
   deletes the LiveKit room to disconnect everyone.

Brief greetings, thanks, goodbyes, and requests to repeat or clarify are allowed.
Model or network errors are logged and never counted as violations.

## Run

Use LiveKit Cloud credentials in `.env`:

```dotenv
LIVEKIT_URL=wss://...
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
```

Then:

```bash
pip install -r simple-ones/math-offscript-observer/requirements.txt
cd simple-ones/math-offscript-observer
lk agent console agent.py
```

Do not use `python agent.py console`; that legacy console runner is deprecated.
Use `lk agent console agent.py --text` to test without microphone, speaker, STT,
or TTS.

The debug log includes the complete model response and policy state:

```text
[TOPIC OBSERVER] DecisionModel trace:
{
  "policy_state": {
    "selected": "off_topic",
    "off_topic_count": 1,
    "strikes_remaining": 2
  },
  "response": {
    "model": "clef-omni",
    "results": { ... }
  }
}
```
