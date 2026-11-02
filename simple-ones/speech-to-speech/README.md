# Speech-to-speech demo

Try **realtime (speech-to-speech)** models through [LiveKit Inference](https://docs.livekit.io/agents/models/inference/) — one model handles listening and speaking instead of separate STT, LLM, and TTS.

You can compare against the classic **cascade** pipeline in the same folder.

## Prerequisites

LiveKit Cloud credentials in the repo root `.env`:

```dotenv
LIVEKIT_URL=wss://...
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
```

Inference realtime models only need LiveKit Cloud credentials. To call **OpenAI’s Realtime API** directly, set `OPENAI_API_KEY` in `.env` and use `REALTIME_BACKEND=openai`.

## Install

From this directory (or use the repo `venv`):

```bash
pip install -r requirements.txt
```

## Run

```bash
python agent.py console   # talk in the terminal
python agent.py dev       # connect via LiveKit Agents playground
```

## Switch models

Default pipeline is **realtime** with OpenAI’s realtime model.

| Variable | Default | Purpose |
| --- | --- | --- |
| `VOICE_PIPELINE` | `realtime` | `realtime`, `cascade`, or `duplex` |
| `REALTIME_BACKEND` | `inference` | `inference` (LiveKit) or `openai` (OpenAI plugin) |
| `REALTIME_MODEL` | `openai/gpt-realtime` | Inference realtime model id |
| `REALTIME_VOICE` | (provider default) | Optional voice (both backends) |
| `OPENAI_REALTIME_MODEL` | `gpt-realtime` | OpenAI plugin model when `REALTIME_BACKEND=openai` |
| `OPENAI_REALTIME_VOICE` | `marin` | Default OpenAI plugin voice |
| `DUPLEX_MODEL` | `openai/gpt-live-1` | Full-duplex model when `VOICE_PIPELINE=duplex` |
| `DUPLEX_VOICE` | (default) | Optional GPT-Live voice |

### Example: OpenAI realtime via LiveKit Inference (default)

```bash
python agent.py console
```

### Example: OpenAI Realtime API (plugin)

Requires `OPENAI_API_KEY` in `.env`.

```bash
REALTIME_BACKEND=openai python agent.py console
```

Other OpenAI realtime models (plugin ids, no `openai/` prefix):

```bash
REALTIME_BACKEND=openai OPENAI_REALTIME_MODEL=gpt-realtime-mini OPENAI_REALTIME_VOICE=alloy python agent.py console
```

### Example: Google Gemini Live

```bash
REALTIME_MODEL=google/gemini-live python agent.py console
```

### Example: xAI Grok Voice

```bash
REALTIME_MODEL=xai/grok-voice-latest REALTIME_VOICE=eve python agent.py console
```

### Example: Compare with STT + LLM + TTS

```bash
VOICE_PIPELINE=cascade python agent.py console
```

### Example: OpenAI GPT-Live (full duplex)

```bash
VOICE_PIPELINE=duplex python agent.py console
```

## Inference realtime model ids

These work with `REALTIME_MODEL` (see LiveKit docs for the full list):

- `openai/gpt-realtime`, `openai/gpt-realtime-mini`, `openai/gpt-realtime-2.1`
- `google/gemini-live`, `google/gemini-3.8-live`
- `xai/grok-voice`, `xai/grok-voice-latest`, `xai/grok-voice-3.0`

Related examples in this repo: `docs/examples/gemini_live_vision/` (Gemini + video), `complex-agents/avatars/hedra/realtime_avatar/` (OpenAI plugin).
