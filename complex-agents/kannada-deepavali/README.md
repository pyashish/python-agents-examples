# Kannada Deepavali + LemonSlice

End-to-end voice demo: speak in Kannada with a **LemonSlice** avatar, **LiveKit Inference** for STT/LLM/TTS (no Sarvam key), and optional **your own voice** from LiveKit Cloud.

Inspired by [avatars/lemonslice](../avatars/lemonslice); this project is self-contained.

## Prerequisites

- LiveKit Cloud project (`LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`)
- [LemonSlice API key](https://lemonslice.com/agents/api) (`LEMONSLICE_API_KEY`)
- **STT / LLM / TTS** via [LiveKit Inference](https://docs.livekit.io/agents/models/inference/) on your Cloud project (usage on your plan—no separate Deepgram/Cartesia accounts)

### Your voice (recorded in LiveKit)

1. Open [LiveKit Cloud](https://cloud.livekit.io) → your project → **Voices** → **Custom voices**.
2. **Record audio** (~10s) or upload a sample → clone completes in under a minute.
3. Copy the voice ID (`v_…`) into `agent/.env.local` as `LIVEKIT_CUSTOM_VOICE_ID`.

Docs: [Custom voices](https://docs.livekit.io/agents/models/tts/custom-voices/). Custom voice cloning requires a **paid Cloud plan (Ship or higher)**; the free tier still runs the agent with a default Inference TTS voice if you leave `LIVEKIT_CUSTOM_VOICE_ID` empty.

**Note:** Clone setup languages in the dashboard include Hindi but not Kannada yet—record in the language closest to your speech (often **Hindi** or **English**). The agent still *writes* Kannada; your clone’s accent follows the sample. For best Kannada *pronunciation*, a dedicated Indic TTS (e.g. Sarvam) is stronger when you add an API key later.

### Kannada STT without Sarvam

Default: `deepgram/nova-3` with `STT_LANGUAGE=multi` (multilingual). Quality for Kannada may be below Sarvam `kn-IN`; tune `STT_MODEL` / `STT_LANGUAGE` in `.env.local` if needed.
- Python 3.11+ with [uv](https://docs.astral.sh/uv/)
- Node.js with pnpm

## Avatar image

Add your portrait as **`agent/assets/avatar.jpg`** (see `agent/assets/README.md`). The agent uploads it to LemonSlice via `agent_image`.

```bash
cd agent/assets
curl -L -o avatar.jpg "https://iili.io/frL9tuj.png"   # placeholder only
```

## Setup

**Agent**

```bash
cd agent
cp .env.example .env.local   # LiveKit + LemonSlice + optional LIVEKIT_CUSTOM_VOICE_ID
uv sync
```

**Frontend**

```bash
cd frontend
cp .env.example .env.local   # same LiveKit creds; AGENT_NAME=kannada-deepavali
pnpm install
```

## Run locally

Terminal A:

```bash
cd agent
uv run agent.py dev
```

Terminal B:

```bash
cd frontend
pnpm dev
```

Open http://localhost:3000 → allow microphone → **Start call**. You should hear Kannada TTS and see the avatar video tile.

## Agent dispatch

The agent registers as `kannada-deepavali`. The frontend sets `AGENT_NAME=kannada-deepavali` so the token API dispatches explicitly.

## Model stack

| Component | Default |
|-----------|---------|
| STT | `deepgram/nova-3`, `language=multi` (Inference) |
| LLM | `google/gemini-2.5-flash` (Inference) |
| TTS | `cartesia/sonic-3` + `LIVEKIT_CUSTOM_VOICE_ID` (Inference) |
| Avatar | `livekit.plugins.lemonslice` |

Override via `STT_MODEL`, `STT_LANGUAGE`, `LLM_MODEL`, `TTS_MODEL` in `agent/.env.local`.

## Project docs

See [AGENTS.md](./AGENTS.md) for LiveKit skills and verification workflow.