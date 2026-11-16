# Kannada Deepavali — agent project notes

Self-contained Kannada voice + LemonSlice avatar demo under `complex-agents/kannada-deepavali`.

## Layout

- `agent/agent.py` — Python worker, `agent_name=kannada-deepavali`
- `frontend/` — Next.js + Agents UI (derived from lemonslice frontend)
- `agent/assets/avatar.jpg` — portrait for LemonSlice `agent_image`

## Before changing LiveKit code

1. **reading-livekit-docs** — verify Sarvam, LemonSlice, and Inference model strings (docs MCP: `user-livekit-docs`)
2. **building-livekit-agents** — keep replies short for voice latency

## Verify behavior

1. **debugging-livekit-agents** — `lk agent debugger` after `uv run agent.py dev`
2. Optional **testing-livekit-agents** — pytest turn tests

## Reference

- LemonSlice patterns: `complex-agents/avatars/lemonslice/`
- [LemonSlice plugin](https://docs.livekit.io/agents/models/avatar/plugins/lemonslice/)
- [Sarvam STT/TTS](https://docs.livekit.io/agents/models/stt/sarvam/)