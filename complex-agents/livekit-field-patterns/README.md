# LiveKit Field Patterns

A small, runnable map of how two customers build phone agents on LiveKit, written in Python against `livekit-agents` 1.8.11.

- **Inbound** follows Strada: the backend answers a Twilio call, bridges it into a room with the agent dispatch attached, and the agent talks first.
- **Outbound** follows Stuut: the backend only books a dispatch, and the agent dials over a SIP trunk, listens first, and lets answering-machine detection decide what to do.

The code is organized by when it runs: **before**, **during** and **after** the call. Each function's docstring names the customer file it mirrors.

Companion explainer with call-flow diagrams: https://claude.ai/artifact/Q1yPg6iBSeqK6p3PQrCP1R

## Run it

```bash
uv sync
```

**1. Walk through a call (no LiveKit account needed).** Runs the real backend functions against a fake LiveKit API and prints every request, then shows how the agent parses each one.

```bash
uv run walkthrough.py
```

**2. Run the tests.** 17 run offline. The 2 in `tests/test_agent_llm.py` call a real model and run only when `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET` are set.

```bash
uv run pytest
```

**3. Talk to the agent in your terminal.** Copy `.env.example` to `.env.local` and fill in your LiveKit Cloud project keys (they pay for LiveKit Inference). Console mode has no dispatch metadata, so the agent uses a sample call in mock mode: no dialing, recording or webhooks, and canned tool results.

```bash
uv run worker.py download-files
```

```bash
uv run worker.py console
```

Set `CONSOLE_DIRECTION=outbound` in `.env.local` to try the outbound agent.

**4. Real calls.** `uv run worker.py dev` registers the worker with LiveKit Cloud. Then call the backend functions with a real `livekit.api.LiveKitAPI()`: `book_outbound_call` needs an outbound SIP trunk ID, and `answer_inbound_twilio_call` needs to be served as a Twilio Voice webhook. This repo has no web server; wire the functions into yours.

## Layout

```
shared/
  contract.py        What both sides import: CallConfig (the dispatch metadata), control
                     commands, handoffs, the signed call report, agent name, identities
  samples.py         Sample calls for the walkthrough, tests and console mode
backend/             Runs in the customer's server
  before_call.py     answer_inbound_twilio_call, book_outbound_call
  during_call.py     say, add_message, hand_off, end_call
  after_call.py      receive_call_report, receive_livekit_webhook, route_livekit_event
  fake_livekit.py    Records LiveKit API requests instead of sending them
agent/               Runs in the LiveKit agent server, one job per call
  main.py            AgentServer, prewarm, entrypoint, load_call_config
  inbound.py         run_inbound (Strada pattern)
  outbound.py        run_outbound, act_on_amd_verdict (Stuut pattern)
  pipeline.py        build_session: STT, LLM and TTS with fallbacks, turn handling
  phone_agent.py     PhoneAgent: every hook and tool
  controllers.py     Silence watchdog, close reason, max-duration timer
  control_channel.py Backend commands (data packets) and handoffs (room metadata)
  recording.py       Audio-only egress to S3
  after_call.py      build_call_report, deliver_report, the shutdown callback
  telephony.py       hang_up, press_keys
  state.py           CallState, stored as AgentSession.userdata
walkthrough.py       Offline tour of both calls
worker.py            Starts the agent server
```

## The call: before, during, after

| When | Function here | Strada (inbound, Node) | Stuut (outbound, Python) |
|---|---|---|---|
| Before | `backend.before_call.answer_inbound_twilio_call` | `apps/server/src/routes/webhook/livekit-inbound-call.ts`, `voice-gateway` `connectInboundTwilioCall` | No inbound |
| Before | `backend.before_call.book_outbound_call` | No outbound | `stuut-core/.../outreach/call/execution.py` `create_dispatch` |
| Before | `agent.main.prewarm` | `main.ts` `prewarm` | `livekit_service/main.py` `prewarm` |
| Before | `agent.main.load_call_config` | `parseJobMetadata`, `fallback-job-metadata.json` | `OutboundCallRequestSchema`, `MockOutboundCallJob` |
| Before | `agent.pipeline.build_session` | `pipeline/*.ts` | `builder.py` `build_agent_session_args` |
| Before | `agent.inbound.run_inbound` | `main.ts` `entry` | No inbound |
| Before | `agent.outbound.run_outbound` (dial + AMD) | No outbound | `jobs/outbound.py` `_create_sip_participant`, `_run_amd` |
| During | `PhoneAgent` tools | `tools/build-tools.ts` | `agents/call.py` |
| During | `agent.controllers` | `utils/inactivity.ts`, `telephony/end-call-phrases.ts` | `utils/reengagement.py`, `utils/ivr_pickup.py` |
| During | `backend.during_call.say` / `add_message` / `end_call` | `voice-gateway` `sendCommand`, `endCall` | Not used; the worker handles the call |
| During | `backend.during_call.hand_off` + `agent.control_channel.apply_handoff` | `client.ts` `handoff`, `agents/handoff.ts` | Not used |
| During | `agent.control_channel.attach_control_channel` | `control/commands.ts` | Not used |
| During | `agent.recording.start_recording` | `room/recording.ts` | `utils/egress.py` |
| After | `agent.after_call.make_after_call` (shutdown callback) | `sendEndOfCallReport` | `job.cleanup`, signed `call_ended` webhook |
| After | `backend.after_call.receive_call_report` | Strada server's VAPI webhook handler | API service webhook endpoint |
| After | `backend.after_call.route_livekit_event` | Not used | `async-service/.../functions/calls.py` |

## Hooks around each turn

`PhoneAgent` in `agent/phone_agent.py` overrides each of these:

| Hook | When it runs | Used here for | Customer use |
|---|---|---|---|
| `on_enter` | The agent takes over the session | Inbound greeting | Strada starts recording and greets |
| `on_user_turn_completed` | The user's turn is final, before the LLM runs | Do-not-call rule enforced in code | |
| `llm_node` | Wraps the LLM's output stream | Time to first token | Stuut strips tool-call syntax the model writes as text |
| `tts_node` | Text on its way to TTS, including `session.say()` | Pronunciation fixes | Stuut lowers volume and guards tool syntax; Strada gets the same effect with a TTS text transform (VAPI's format plan) |
| `on_exit` | The agent hands off or the session closes | Logging | |
| Shutdown callback | The room has closed | End-of-call report | Both |

## Mock mode

`CallConfig.mock` is set for console runs, simulations and the walkthrough. In mock mode the agent does not dial, record or post its report, and `lookup_invoice` returns a canned result. Both customers keep a path like this so local runs never touch real tools or phone lines.

## What is simplified

- No web server. Backend functions are plain async functions that take the LiveKit API client as a parameter.
- No Twilio signature check. Strada validates `X-Twilio-Signature` with the number's auth token before bridging.
- No transfers. Strada's server redirects the Twilio call; Stuut uses the SDK's `WarmTransferTask`. For a SIP leg, `JobContext.transfer_sip_participant` does a cold transfer.
- AMD runs with default thresholds. Stuut tunes them and adds a backstop for speech the VAD misses.
- `tts_node` rewrites text chunk by chunk, so a word split across chunks is missed. Strada buffers to punctuation first.
- The customers run other SDK versions (Strada: Node `@livekit/agents` 1.9.3, Stuut: Python `livekit-agents` 1.6.8). The signatures here were checked against Python 1.8.11.
