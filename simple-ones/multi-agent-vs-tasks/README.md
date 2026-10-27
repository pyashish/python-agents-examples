# Multi-agent handoffs vs AgentTasks

One phone line, a fake dental clinic, using both patterns so you can hear the difference.

| | **Agent + handoff** | **AgentTask** |
|---|---|---|
| Who's in control afterwards | The new agent, until a tool hands control back | The agent that awaited the task, right after `complete()` |
| How it's started | A tool returns `OtherAgent()` (the LLM decides) | `await SomeTask()` in `on_enter`, `on_exit`, or a tool |
| Result | None. The conversation just continues | A typed value (`bool`, dataclass) your code uses |
| Good for | A different role, voice, model, tool set, or permissions | One bounded step: consent, verify, collect a field, a form |
| Context | Fresh prompt by default. You pick what to carry over | Optional `chat_ctx`. A `TaskGroup` summarizes back into the parent |

**Can tasks do the same job?** Yes for anything shaped like "do X, then come back": consent, ID checks, collecting a name or slot, a whole intake form. A task can even use its own voice or LLM. Use a separate **agent** when the caller should be talking to someone else for an open-ended stretch, with tools and data the first agent must not have. In this demo, Billing is the only code path that can read account data, and it runs a bigger model. It's never "done", so it has no result to return.

Rule of thumb: if you can write down the return type, it's a task. If it's a new desk, it's an agent.

## Flow

```
FrontDeskAgent (FRONT_DESK_LLM, FRONT_DESK_VOICE)
 ├─ on_enter    → await ConsentTask()                    TASK, returns bool
 ├─ tool book_appointment → await TaskGroup(             TASK GROUP, returns results
 │                 CollectNameTask, CollectTimeTask)       caller can go back and correct
 └─ tool transfer_to_billing → return BillingAgent()     HANDOFF
        BillingAgent (BILLING_LLM, BILLING_VOICE)
         ├─ on_enter → await VerifyIdentityTask()        TASK, gates account data
         ├─ get_balance / send_payment_link              code checks verified_dob
         └─ tool transfer_to_front_desk → FrontDeskAgent HANDOFF back
```

Every transition is logged with a `[flow]` prefix, and the whole call flow is printed at shutdown.

## Run

LiveKit credentials come from the repo-root `.env`. STT, TTS, voices, and LLM models are required in this directory’s `.env` (see `.env.example`).

```bash
pip install -r requirements.txt
```

Talk to it from your terminal:

```bash
lk agent console agent.py
```

Or register it for SIP / the Agents Playground (`agent_name: multi-agent-vs-tasks`):

```bash
lk agent dev agent.py
```

Text-only tests (real LLM calls, no audio):

```bash
pytest -v
```

## Script to try

1. "Yes, you can record." → the consent task returns and Front Desk picks up.
2. "I want to book a cleaning." → "For Sam." → "Actually, it's for Leo." (the TaskGroup goes back a step) → "Tuesday at 10."
3. "What do I owe?" → the voice changes (handoff to Billing).
4. Date of birth "April 12, 1990" → verified (the task returns to Billing). "What's my balance?" → $145.50.
5. "Can I book another visit?" → handoff back to Front Desk.

Try asking Front Desk for your balance: it has no tool for that, so it can only transfer you.
