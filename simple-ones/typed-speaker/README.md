# Terminal-Operated Voice Agent

A local human-operated LiveKit agent. The user speaks into the computer's
microphone, Deepgram transcribes their speech, and a human operator types the
response in a second terminal. Cartesia speaks that response with the voice
configured in `PAPPA_VOICE`.

## Run locally

The agent loads LiveKit credentials and `PAPPA_VOICE` from the repository root
`.env` file.

```bash
cd simple-ones/typed-speaker
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
lk agent console
```

This starts a simulated room using the local microphone and speakers. It does
not register or deploy a worker to LiveKit Cloud.

In a second terminal, start the operator UI:

```bash
cd simple-ones/typed-speaker
source .venv/bin/activate
python operator.py
```

After each final transcript, the operator terminal prompts for a response:

```text
User: Hello, can you help me?
Agent> Yes, what do you need?
```

Press Enter and the response is spoken to the user using `PAPPA_VOICE`.

## Local versus offline

The agent process, audio room, and operator connection all run locally.
Deepgram STT and Cartesia TTS through LiveKit Inference are hosted services, so
an internet connection and the LiveKit credentials in the repository `.env`
are still required. Nothing is deployed to LiveKit Cloud.
