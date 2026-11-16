"""Start the agent server: `uv run worker.py console | dev | start`."""

from dotenv import load_dotenv

load_dotenv(".env.local")

from agent.main import main  # noqa: E402  (load env before LiveKit reads it)

if __name__ == "__main__":
    main()
