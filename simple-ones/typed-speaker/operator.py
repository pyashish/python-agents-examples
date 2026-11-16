"""Terminal UI for the local human-operated voice agent."""

from __future__ import annotations

import json
import os
import socket
import sys

HOST = os.getenv("OPERATOR_HOST", "127.0.0.1")
PORT = int(os.getenv("OPERATOR_PORT", "8765"))


def send_reply(connection: socket.socket, text: str) -> None:
    message = json.dumps({"type": "reply", "text": text}) + "\n"
    connection.sendall(message.encode())


def main() -> None:
    try:
        connection = socket.create_connection((HOST, PORT))
    except OSError:
        sys.exit(
            f"Could not connect to {HOST}:{PORT}. "
            "Start `lk agent console` in another terminal first."
        )

    print("Operator console connected. Waiting for the user to speak...")

    with connection, connection.makefile(encoding="utf-8") as messages:
        for line in messages:
            message = json.loads(line)
            message_type = message.get("type")
            text = str(message.get("text", ""))

            if message_type in {"status", "error"}:
                print(text)
                continue
            if message_type != "transcript":
                continue

            print(f"\nUser: {text}")
            reply = input("Agent> ").strip()
            if reply:
                send_reply(connection, reply)


if __name__ == "__main__":
    main()
