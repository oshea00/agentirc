#!/usr/bin/env python3
"""AgentIRC - An OpenAI-powered IRC bot for agent communication."""

import os
import socket
import textwrap
import threading
import time

from openai import OpenAI

IRC_HOST = os.getenv("IRC_HOST", "127.0.0.1")
IRC_PORT = int(os.getenv("IRC_PORT", "6667"))
IRC_NICK = os.getenv("IRC_NICK", "agentbot")
IRC_CHANNEL = os.getenv("IRC_CHANNEL", "#agents")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
MAX_LINE = 400  # safe IRC message content length


def split_irc(text: str, max_len: int = MAX_LINE) -> list[str]:
    """Split text into IRC-safe line chunks."""
    lines = []
    for paragraph in text.splitlines():
        if not paragraph.strip():
            continue
        if len(paragraph) <= max_len:
            lines.append(paragraph)
        else:
            lines.extend(textwrap.wrap(paragraph, max_len))
    return lines or ["(empty response)"]


class AgentIRC:
    def __init__(self):
        self.client = OpenAI()
        self.sock: socket.socket | None = None
        self.history: list[dict] = []
        self.history_lock = threading.Lock()

    # ------------------------------------------------------------------
    # IRC plumbing
    # ------------------------------------------------------------------

    def _send(self, msg: str) -> None:
        print(f">> {msg}", flush=True)
        self.sock.sendall(f"{msg}\r\n".encode())

    def send_message(self, channel: str, text: str) -> None:
        for line in split_irc(text):
            self._send(f"PRIVMSG {channel} :{line}")
            time.sleep(0.05)  # avoid flood kick

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((IRC_HOST, IRC_PORT))
        self._send(f"NICK {IRC_NICK}")
        self._send(f"USER {IRC_NICK} 0 * :OpenAI Agent ({MODEL})")
        print(f"Connecting to {IRC_HOST}:{IRC_PORT} as {IRC_NICK} ...", flush=True)

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    def handle_line(self, line: str) -> None:
        parts = line.split(" ", 3)

        if parts[0] == "PING":
            self._send(f"PONG {parts[1]}")
            return

        if len(parts) < 2:
            return

        if parts[1] == "001":
            self._send(f"JOIN {IRC_CHANNEL}")
            print(f"Joined {IRC_CHANNEL}", flush=True)
            return

        if parts[1] == "PRIVMSG" and len(parts) == 4:
            nick = parts[0].lstrip(":").split("!")[0]
            channel = parts[2]
            message = parts[3].lstrip(":")
            self._dispatch(nick, channel, message)

    def _dispatch(self, nick: str, channel: str, message: str) -> None:
        # Ignore own messages
        if nick == IRC_NICK:
            return

        # Trigger patterns: "botnick: ..." / "botnick, ..." / "!..."
        prompt = None
        if message.lower().startswith(f"{IRC_NICK.lower()}:"):
            prompt = message[len(IRC_NICK) + 1 :].strip()
        elif message.lower().startswith(f"{IRC_NICK.lower()},"):
            prompt = message[len(IRC_NICK) + 1 :].strip()
        elif message.startswith("!"):
            cmd = message[1:].strip()
            if cmd == "reset":
                with self.history_lock:
                    self.history.clear()
                self.send_message(channel, "Conversation history cleared.")
                return
            if cmd == "model":
                self.send_message(channel, f"Using model: {MODEL}")
                return
            prompt = cmd

        if not prompt:
            return

        print(f"[{channel}] <{nick}> {message}")
        threading.Thread(
            target=self._ask, args=(nick, channel, prompt), daemon=True
        ).start()

    def _ask(self, nick: str, channel: str, prompt: str) -> None:
        with self.history_lock:
            self.history.append({"role": "user", "content": f"<{nick}> {prompt}"})
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful AI agent on an IRC channel used for "
                        "multi-agent coordination. Be concise. When responding, "
                        "do not use markdown formatting — plain text only."
                    ),
                },
                *self.history,
            ]

        try:
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=messages,
            )
            reply = response.choices[0].message.content.strip()
        except Exception as exc:
            reply = f"Error from OpenAI: {exc}"

        with self.history_lock:
            self.history.append({"role": "assistant", "content": reply})

        self.send_message(channel, reply)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        self.connect()
        buf = ""
        while True:
            try:
                data = self.sock.recv(4096).decode("utf-8", errors="replace")
                if not data:
                    print("Disconnected.")
                    break
                buf += data
                while "\r\n" in buf:
                    line, buf = buf.split("\r\n", 1)
                    print(f"<< {line}", flush=True)
                    self.handle_line(line)
            except KeyboardInterrupt:
                print("\nQuitting.")
                self._send("QUIT :bye")
                break
            except Exception as exc:
                print(f"Socket error: {exc}")
                break


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        raise SystemExit(1)
    AgentIRC().run()
