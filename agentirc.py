#!/usr/bin/env python3
"""AgentIRC - IRC transport layer for AgentCore.

Connects to an IRC server, joins a channel, and delegates all AI logic to
:class:`agentcore.AgentCore`.  Responds to messages that start with the bot
nick (``botnick: ...`` / ``botnick, ...``) or the ``!`` prefix.

Special commands:
  !reset  — clear conversation history
  !model  — report the active model name
"""

import argparse
import os
import socket
import threading
import time

from agentcore import AgentCore
from agentcore.text import split_text

IRC_HOST = os.getenv("IRC_HOST", "127.0.0.1")
IRC_PORT = int(os.getenv("IRC_PORT", "6667"))
IRC_NICK = os.getenv("IRC_NICK", "agentbot")
IRC_CHANNEL = os.getenv("IRC_CHANNEL", "#agents")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
MCP_CONFIG = os.getenv("MCP_CONFIG", "mcp.json")
TOOL_TIMEOUT = int(os.getenv("TOOL_TIMEOUT", "30"))
INIT_TIMEOUT = int(os.getenv("MCP_INIT_TIMEOUT", "30"))
MAX_TOOL_ITERS = int(os.getenv("MAX_TOOL_ITERS", "10"))
CONTEXT_LIMIT = int(os.getenv("CONTEXT_LIMIT", "100000"))  # tokens; 0 = disabled
COMPACT_PERCENT = int(os.getenv("COMPACT_PERCENT", "80"))   # 0-100
MAX_LINE = 400  # safe IRC message content length

_SYSTEM_PROMPT = (
    "You are a helpful AI agent on an IRC channel used for "
    "multi-agent coordination. Be concise. When responding, "
    "do not use markdown formatting — plain text only."
)


class AgentIRC:
    """IRC transport that wraps :class:`AgentCore` and speaks the IRC protocol.

    Each incoming message that matches the trigger pattern is dispatched to a
    background thread which calls :meth:`AgentCore.chat` and sends the reply
    back to the originating channel.  Per-channel locks serialise concurrent
    requests so the shared history stays coherent.

    Args:
        debug: Pass ``True`` to enable verbose tool-execution logging.
    """

    def __init__(self, debug: bool = False) -> None:
        self.agent = AgentCore(
            model=MODEL,
            system_prompt=_SYSTEM_PROMPT,
            mcp_config=MCP_CONFIG,
            tool_timeout=TOOL_TIMEOUT,
            init_timeout=INIT_TIMEOUT,
            max_tool_iters=MAX_TOOL_ITERS,
            debug=debug,
        )
        self.sock: socket.socket | None = None
        self._send_lock = threading.RLock()
        self._channel_locks: dict[str, threading.Lock] = {}
        self._channel_locks_lock = threading.Lock()

    # ------------------------------------------------------------------
    # IRC plumbing
    # ------------------------------------------------------------------

    def _send(self, msg: str) -> None:
        print(f">> {msg}", flush=True)
        with self._send_lock:
            self.sock.sendall(f"{msg}\r\n".encode())

    def send_message(self, channel: str, text: str) -> None:
        """Send *text* to *channel*, splitting it into IRC-safe lines."""
        with self._send_lock:
            for line in split_text(text, MAX_LINE):
                self._send(f"PRIVMSG {channel} :{line}")
                time.sleep(0.05)  # avoid flood kick

    def _get_channel_lock(self, channel: str) -> threading.Lock:
        with self._channel_locks_lock:
            if channel not in self._channel_locks:
                self._channel_locks[channel] = threading.Lock()
            return self._channel_locks[channel]

    def connect(self) -> None:
        """Open the IRC socket and register the bot nick."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((IRC_HOST, IRC_PORT))
        self._send(f"NICK {IRC_NICK}")
        self._send(f"USER {IRC_NICK} 0 * :OpenAI Agent ({MODEL})")
        print(f"Connecting to {IRC_HOST}:{IRC_PORT} as {IRC_NICK} ...", flush=True)

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    def handle_line(self, line: str) -> None:
        """Parse one raw IRC line and dispatch the appropriate action."""
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
        """Identify trigger patterns and start a response thread."""
        if nick == IRC_NICK:
            return

        prompt = None
        if message.lower().startswith(f"{IRC_NICK.lower()}:"):
            prompt = message[len(IRC_NICK) + 1:].strip()
        elif message.lower().startswith(f"{IRC_NICK.lower()},"):
            prompt = message[len(IRC_NICK) + 1:].strip()
        elif message.startswith("!"):
            cmd = message[1:].strip()
            if cmd == "reset":
                self.agent.reset()
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
        """Run agent.chat() and send the reply; compact history if needed."""
        ch_lock = self._get_channel_lock(channel)
        with ch_lock:
            if CONTEXT_LIMIT > 0:
                prev = self.agent.token_count
                if prev >= CONTEXT_LIMIT * COMPACT_PERCENT / 100:
                    pct = int(prev / CONTEXT_LIMIT * 100)
                    self.send_message(
                        channel,
                        f"[Context at {pct}% ({prev}/{CONTEXT_LIMIT} tokens) — compacting...]",
                    )
                    self.agent.compact()

            try:
                reply = self.agent.chat(f"<{nick}> {prompt}")
            except Exception as exc:
                reply = f"Error from OpenAI: {exc}"

        self.send_message(channel, reply)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Connect to the IRC server and read lines until disconnected."""
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
    parser = argparse.ArgumentParser(description="AgentIRC bot")
    parser.add_argument(
        "--debug", action="store_true", help="Enable verbose tool-execution logging"
    )
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        raise SystemExit(1)

    AgentIRC(debug=args.debug).run()
