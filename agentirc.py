#!/usr/bin/env python3
"""AgentIRC - An OpenAI-powered IRC bot for agent communication."""

import argparse
import asyncio
import json
import os
import socket
import textwrap
import threading
import time

from openai import OpenAI

from mcptoolhandler import MCPToolExecutor

IRC_HOST = os.getenv("IRC_HOST", "127.0.0.1")
IRC_PORT = int(os.getenv("IRC_PORT", "6667"))
IRC_NICK = os.getenv("IRC_NICK", "agentbot")
IRC_CHANNEL = os.getenv("IRC_CHANNEL", "#agents")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
MCP_CONFIG = os.getenv("MCP_CONFIG", "mcp.json")
TOOL_TIMEOUT = int(os.getenv("TOOL_TIMEOUT", "30"))
INIT_TIMEOUT = int(os.getenv("MCP_INIT_TIMEOUT", "30"))
MAX_TOOL_ITERS = int(os.getenv("MAX_TOOL_ITERS", "10"))
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
    def __init__(self, debug: bool = False):
        self.client = OpenAI()
        self.sock: socket.socket | None = None
        self.history: list[dict] = []
        self.history_lock = threading.Lock()
        self.mcp: MCPToolExecutor | None = None
        self.tools: list[dict] = []
        self.debug = debug
        self._init_mcp()

    # ------------------------------------------------------------------
    # MCP setup
    # ------------------------------------------------------------------

    def _init_mcp(self) -> None:
        if not os.path.exists(MCP_CONFIG):
            print(f"No MCP config found at {MCP_CONFIG}, running without tools.", flush=True)
            return
        try:
            self.mcp = MCPToolExecutor(MCP_CONFIG)
            self.tools = asyncio.run(
                asyncio.wait_for(self.mcp.initialize_tools(), timeout=INIT_TIMEOUT)
            )
            print(f"Loaded {len(self.tools)} MCP tool(s): {[t['function']['name'] for t in self.tools]}", flush=True)
        except asyncio.TimeoutError:
            print(f"MCP init timed out after {INIT_TIMEOUT}s, running without tools.", flush=True)
        except Exception as exc:
            print(f"MCP init error: {exc}", flush=True)

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
            reply = self._run_with_tools(messages)
        except Exception as exc:
            reply = f"Error from OpenAI: {exc}"

        with self.history_lock:
            self.history.append({"role": "assistant", "content": reply})

        self.send_message(channel, reply)

    def _run_with_tools(self, messages: list[dict]) -> str:
        kwargs: dict = {"model": MODEL, "messages": messages}
        if self.tools:
            kwargs["tools"] = self.tools
            kwargs["tool_choice"] = "auto"

        for iteration in range(MAX_TOOL_ITERS + 1):
            if iteration == MAX_TOOL_ITERS:
                return "(max tool iterations reached)"

            response = self.client.chat.completions.create(**kwargs)
            msg = response.choices[0].message

            if not msg.tool_calls:
                return (msg.content or "").strip()

            kwargs["messages"] = list(kwargs["messages"]) + [msg]

            tool_results = []
            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                if self.debug:
                    print(f"[tool] {name} args={json.dumps(args, indent=2)}", flush=True)

                t0 = time.monotonic()
                try:
                    result = asyncio.run(
                        asyncio.wait_for(self.mcp.execute_tool(name, args), timeout=TOOL_TIMEOUT)
                    )
                except asyncio.TimeoutError:
                    result = json.dumps({"error": f"tool {name!r} timed out after {TOOL_TIMEOUT}s"})
                elapsed = time.monotonic() - t0

                if self.debug:
                    print(f"[tool] {name} result ({elapsed:.2f}s): {result}", flush=True)
                else:
                    print(f"[tool] {name} ({elapsed:.2f}s)", flush=True)

                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

            kwargs["messages"] = list(kwargs["messages"]) + tool_results

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
    parser = argparse.ArgumentParser(description="AgentIRC bot")
    parser.add_argument("--debug", action="store_true", help="Enable verbose tool-execution logging")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        raise SystemExit(1)
    AgentIRC(debug=args.debug).run()
