"""Core AI agent — LLM conversation loop, MCP tool execution, and history management.

This module provides :class:`AgentCore`, a self-contained, thread-safe agent
that can be dropped into any chat loop.  It handles:

* Maintaining a conversation history (``system`` + ``user``/``assistant`` turns).
* Calling the OpenAI chat-completions API with optional tool use.
* Executing MCP tools when the model requests them (up to a configurable limit).
* Compacting the history via an LLM-generated summary when context grows long.
* Hot-reloading the MCP configuration file when it changes on disk.

Minimal Slack-style usage example::

    from agentcore import AgentCore

    agent = AgentCore(
        model="gpt-4.1",
        system_prompt="You are a helpful Slack assistant. Be concise.",
        mcp_config="mcp.json",
    )

    def on_slack_message(text: str) -> str:
        return agent.chat(text)
"""

import asyncio
import json
import os
import threading
import time
from typing import Any

from openai import OpenAI

from agentcore.mcp import MCPToolExecutor

_MCP_WATCH_INTERVAL = 60  # seconds between mtime checks


class AgentCore:
    """Thread-safe AI agent for embedding in any chat loop.

    Wraps the OpenAI chat-completions API with:

    * A stateful conversation history (user + assistant turns).
    * An iterative tool-execution loop driven by MCP servers.
    * LLM-based history compaction to keep token usage bounded.
    * Background watching and hot-reload of the MCP config file.

    Args:
        model:           OpenAI model ID (e.g. ``"gpt-4.1"``).
        system_prompt:   Static system prompt prepended to every API call.
        mcp_config:      Path to an ``mcp.json`` file.  Pass ``None`` to run
                         without tools.
        tool_timeout:    Seconds to wait for a single MCP tool call before
                         returning a timeout error result.
        init_timeout:    Seconds to wait for MCP server initialisation on
                         startup or reload.
        max_tool_iters:  Maximum number of tool-call rounds per :meth:`chat`
                         call before giving up.
        debug:           When ``True``, log full tool arguments and results to
                         stdout.
    """

    def __init__(
        self,
        *,
        model: str,
        system_prompt: str,
        mcp_config: str | None = None,
        tool_timeout: int = 30,
        init_timeout: int = 30,
        max_tool_iters: int = 10,
        debug: bool = False,
    ) -> None:
        self._client = OpenAI()
        self._model = model
        self._system_prompt = system_prompt
        self._tool_timeout = tool_timeout
        self._init_timeout = init_timeout
        self._max_tool_iters = max_tool_iters
        self._debug = debug

        self._history: list[dict[str, Any]] = []
        self._history_lock = threading.Lock()

        self._mcp: MCPToolExecutor | None = None
        self._tools: list[dict[str, Any]] = []
        self._mcp_lock = threading.Lock()
        self._mcp_config = mcp_config
        self._mcp_mtime: float | None = None

        self._token_count: int = 0

        if mcp_config:
            self._init_mcp()
            threading.Thread(
                target=self._watch_mcp, daemon=True, name="mcp-watcher"
            ).start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def token_count(self) -> int:
        """Prompt token count reported by the last :meth:`chat` call.

        Useful for deciding whether to call :meth:`compact` before the next
        message.  Resets to ``0`` after :meth:`reset` or :meth:`compact`.
        """
        return self._token_count

    @property
    def tool_names(self) -> list[str]:
        """Names of the MCP tools currently loaded, in discovery order."""
        with self._mcp_lock:
            return [t["function"]["name"] for t in self._tools]

    def chat(self, user_message: str) -> str:
        """Process a user message and return the assistant's reply.

        Appends *user_message* to the conversation history, calls the OpenAI
        API (with tool execution if MCP tools are loaded), appends the
        assistant reply to history, and updates :attr:`token_count`.

        Thread-safe: concurrent calls are serialised by an internal lock so
        the shared history stays consistent.

        Args:
            user_message: The raw message text from the user (may include a
                          speaker prefix such as ``"<nick> hello"``).

        Returns:
            The assistant's reply as a plain string.
        """
        with self._history_lock:
            self._history.append({"role": "user", "content": user_message})
            messages = [
                {"role": "system", "content": self._system_prompt},
                *self._history,
            ]

        try:
            reply, prompt_tokens = self._run_with_tools(messages)
        except Exception as exc:
            reply = f"Error: {exc}"
            prompt_tokens = 0

        self._token_count = prompt_tokens

        with self._history_lock:
            self._history.append({"role": "assistant", "content": reply})

        return reply

    def reset(self) -> None:
        """Clear the conversation history and reset the token count."""
        with self._history_lock:
            self._history.clear()
        self._token_count = 0

    def compact(self) -> None:
        """Summarise the conversation history to reduce token usage.

        Calls the LLM to produce a concise summary of the current history,
        then replaces the entire history with that single summary message.
        Resets :attr:`token_count` to ``0``.

        If the LLM call fails the history is replaced with a placeholder so
        the agent can continue, and the error is printed to stdout.
        """
        with self._history_lock:
            snapshot = list(self._history)
        if not snapshot:
            return

        summary_messages = [
            {
                "role": "system",
                "content": (
                    "Summarize the following conversation concisely in plain text, "
                    "capturing all key results, decisions, and context needed to continue."
                ),
            },
            *snapshot,
            {
                "role": "user",
                "content": "Summarize the conversation above. Plain text, no markdown.",
            },
        ]
        try:
            resp = self._client.chat.completions.create(
                model=self._model, messages=summary_messages
            )
            summary = resp.choices[0].message.content.strip()
        except Exception as exc:
            print(f"Compaction error: {exc}", flush=True)
            summary = "(summary unavailable)"

        with self._history_lock:
            self._history = [
                {"role": "assistant", "content": f"[Conversation summary: {summary}]"}
            ]
        self._token_count = 0

    # ------------------------------------------------------------------
    # LLM + tool loop
    # ------------------------------------------------------------------

    def _run_with_tools(
        self, messages: list[dict[str, Any]]
    ) -> tuple[str, int]:
        """Run the OpenAI chat-completions loop with optional tool execution.

        Iterates until the model stops requesting tool calls or
        :attr:`_max_tool_iters` is reached.

        Args:
            messages: Full message list including the system prompt.

        Returns:
            A ``(reply, prompt_tokens)`` tuple.
        """
        with self._mcp_lock:
            mcp = self._mcp
            tools = list(self._tools)

        kwargs: dict[str, Any] = {"model": self._model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        prompt_tokens = 0
        for iteration in range(self._max_tool_iters + 1):
            if iteration == self._max_tool_iters:
                return "(max tool iterations reached)", prompt_tokens

            response = self._client.chat.completions.create(**kwargs)
            if response.usage:
                prompt_tokens = response.usage.prompt_tokens
            msg = response.choices[0].message

            if not msg.tool_calls:
                return (msg.content or "").strip(), prompt_tokens

            kwargs["messages"] = list(kwargs["messages"]) + [msg]

            tool_results = self._execute_tool_calls(mcp, msg.tool_calls)
            kwargs["messages"] = list(kwargs["messages"]) + tool_results

        # Unreachable, but satisfies the type checker.
        return "(max tool iterations reached)", prompt_tokens

    def _execute_tool_calls(
        self, mcp: MCPToolExecutor | None, tool_calls: list[Any]
    ) -> list[dict[str, Any]]:
        """Execute a batch of tool calls and return their result messages."""
        results: list[dict[str, Any]] = []
        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            if self._debug:
                print(f"[tool] {name} args={json.dumps(args, indent=2)}", flush=True)

            t0 = time.monotonic()
            try:
                result = asyncio.run(
                    asyncio.wait_for(
                        mcp.execute_tool(name, args), timeout=self._tool_timeout
                    )
                )
            except asyncio.TimeoutError:
                result = json.dumps(
                    {"error": f"tool {name!r} timed out after {self._tool_timeout}s"}
                )
            elapsed = time.monotonic() - t0

            if self._debug:
                print(f"[tool] {name} result ({elapsed:.2f}s): {result}", flush=True)
            else:
                print(f"[tool] {name} ({elapsed:.2f}s)", flush=True)

            results.append(
                {"role": "tool", "tool_call_id": tc.id, "content": result}
            )
        return results

    # ------------------------------------------------------------------
    # MCP lifecycle
    # ------------------------------------------------------------------

    def _init_mcp(self) -> None:
        """Load MCP config and initialise tool list on startup."""
        config = self._mcp_config
        if not config or not os.path.exists(config):
            print(
                f"No MCP config found at {config!r}, running without tools.",
                flush=True,
            )
            return
        try:
            mcp = MCPToolExecutor(config)
            tools = asyncio.run(
                asyncio.wait_for(mcp.initialize_tools(), timeout=self._init_timeout)
            )
            mtime = os.stat(config).st_mtime
            with self._mcp_lock:
                self._mcp = mcp
                self._tools = tools
                self._mcp_mtime = mtime
            print(
                f"Loaded {len(tools)} MCP tool(s): {[t['function']['name'] for t in tools]}",
                flush=True,
            )
        except asyncio.TimeoutError:
            print(
                f"MCP init timed out after {self._init_timeout}s, running without tools.",
                flush=True,
            )
        except Exception as exc:
            print(f"MCP init error: {exc}", flush=True)

    def _reload_mcp(self) -> None:
        """Reload MCP config after a detected file change."""
        config = self._mcp_config
        try:
            new_mcp = MCPToolExecutor(config)
            new_tools = asyncio.run(
                asyncio.wait_for(new_mcp.initialize_tools(), timeout=self._init_timeout)
            )
            new_mtime = os.stat(config).st_mtime
            with self._mcp_lock:
                self._mcp = new_mcp
                self._tools = new_tools
                self._mcp_mtime = new_mtime
            print(
                f"MCP config reloaded: {len(new_tools)} tool(s): "
                f"{[t['function']['name'] for t in new_tools]}",
                flush=True,
            )
        except asyncio.TimeoutError:
            print(
                f"MCP reload timed out after {self._init_timeout}s, "
                "keeping previous config.",
                flush=True,
            )
        except Exception as exc:
            print(f"MCP reload error: {exc}, keeping previous config.", flush=True)

    def _watch_mcp(self) -> None:
        """Background thread: check mcp.json mtime every 60 s and reload on change."""
        config = self._mcp_config
        while True:
            time.sleep(_MCP_WATCH_INTERVAL)
            if not config or not os.path.exists(config):
                continue
            try:
                mtime = os.stat(config).st_mtime
            except OSError:
                continue
            with self._mcp_lock:
                current = self._mcp_mtime
            if mtime != current:
                print("MCP config changed, reloading ...", flush=True)
                self._reload_mcp()
