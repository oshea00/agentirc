"""MCP Tool Handler — manages MCP server connections, tool discovery, and execution.

This module provides :class:`MCPToolExecutor`, which reads an ``mcp.json``
configuration file and acts as a bridge between the OpenAI tool-calling API
and one or more MCP servers running over stdio or HTTP.

Typical usage::

    executor = MCPToolExecutor("mcp.json")
    tools = await executor.initialize_tools()   # list of OpenAI-format tool dicts
    result = await executor.execute_tool("my_tool", {"arg": "value"})
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client


class MCPToolExecutor:
    """Bridge between the OpenAI function-calling API and MCP servers.

    Reads a JSON configuration file that lists one or more MCP servers (each
    identified by name and either a ``command`` for stdio transport or a
    ``url`` for HTTP transport).  Discovers the tools exposed by every server
    and converts them to the OpenAI ``tools`` list format.  Routes tool-call
    requests back to the correct server at execution time.

    Args:
        config_path: Path to an ``mcp.json`` configuration file.

    Raises:
        FileNotFoundError: If *config_path* does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """

    def __init__(self, config_path: str) -> None:
        self.config_path = config_path
        self.config = self._load_config()
        # Maps tool name → server name so execute_tool knows where to route calls.
        self.tool_to_server: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _load_config(self) -> dict[str, Any]:
        config_file = Path(self.config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        with open(config_file) as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Tool discovery
    # ------------------------------------------------------------------

    async def _get_tools_from_server(
        self, server_name: str, server_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Connect to a single MCP server and list its tools.

        Returns a dict with keys ``server``, ``tools`` (list), and
        ``tool_count`` on success, or ``server`` and ``error`` on failure.
        """

        async def _list_tools(session: ClientSession) -> dict[str, Any]:
            await session.initialize()
            tools_list = await session.list_tools()
            tools = []
            for tool in tools_list.tools:
                tool_info: dict[str, Any] = {
                    "name": tool.name,
                    "description": tool.description,
                }
                if hasattr(tool, "inputSchema"):
                    tool_info["inputSchema"] = tool.inputSchema
                tools.append(tool_info)
            return {"server": server_name, "tools": tools, "tool_count": len(tools)}

        try:
            url = server_config.get("url")
            if url:
                async with streamablehttp_client(url) as (read, write, _):
                    async with ClientSession(read, write) as session:
                        return await _list_tools(session)

            command = server_config.get("command")
            if not command:
                return {
                    "server": server_name,
                    "error": "No command or url specified in configuration",
                }

            args = server_config.get("args", [])
            env = server_config.get("env", None)
            server_params = StdioServerParameters(command=command, args=args, env=env)
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    return await _list_tools(session)

        except Exception as exc:
            return {"server": server_name, "error": str(exc)}

    async def initialize_tools(self) -> list[dict[str, Any]]:
        """Discover tools from all configured MCP servers.

        Queries every server in the ``mcpServers`` section of the config
        file concurrently.  Servers that fail are skipped with a warning
        printed to stderr.

        Returns:
            A list of tool definitions in OpenAI function-calling format,
            ready to pass as the ``tools`` argument to
            ``client.chat.completions.create()``.
        """
        mcp_servers = self.config.get("mcpServers", {})
        if not mcp_servers:
            return []

        tasks = [
            self._get_tools_from_server(name, cfg)
            for name, cfg in mcp_servers.items()
        ]
        server_tools = await asyncio.gather(*tasks)

        openai_tools: list[dict[str, Any]] = []
        for server_info in server_tools:
            if "error" in server_info:
                print(
                    f"Warning: Error from server "
                    f"{server_info.get('server', 'unknown')}: {server_info['error']}",
                    file=sys.stderr,
                )
                continue

            server_name = server_info["server"]
            for tool in server_info.get("tools", []):
                tool_name = tool["name"]
                self.tool_to_server[tool_name] = server_name
                openai_tool: dict[str, Any] = {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": tool.get("description", ""),
                    },
                }
                if "inputSchema" in tool:
                    openai_tool["function"]["parameters"] = tool["inputSchema"]
                openai_tools.append(openai_tool)

        return openai_tools

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute a named tool on its configured MCP server.

        Routes the call to whichever server advertised *tool_name* during
        :meth:`initialize_tools`.

        Args:
            tool_name:  The name of the tool to call.
            arguments:  A dict of arguments matching the tool's input schema.

        Returns:
            The tool result as a string (JSON-encoded on error).
        """
        server_name = self.tool_to_server.get(tool_name)
        if not server_name:
            return json.dumps({"error": f"Tool {tool_name!r} not found"})

        server_config = self.config.get("mcpServers", {}).get(server_name)
        if not server_config:
            return json.dumps({"error": f"Server {server_name!r} not configured"})

        async def _call_tool(session: ClientSession) -> str:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments=arguments)
            if hasattr(result, "content"):
                parts = [
                    item.text if hasattr(item, "text") else str(item)
                    for item in result.content
                ]
                return "\n".join(parts)
            return str(result)

        try:
            url = server_config.get("url")
            if url:
                async with streamablehttp_client(url) as (read, write, _):
                    async with ClientSession(read, write) as session:
                        return await _call_tool(session)

            command = server_config.get("command")
            args = server_config.get("args", [])
            env = server_config.get("env", None)
            server_params = StdioServerParameters(command=command, args=args, env=env)
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    return await _call_tool(session)

        except Exception as exc:
            return json.dumps({"error": str(exc)})
