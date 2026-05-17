"""
MCP Tool Handler

Manages MCP server connections, tool discovery, and tool execution.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client


class MCPToolExecutor:
    """Manages MCP server connections, tool discovery, and tool execution."""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self.tool_to_server: Dict[str, str] = {}

    def _load_config(self) -> Dict[str, Any]:
        config_file = Path(self.config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        with open(config_file, "r") as f:
            return json.load(f)

    async def _get_tools_from_server(
        self, server_name: str, server_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        async def _list_tools(session: ClientSession) -> Dict[str, Any]:
            await session.initialize()
            tools_list = await session.list_tools()
            tools = []
            for tool in tools_list.tools:
                tool_info = {"name": tool.name, "description": tool.description}
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

        except Exception as e:
            return {"server": server_name, "error": str(e)}

    async def initialize_tools(self) -> List[Dict[str, Any]]:
        """
        Discover tools from all configured MCP servers and return them in OpenAI format.
        """
        mcp_servers = self.config.get("mcpServers", {})
        if not mcp_servers:
            return []

        tasks = [
            self._get_tools_from_server(name, cfg)
            for name, cfg in mcp_servers.items()
        ]
        server_tools = await asyncio.gather(*tasks)

        openai_tools = []
        for server_info in server_tools:
            if "error" in server_info:
                print(
                    f"Warning: Error from server {server_info.get('server', 'unknown')}: {server_info['error']}",
                    file=sys.stderr,
                )
                continue

            server_name = server_info["server"]
            for tool in server_info.get("tools", []):
                tool_name = tool["name"]
                self.tool_to_server[tool_name] = server_name
                openai_tool: Dict[str, Any] = {
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

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Execute a tool on the appropriate MCP server.
        """
        server_name = self.tool_to_server.get(tool_name)
        if not server_name:
            return json.dumps({"error": f"Tool {tool_name} not found"})

        server_config = self.config.get("mcpServers", {}).get(server_name)
        if not server_config:
            return json.dumps({"error": f"Server {server_name} not configured"})

        async def _call_tool(session: ClientSession) -> str:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments=arguments)
            if hasattr(result, "content"):
                parts = []
                for item in result.content:
                    parts.append(item.text if hasattr(item, "text") else str(item))
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

        except Exception as e:
            return json.dumps({"error": str(e)})


async def main():
    """List tools from all configured MCP servers (OpenAI format)."""
    if len(sys.argv) < 2:
        print("Usage: python mcptoolhandler.py <path_to_mcp.json>")
        sys.exit(1)

    config_path = sys.argv[1]
    try:
        executor = MCPToolExecutor(config_path)
        tools = await executor.initialize_tools()
        print(json.dumps(tools, indent=2))
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON configuration: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
