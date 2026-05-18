"""Compatibility shim — MCPToolExecutor has moved to agentcore.mcp.

This file is kept so that ``python mcptoolhandler.py <mcp.json>`` still works
as a standalone CLI tool for listing available MCP tools.
"""

import asyncio
import json
import sys

from agentcore.mcp import MCPToolExecutor

__all__ = ["MCPToolExecutor"]


async def main() -> None:
    """List tools from all configured MCP servers (OpenAI format)."""
    if len(sys.argv) < 2:
        print("Usage: python mcptoolhandler.py <path_to_mcp.json>")
        sys.exit(1)

    config_path = sys.argv[1]
    try:
        executor = MCPToolExecutor(config_path)
        tools = await executor.initialize_tools()
        print(json.dumps(tools, indent=2))
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"Error parsing JSON configuration: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
