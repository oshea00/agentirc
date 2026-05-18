"""agentcore — reusable AI agent core for chat loop integration.

Exports the primary public API surface:

* :class:`~agentcore.agent.AgentCore` — drop-in agent for any chat loop.

Lower-level components are available via their submodules:

* :mod:`agentcore.mcp` — :class:`~agentcore.mcp.MCPToolExecutor`
* :mod:`agentcore.text` — :func:`~agentcore.text.split_text`
"""

from agentcore.agent import AgentCore

__all__ = ["AgentCore"]
