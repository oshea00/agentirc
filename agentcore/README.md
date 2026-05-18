# agentcore

A reusable AI agent core that can be embedded in any chat loop — IRC, Slack, Discord, CLI, or a custom interface.

It wraps the OpenAI chat-completions API with:

- **Stateful conversation history** (user + assistant turns).
- **MCP tool execution** — discovers and calls tools from any MCP server (stdio or HTTP).
- **History compaction** — LLM-generated summaries keep token usage bounded.
- **Hot-reload** — watches `mcp.json` for changes and reloads tool definitions automatically.
- **Thread safety** — internal locks make it safe to call from concurrent request handlers.

## Requirements

- Python 3.12+
- `OPENAI_API_KEY` environment variable set
- Dependencies installed (see project root `pyproject.toml`): `openai`, `mcp`

## Quick Start

```python
from agentcore import AgentCore

agent = AgentCore(
    model="gpt-4.1",
    system_prompt="You are a helpful assistant. Be concise. Plain text only.",
    mcp_config="mcp.json",   # optional; omit to run without tools
)

reply = agent.chat("What is the capital of France?")
print(reply)
```

## `AgentCore` API Reference

### Constructor

```python
AgentCore(
    *,
    model: str,
    system_prompt: str,
    mcp_config: str | None = None,
    tool_timeout: int = 30,
    init_timeout: int = 30,
    max_tool_iters: int = 10,
    debug: bool = False,
)
```

| Parameter | Description |
|---|---|
| `model` | OpenAI model ID, e.g. `"gpt-4.1"`. |
| `system_prompt` | Static system message prepended to every API call. |
| `mcp_config` | Path to an `mcp.json` file. `None` disables tool use. |
| `tool_timeout` | Seconds to wait for a single MCP tool call (default `30`). |
| `init_timeout` | Seconds to wait for MCP server init on startup/reload (default `30`). |
| `max_tool_iters` | Maximum tool-call rounds per `chat()` call (default `10`). |
| `debug` | Log full tool arguments and results to stdout (default `False`). |

### Methods

#### `chat(user_message: str) -> str`

Process a message and return the assistant's reply.  
Appends both the user message and the reply to conversation history.  
Thread-safe.

```python
reply = agent.chat("<alice> what's the weather in Paris?")
```

#### `reset() -> None`

Clear conversation history and reset the token count.

```python
agent.reset()
```

#### `compact() -> None`

Summarise the conversation into a single assistant message to reduce token usage.  
Resets `token_count` to `0`.

```python
if agent.token_count > 80_000:
    agent.compact()
```

### Properties

| Property | Type | Description |
|---|---|---|
| `token_count` | `int` | Prompt tokens used in the last `chat()` call. Resets after `reset()` or `compact()`. |
| `tool_names` | `list[str]` | Names of currently loaded MCP tools. |

## MCP Configuration

`mcp.json` follows the standard MCP server config format:

```json
{
  "mcpServers": {
    "myserver": {
      "command": "python3",
      "args": ["tools/myserver.py"],
      "env": { "MY_API_KEY": "..." }
    },
    "remote": {
      "url": "http://localhost:9000/mcp"
    }
  }
}
```

The file is watched for changes; saving it triggers an automatic reload with no restart required.

## Advanced: `MCPToolExecutor`

If you need direct access to MCP tool discovery or execution:

```python
import asyncio
from agentcore.mcp import MCPToolExecutor

executor = MCPToolExecutor("mcp.json")
tools = asyncio.run(executor.initialize_tools())   # OpenAI-format tool list
result = asyncio.run(executor.execute_tool("my_tool", {"arg": "value"}))
```

## Integration Examples

### Slack bot (python-slack-sdk)

```python
import os
from slack_bolt import App
from agentcore import AgentCore

app = App(token=os.environ["SLACK_BOT_TOKEN"])

agent = AgentCore(
    model="gpt-4.1",
    system_prompt="You are a helpful Slack assistant. Plain text only.",
    mcp_config="mcp.json",
)

@app.event("app_mention")
def handle_mention(event, say):
    text = event.get("text", "")
    reply = agent.chat(text)
    say(reply)

if __name__ == "__main__":
    app.start(port=3000)
```

### CLI chat loop

```python
from agentcore import AgentCore

agent = AgentCore(
    model="gpt-4.1",
    system_prompt="You are a helpful assistant.",
)

while True:
    try:
        user_input = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not user_input:
        continue
    if user_input == "/reset":
        agent.reset()
        print("History cleared.")
        continue
    print(f"Agent: {agent.chat(user_input)}")
```

## Text Utilities

`agentcore.text.split_text` splits a long string into transport-safe chunks:

```python
from agentcore.text import split_text

for chunk in split_text(long_reply, max_len=400):
    send_to_transport(chunk)
```
