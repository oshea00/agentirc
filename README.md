# agentirc

An experiment using IRC as a communication channel for AI agents.

The idea: IRC is a simple, well-understood protocol that has been coordinating humans in channels for decades. What happens if agents use it the same way?

## What it is

`agentirc.py` is an OpenAI-powered IRC bot that joins a channel and responds to messages. It maintains a shared conversation history, so multiple agents (or humans) in the same channel can collaborate around a persistent context.

The bot supports [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) servers for tool use. On startup it discovers available tools from all configured MCP servers and passes them to the model. When the model invokes a tool, the bot executes it and feeds the result back in a loop until a final text reply is produced.

The IRC server is provided by [ngircd](https://ngircd.barton.de/), run locally via Docker.

## Setup

**Start the IRC server:**

```bash
docker compose up -d
```

**Run the bot:**

```bash
OPENAI_API_KEY=sk-... uv run agentirc.py
```

Add `--debug` to log full tool arguments and results to stdout:

```bash
OPENAI_API_KEY=sk-... uv run agentirc.py --debug
```

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `IRC_HOST` | `127.0.0.1` | IRC server host |
| `IRC_PORT` | `6667` | IRC server port |
| `IRC_NICK` | `agentbot` | Bot's IRC nickname |
| `IRC_CHANNEL` | `#agents` | Channel to join |
| `OPENAI_API_KEY` | *(required)* | API key sent with every request |
| `OPENAI_BASE_URL` | *(OpenAI default)* | Base URL for an OpenAI-compatible API |
| `OPENAI_MODEL` | `gpt-4.1` | Model to use |
| `MCP_CONFIG` | `mcp.json` | Path to MCP server config |
| `TOOL_TIMEOUT` | `30` | Seconds before a single tool call is cancelled |
| `MCP_INIT_TIMEOUT` | `30` | Seconds allowed for tool discovery at startup |
| `MAX_TOOL_ITERS` | `10` | Maximum tool-call iterations per message |

## Using local or alternative LLM servers

Any OpenAI-compatible server works — set `OPENAI_BASE_URL` to point the bot at it and `OPENAI_MODEL` to a model the server exposes.

**Ollama** (default port 11434):

```bash
OPENAI_BASE_URL=http://localhost:11434/v1 \
OPENAI_API_KEY=ollama \
OPENAI_MODEL=llama3.2 \
uv run agentirc.py
```

**LM Studio** (default port 1234):

```bash
OPENAI_BASE_URL=http://localhost:1234/v1 \
OPENAI_API_KEY=lmstudio \
OPENAI_MODEL=local-model \
uv run agentirc.py
```

`OPENAI_API_KEY` must be set to a non-empty string even when the server does not enforce authentication; most local servers accept any value.

## MCP tool configuration

Create `mcp.json` in the working directory to configure MCP servers. The bot loads it at startup and discovers tools automatically. Both stdio (subprocess) and HTTP servers are supported.

```json
{
  "mcpServers": {
    "mytools": {
      "command": "python3",
      "args": ["tools/server.py"],
      "env": { "MY_VAR": "value" }
    },
    "remote": {
      "url": "http://localhost:9000"
    }
  }
}
```

If `mcp.json` is absent the bot runs normally without tools.

## Usage

Address the bot in the channel:

```
agentbot: what time is it in Tokyo?
!summarize what we've discussed so far
```

Special commands:

- `!reset` — clear conversation history
- `!model` — show the active model

## Why IRC?

IRC provides a natural multi-agent coordination primitive: a shared channel where any number of bots and humans can observe and participate. Messages are ordered, nicknames identify speakers, and the protocol is trivially simple to implement. No API, no auth, no webhooks — just TCP sockets and plain text.
