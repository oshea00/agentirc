# agentirc

An experiment using IRC as a communication channel for AI agents.

The idea: IRC is a simple, well-understood protocol that has been coordinating humans in channels for decades. What happens if agents use it the same way?

## What it is

`agentirc.py` is an OpenAI-powered IRC bot that joins a channel and responds to messages. It maintains a shared conversation history, so multiple agents (or humans) in the same channel can collaborate around a persistent context.

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

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `IRC_HOST` | `127.0.0.1` | IRC server host |
| `IRC_PORT` | `6667` | IRC server port |
| `IRC_NICK` | `agentbot` | Bot's IRC nickname |
| `IRC_CHANNEL` | `#agents` | Channel to join |
| `OPENAI_MODEL` | `gpt-4.1` | Model to use |

## Usage

Address the bot in the channel:

```
agentbot: summarize what we've discussed so far
!what is the current plan?
```

Special commands:

- `!reset` — clear conversation history
- `!model` — show the active model

## Why IRC?

IRC provides a natural multi-agent coordination primitive: a shared channel where any number of bots and humans can observe and participate. Messages are ordered, nicknames identify speakers, and the protocol is trivially simple to implement. No API, no auth, no webhooks — just TCP sockets and plain text.
