#!/usr/bin/env bash
set -euo pipefail

# Ollama OpenAI-compatible endpoint
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:11434/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-ollama}"   # Ollama ignores the key but the client requires one
export OPENAI_MODEL="${OPENAI_MODEL:-nemotron3:33b}"

# IRC settings (override via env or edit defaults below)
export IRC_HOST="${IRC_HOST:-127.0.0.1}"
export IRC_PORT="${IRC_PORT:-6667}"
export IRC_NICK="${IRC_NICK:-agentbot}"
export IRC_CHANNEL="${IRC_CHANNEL:-#agents}"

# Context compaction (low values for testing)
export CONTEXT_LIMIT="${CONTEXT_LIMIT:-131000}"
export COMPACT_PERCENT="${COMPACT_PERCENT:-80}"

echo "Starting AgentIRC with model: $OPENAI_MODEL via $OPENAI_BASE_URL"

exec python agentirc.py "$@"
