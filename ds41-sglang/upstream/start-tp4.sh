#!/usr/bin/env bash
# start-tp4.sh — DeepSeek-V4.1-Flash on 4× DGX Spark (TP4/EP4).
#
# Same engine, image and commands as start.sh, with a profile of its own:
#   .env.tp4      settings for this profile (copied from .env.tp4.example on first run)
#   state-tp4/    launch record, smoke result, api key, DSpark tables
#   logs-tp4/     engine log
# The 3-Spark profile (.env, state/, logs/) is untouched, so one checkout can drive
# either fleet. Every TP4-specific runtime choice (context, KV pool, concurrency,
# prefill chunk, memory fraction, no head padding) lives in .env.tp4.example.
#
# Usage: ./start-tp4.sh doctor | build | share | pack | serve | stop | status | logs | smoke
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ENV_FILE="$ROOT/.env.tp4"
export ENV_EXAMPLE="$ROOT/.env.tp4.example"
export STATE_DIR="${STATE_DIR:-$ROOT/state-tp4}"
export LOG_DIR="${LOG_DIR:-$ROOT/logs-tp4}"
export SERVE_LOG="${SERVE_LOG:-$LOG_DIR/dsv41.log}"
exec "$ROOT/start.sh" "$@"
