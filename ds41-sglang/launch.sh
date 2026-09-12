#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOPK="$1"; shift
case "$TOPK" in 512|2048) ;; *) echo 'top-k must be 512 or 2048' >&2; exit 1 ;; esac
export DSV41_FORCED_INDEX_TOPK="$TOPK"
readonly DSV41_FORCED_INDEX_TOPK
export ENV_FILE="${DSV41_ENV_FILE:-$ROOT/.env.fleet}"
export ENV_EXAMPLE="$ROOT/fleet.env.example"
# One fleet, one set of container names. Profiles are alternatives, not co-tenants.
export STATE_DIR="$ROOT/upstream/state-tp4"
export LOG_DIR="$ROOT/upstream/logs-tp4"
export SERVE_LOG="$LOG_DIR/dsv41.log"
# Source start.sh in this process so the readonly policy survives .env loading.
source "$ROOT/upstream/start.sh" "$@"
