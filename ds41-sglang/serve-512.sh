#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -eq 0 ]]; then set -- serve; fi
exec bash "$ROOT/launch.sh" 512 "$@"
