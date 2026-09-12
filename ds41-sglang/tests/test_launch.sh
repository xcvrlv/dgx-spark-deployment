#!/usr/bin/env bash
# CPU-only: no SSH, Docker or model access. Run with Bash on Linux.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
scratch=$(mktemp -d)
trap 'rm -f "$scratch/fleet.env" "$scratch/512.txt" "$scratch/2048.txt"; rmdir "$scratch"' EXIT
cat > "$scratch/fleet.env" <<'EOF'
HEAD_IP=192.168.0.1
WORKER_IPS="192.168.0.2 192.168.0.3 192.168.0.4"
WORKER_USER=juho
DSV41_INDEX_TOPK=17
TP_SIZE=3
EP_SIZE=3
NNODES=3
OFFLOAD_MODE=ram
EOF
for k in 512 2048; do
  DSV41_ENV_FILE="$scratch/fleet.env" "$BASH" "$ROOT/launch.sh" "$k" plan > "$scratch/$k.txt"
  grep -q "index_topk=$k TP=4 EP=4 nodes=4 offload=nvme" "$scratch/$k.txt"
  test "$(grep -c "DSV41_INDEX_TOPK=$k" "$scratch/$k.txt")" -eq 4
done
echo 'Both profiles enforce their selection on all four ranks despite stale environment settings'
