#!/usr/bin/env bash
# Profile N decode steps of the running server with torch.profiler and pull the
# rank-0 trace out of the head container. Analyse it on a worker, never on the
# head (the head has ~6 GB free while serving; json.load of a trace wants more).
#   ./scripts/profile/profile_decode.sh 45 /tmp/prof            # start + wait
#   python3 scripts/profile/analyze_steps.py trace.json.gz [step]  # per-step budget
#   python3 scripts/profile/analyze_trace.py trace.json.gz         # kernel table, gaps
#   python3 scripts/profile/engram_wait.py trace.json.gz           # Engram all-reduce wait
set -euo pipefail
STEPS="${1:-45}"; DIR="${2:-/tmp/prof}"; PORT="${PORT:-8888}"; CTN="${HEAD_CTN:-dsv41-head}"
L0=$(docker logs "$CTN" 2>&1 | wc -l)
curl -s -m 20 -X POST "localhost:$PORT/start_profile" -H 'Content-Type: application/json' \
  -d "{\"output_dir\":\"$DIR\",\"num_steps\":$STEPS,\"activities\":[\"CPU\",\"GPU\"],\"with_stack\":false,\"record_shapes\":false}"; echo
echo "profiling armed: send ONE request now (e.g. 300 tokens from a worker); waiting for 'Profiling done'..."
for _ in $(seq 1 120); do docker logs "$CTN" 2>&1 | tail -n +$((L0+1)) | grep -q "Profiling done" && break; sleep 3; done
f=$(docker exec "$CTN" ls "$DIR" | head -1)
docker cp "$CTN:$DIR/$f" "./${f}" && echo "trace: ./$f (rank 0; workers keep theirs in $DIR inside dsv41-worker)"
