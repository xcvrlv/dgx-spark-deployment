#!/usr/bin/env bash
# Fail-closed SparkRun wrapper: all nodes must have a live cache flusher before
# the GLM-5.3 executor containers are created.
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manager="$root_dir/scripts/cache-flusher-cluster.sh"
hosts=""
dry_run=0
args=("$@")

for ((index = 0; index < ${#args[@]}; index++)); do
  case "${args[$index]}" in
    --hosts|-H)
      ((index + 1 < ${#args[@]})) || {
        echo "${args[$index]} requires a comma-separated host list" >&2
        exit 2
      }
      hosts="${args[$((index + 1))]}"
      ;;
    --hosts=*) hosts="${args[$index]#--hosts=}" ;;
    --dry-run) dry_run=1 ;;
  esac
done

[[ -n "$hosts" ]] || {
  echo "usage: $0 <recipe> --hosts host1,host2,host3,host4 [sparkrun run options]" >&2
  exit 2
}

if ((dry_run == 1)); then
  echo "dry run: cache flusher not started"
  exec sparkrun run "${args[@]}"
fi

bash "$manager" start "$hosts"
if ! bash "$manager" status "$hosts"; then
  bash "$manager" stop "$hosts" || true
  echo "FATAL: not every node has an active cache flusher" >&2
  exit 1
fi

echo "All node cache flushers are active; launching GLM-5.3."
if ! sparkrun run "${args[@]}"; then
  # SparkRun may already have created ranks before a follow/log command is
  # interrupted. Keep the bounded flush window alive rather than starving a
  # partially launched load; the operator can inspect status and stop it.
  echo "SparkRun returned nonzero; cache flushers remain active for safety." >&2
  echo "Inspect the job, then stop them with: $manager stop $hosts" >&2
  exit 1
fi

echo "Cache flushers remain active for the bounded load window."
echo "After /health is ready, stop them with:"
echo "  $manager stop $hosts"
