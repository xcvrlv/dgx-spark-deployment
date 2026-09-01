#!/usr/bin/env bash
set -euo pipefail

# Run this on the head Spark in a separate SSH session after starting the
# recipe. strace records siginfo (including si_pid/si_uid when the kernel
# provides it), while the in-container patch records the worker's Python stacks.

container="${1:-}"
diagnostics_dir="${GLM53_HOST_DIAGNOSTICS_DIR:-$HOME/glm53-runtime-diagnostics}"
mkdir -p "$diagnostics_dir"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required" >&2
  exit 1
fi
if ! command -v strace >/dev/null 2>&1; then
  echo "strace is not installed on $(hostname); install it before the diagnostic run" >&2
  exit 1
fi

sudo -v
deadline=$((SECONDS + 900))
worker_pid=""

while (( SECONDS < deadline )); do
  if [[ -z "$container" ]]; then
    container="$(docker ps --filter 'name=sparkrun_' --format '{{.ID}}' | head -n 1)"
  fi
  if [[ -n "$container" ]]; then
    worker_pid="$({ docker top "$container" -eo pid,args 2>/dev/null || true; } \
      | awk '/VLLM::Worker_TP0_DCP0/ {print $1; exit}')"
  fi
  [[ -n "$worker_pid" ]] && break
  sleep 2
done

if [[ -z "$worker_pid" ]]; then
  echo "Timed out waiting for Worker_TP0_DCP0 in SparkRun container" >&2
  exit 1
fi

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
prefix="$diagnostics_dir/signal-trace.$(hostname).pid${worker_pid}.${stamp}"
{
  echo "host=$(hostname)"
  echo "container=$container"
  echo "worker_pid=$worker_pid"
  echo "started_utc=$stamp"
  echo "output_prefix=$prefix"
} | tee "$prefix.meta"

echo "Tracing signals until worker $worker_pid exits; keep this SSH session open."
exec sudo strace -ff -ttt -T -yy -s 256 \
  -e trace=none -e signal=all -o "$prefix" -p "$worker_pid"
