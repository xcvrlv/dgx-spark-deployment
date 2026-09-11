#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
base="${BASE_IMAGE:-spark-vllm-ds41:mx-fp4-engram-v2}"
test "$(docker image inspect --format '{{.Architecture}}' "$base")" = arm64
docker build --platform linux/arm64 --build-arg BASE_IMAGE="$base" \
  -f "$root/Dockerfile.perf" -t spark-vllm-ds41:perf-v1 "$root"
