#!/usr/bin/env bash
# Build the small prefill overlay on the head Spark and optionally stream it
# to the worker Sparks. Run this from the repository root on the head node.
set -euo pipefail

root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
context="$root_dir/sparkrun-glm53-exl3"
image="${GLM53_PREFILL_IMAGE:-spark-vllm-glm52-exl3:sparkring-switch-prefill-v2}"
base="spark-vllm-glm52-exl3:sparkring-switch-v1"

docker image inspect "$base" >/dev/null
docker build --pull=false \
  --file "$context/Dockerfile.prefill" \
  --tag "$image" \
  "$context"

for worker in "$@"; do
  echo "Streaming $image to $worker..."
  docker save "$image" | ssh "$worker" docker load >/dev/null
done

echo "Built $image"
