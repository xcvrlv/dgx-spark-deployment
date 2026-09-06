#!/usr/bin/env bash
# Retain v10/v11 bases locally; distribute and GPU-check only the v12 result.
set -euo pipefail
if (( $# != 3 )); then
  echo "usage: $0 WORKER1 WORKER2 WORKER3" >&2
  exit 2
fi
test "$(uname -m)" = "aarch64"
root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
context="$root_dir/sparkrun-glm53-exl3"
fabric_base="spark-vllm-glm52-exl3:sparkring-switch-v1"
fabric_id="$(docker image inspect --format '{{.Id}}' "$fabric_base")"
docker build --platform linux/arm64 \
  --file "$context/Dockerfile.r22-dflash2" \
  --build-arg "SPARKRING_FABRIC_IMAGE=$fabric_base" \
  --build-arg "SPARKRING_FABRIC_IMAGE_ID=$fabric_id" \
  --tag spark-vllm-glm53-exl3:r22-dflash2-sm121-v10 "$context"
docker build --platform linux/arm64 \
  --file "$context/Dockerfile.r22-dflash2-v11" \
  --tag spark-vllm-glm53-exl3:r22-dflash2-sm121-v11 "$context"
GLM53_R22_IMAGE="${GLM53_R22_IMAGE:-spark-vllm-glm53-exl3:r22-dflash2-sm121-v12}" \
GLM53_R22_DOCKERFILE="$context/Dockerfile.r22-dflash2-v12" \
GLM53_R22_V11_SMOKE=1 GLM53_R22_V12_SMOKE=1 \
  bash "$context/scripts/build-r22-dflash2-image.sh" "$@"
