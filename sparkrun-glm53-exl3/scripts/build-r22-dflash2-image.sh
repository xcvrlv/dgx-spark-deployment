#!/usr/bin/env bash
# Build on the head DGX Spark, then stream the exact image to each worker.
# Usage: ./sparkrun-glm53-exl3/scripts/build-r22-dflash2-image.sh worker1 worker2 worker3
set -euo pipefail

if (( $# != 3 )); then
  echo "usage: $0 WORKER1 WORKER2 WORKER3" >&2
  exit 2
fi
test "$(uname -m)" = "aarch64"

root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
context="$root_dir/sparkrun-glm53-exl3"
image="${GLM53_R22_IMAGE:-spark-vllm-glm53-exl3:r22-dflash2-sm121-v10}"
fabric_base="spark-vllm-glm52-exl3:sparkring-switch-v1"

docker image inspect "$fabric_base" >/dev/null
fabric_id="$(docker image inspect --format '{{.Id}}' "$fabric_base")"
docker build \
  --platform linux/arm64 \
  --file "${GLM53_R22_DOCKERFILE:-$context/Dockerfile.r22-dflash2}" \
  --build-arg "SPARKRING_FABRIC_IMAGE=$fabric_base" \
  --build-arg "SPARKRING_FABRIC_IMAGE_ID=$fabric_id" \
  --tag "$image" \
  "$context"

local_id="$(docker image inspect --format '{{.Id}}' "$image")"
local_platform="$(docker image inspect --format '{{.Os}}/{{.Architecture}}' "$image")"
test "$local_platform" = "linux/arm64"
recorded_fabric_id="$(
  docker image inspect --format '{{index .Config.Labels "sparkring.fabric.image.id"}}' \
    "$image"
)"
test "$recorded_fabric_id" = "$fabric_id"
docker run --rm --gpus all --entrypoint python3 "$image" \
  /opt/compose/smoke_r22_image.py --gpu
if [[ "${GLM53_R22_V12_SMOKE:-0}" == 1 ]]; then
  docker run --rm --gpus all --entrypoint python3 "$image" \
    /opt/compose/smoke_r22_v12.py --gpu
else
  docker run --rm --gpus all --entrypoint python3 "$image" \
    /opt/compose/smoke_r22_performance.py --gpu
fi
if [[ "${GLM53_R22_V14_SMOKE:-0}" == 1 ]]; then
  docker run --rm --gpus all --entrypoint python3 "$image" \
    /opt/compose/smoke_r22_v14.py --gpu
elif [[ "${GLM53_R22_V13_SMOKE:-0}" == 1 ]]; then
  docker run --rm --gpus all --entrypoint python3 "$image" \
    /opt/compose/smoke_r22_v13.py --gpu
elif [[ "${GLM53_R22_V11_SMOKE:-0}" == 1 ]]; then
  docker run --rm --gpus all --entrypoint python3 "$image" \
    /opt/compose/smoke_r22_v11.py --gpu
fi

for worker in "$@"; do
  echo "Streaming $image ($local_id) to $worker..."
  docker save "$image" | ssh "$worker" docker load >/dev/null
  remote_id="$(ssh "$worker" docker image inspect --format '{{.Id}}' "$image")"
  test "$remote_id" = "$local_id"
  remote_platform="$(
    ssh "$worker" docker image inspect --format '{{.Os}}/{{.Architecture}}' "$image"
  )"
  test "$remote_platform" = "linux/arm64"
  ssh "$worker" docker run --rm --gpus all --entrypoint python3 "$image" \
    /opt/compose/smoke_r22_image.py --gpu
  if [[ "${GLM53_R22_V12_SMOKE:-0}" == 1 ]]; then
    ssh "$worker" docker run --rm --gpus all --entrypoint python3 "$image" \
      /opt/compose/smoke_r22_v12.py --gpu
  else
    ssh "$worker" docker run --rm --gpus all --entrypoint python3 "$image" \
      /opt/compose/smoke_r22_performance.py --gpu
  fi
  if [[ "${GLM53_R22_V14_SMOKE:-0}" == 1 ]]; then
    ssh "$worker" docker run --rm --gpus all --entrypoint python3 "$image" \
      /opt/compose/smoke_r22_v14.py --gpu
  elif [[ "${GLM53_R22_V13_SMOKE:-0}" == 1 ]]; then
    ssh "$worker" docker run --rm --gpus all --entrypoint python3 "$image" \
      /opt/compose/smoke_r22_v13.py --gpu
  elif [[ "${GLM53_R22_V11_SMOKE:-0}" == 1 ]]; then
    ssh "$worker" docker run --rm --gpus all --entrypoint python3 "$image" \
      /opt/compose/smoke_r22_v11.py --gpu
  fi
done

echo "Built and verified $image ($local_id) from fabric runtime $fabric_id"
