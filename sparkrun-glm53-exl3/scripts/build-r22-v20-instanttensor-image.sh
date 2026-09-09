#!/usr/bin/env bash
# Use an already-built v20; no kernel or vLLM rebuild is needed.
set -euo pipefail
if (( $# != 3 )); then
  echo "usage: $0 WORKER1 WORKER2 WORKER3" >&2
  exit 2
fi
test "$(uname -m)" = aarch64
root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
context="$root_dir/sparkrun-glm53-exl3"
base=spark-vllm-glm53-exl3:r22-dflash2-sm121-v20
image=spark-vllm-glm53-exl3:r22-dflash2-sm121-v20-instanttensor-r1
gpu_smoke="${GLM53_IT_GPU_SMOKE:-1}"
[[ "$gpu_smoke" == 0 || "$gpu_smoke" == 1 ]]
base_id="$(docker image inspect --format '{{.Id}}' "$base")"
docker build --platform linux/arm64 \
  --file "$context/Dockerfile.r22-dflash2-v20-instanttensor" \
  --label "local-inference.instanttensor.parent=$base_id" --tag "$image" "$context"
test "$(docker image inspect --format '{{.Id}}' "$base")" = "$base_id"
image_id="$(docker image inspect --format '{{.Id}}' "$image")"
test "$(docker image inspect --format '{{.Os}}/{{.Architecture}}' "$image")" = linux/arm64
smoke=(docker run --rm --entrypoint python3 "$image" /opt/compose/smoke_r22_instanttensor.py)
if [[ "$gpu_smoke" == 1 ]]; then
  smoke=(docker run --rm --gpus all --entrypoint python3 "$image" /opt/compose/smoke_r22_instanttensor.py --gpu)
fi
"${smoke[@]}"
for worker in "$@"; do
  docker save "$image" | ssh "$worker" docker load >/dev/null
  test "$(ssh "$worker" docker image inspect --format '{{.Id}}' "$image")" = "$image_id"
  ssh "$worker" "${smoke[@]}"
done
if [[ "$gpu_smoke" == 0 ]]; then
  echo "Built/distributed $image ($image_id); GPU loading and serving NOT qualified."
else
  echo "Built/distributed $image ($image_id); small loader smoke passed on all nodes. Full model serving still requires qualification."
fi
