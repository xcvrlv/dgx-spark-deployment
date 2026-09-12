#!/usr/bin/env bash
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$here/versions.env"
[[ $(uname -m) == aarch64 ]] || { echo 'Build natively on a Spark (ARM64).' >&2; exit 1; }
mkdir -p "$here/.build"
src="$here/.build/vllm-$VLLM_COMMIT"
if [[ ! -d "$src/.git" ]]; then
  git clone --filter=blob:none --no-checkout https://github.com/local-inference-lab/vllm.git "$src"
  git -C "$src" checkout --detach "$VLLM_COMMIT"
fi
[[ $(git -C "$src" rev-parse HEAD) == "$VLLM_COMMIT" ]]
[[ -z $(git -C "$src" status --porcelain) ]] || { echo 'Build source is dirty' >&2; exit 1; }
base="spark-vllm-ds41:jj-$VLLM_COMMIT-base"
# Use JJ's complete build/dependency pipeline, including Rust and CUDA extensions.
# The nightly path resolves one Torch/vision/audio set and shares it across stages.
docker build --platform linux/arm64 --target vllm-openai \
  --file "$src/docker/Dockerfile" \
  --build-arg PYTORCH_NIGHTLY=1 \
  --build-arg torch_cuda_arch_list=12.1a \
  --build-arg max_jobs="${MAX_JOBS:-4}" --build-arg nvcc_threads=1 \
  --build-arg VLLM_BUILD_COMMIT="$VLLM_COMMIT" \
  --tag "$base" "$src"
base_id="$(docker image inspect --format '{{.Id}}' "$base")"
docker build --platform linux/arm64 --file "$here/Dockerfile" \
  --build-arg JJ_IMAGE="$base" --build-arg B12X_COMMIT="$B12X_COMMIT" \
  --build-arg VLLM_COMMIT="$VLLM_COMMIT" \
  --label "local-inference.jj-base-id=$base_id" --tag "$IMAGE" "$here"
[[ $(docker image inspect --format '{{.Os}}/{{.Architecture}}' "$IMAGE") == linux/arm64 ]]
docker run --rm --gpus all --entrypoint python3 "$IMAGE" /opt/ds41/image-check.py --gpu
docker image inspect "$IMAGE" > "$here/.build/image-inspect.json"
docker run --rm --entrypoint python3 "$IMAGE" -m pip freeze > "$here/.build/pip-freeze.txt"
echo "Built $IMAGE; distribute with python3 $here/fleet.py share"
echo 'Four-node RoCEnante and model serving still require fleet qualification.'
