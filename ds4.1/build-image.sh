#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
test "$(uname -m)" = aarch64
vllm_commit=a6571d0a602ed42525a7a77cf960f39ca7f8d7a8
build_root="${DS41_BUILD_ROOT:-$root/build}"
upstream_image=spark-vllm-ds41:upstream-a6571d0
image=spark-vllm-ds41:mx-fp4-engram-v2
fabric_image="${FABRIC_IMAGE:-spark-vllm-glm52-exl3:sparkring-switch-v1}"
fabric_id="$(docker image inspect --format '{{.Id}}' "$fabric_image")"
mkdir -p "$build_root"
if [[ ! -d "$build_root/vllm/.git" ]]; then
  git init "$build_root/vllm"
  git -C "$build_root/vllm" remote add origin https://github.com/local-inference-lab/vllm.git
  git -C "$build_root/vllm" fetch --depth 1 origin "$vllm_commit"
  git -C "$build_root/vllm" checkout --detach FETCH_HEAD
fi
test "$(git -C "$build_root/vllm" rev-parse HEAD)" = "$vllm_commit"
test -z "$(git -C "$build_root/vllm" status --porcelain)"
git -C "$build_root/vllm" submodule update --init --recursive --depth 1
docker build --platform linux/arm64 --target vllm-openai \
  --build-arg torch_cuda_arch_list=12.1a \
  --build-arg max_jobs="${MAX_JOBS:-8}" --build-arg nvcc_threads=1 \
  --build-arg VLLM_BUILD_COMMIT="$vllm_commit" \
  -f "$build_root/vllm/docker/Dockerfile" -t "$upstream_image" "$build_root/vllm"
docker build --platform linux/arm64 --build-arg UPSTREAM_IMAGE="$upstream_image" \
  --build-arg FABRIC_IMAGE="$fabric_image" --build-arg FABRIC_IMAGE_ID="$fabric_id" \
  -f "$root/Dockerfile" -t "$image" "$root"
test "$(docker image inspect --format '{{.Os}}/{{.Architecture}}' "$image")" = linux/arm64
echo "Built $image. Run cluster.py copy-image, then preflight before starting."
