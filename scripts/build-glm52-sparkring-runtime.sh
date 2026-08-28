#!/usr/bin/env bash
# Build SparkRing's pinned R7 runtime and materialize image-bound Q40 overlays.
set -euo pipefail

image="${IMAGE:?IMAGE is required}"
revision="${SPARKRING_UPSTREAM_COMMIT:?SPARKRING_UPSTREAM_COMMIT is required}"
base_image="${SPARKRING_BASE_IMAGE:?SPARKRING_BASE_IMAGE is required}"
base_licenses="${SPARKRING_BASE_IMAGE_LICENSES:?set SPARKRING_BASE_IMAGE_LICENSES to the audited SPDX expression}"
build_root="${SPARKRING_BUILD_ROOT:-/var/tmp/sparkring-r7-build}"
q40_root="${Q40_HOST_PATH:-/var/tmp/sparkring-q40-exact-state-v1}"
model_revision="${MODEL_REVISION:?MODEL_REVISION is required}"
repository=https://github.com/FujitsuPolycom/sparkring.git

source_root="$build_root/source"
prepared_root="$build_root/prepared-sources"

mkdir -p "$build_root"
if [[ ! -d "$source_root/.git" ]]; then
  git clone --filter=blob:none --no-checkout "$repository" "$source_root"
fi
git -C "$source_root" fetch --depth 1 origin "$revision"
git -C "$source_root" checkout --detach "$revision"
[[ "$(git -C "$source_root" rev-parse HEAD)" == "$revision" ]]
git -C "$source_root" diff --quiet
[[ -z "$(git -C "$source_root" ls-files --others --exclude-standard)" ]]

docker pull "$base_image"
base_image_id="$(docker image inspect "$base_image" --format '{{.Id}}')"

if [[ -d "$prepared_root" ]]; then
  python3 "$source_root/runtime/exl3-r7/prepare_context.py" \
    --verify "$prepared_root"
else
  python3 "$source_root/runtime/exl3-r7/prepare_context.py" "$prepared_root"
fi

BASE_IMAGE="$base_image" \
BASE_IMAGE_ID="$base_image_id" \
BASE_IMAGE_LICENSES="$base_licenses" \
PREPARED_SOURCES="$prepared_root" \
IMAGE="$image" \
  "$source_root/runtime/exl3-r7/build-image.sh"

image_id="$(docker image inspect "$image" --format '{{.Id}}')"
overlay_work="$(mktemp -d "$build_root/q40-work.XXXXXX")"
trap 'rm -rf -- "$overlay_work"' EXIT
cp -a "$prepared_root/vllm" "$overlay_work/vllm"
python3 "$source_root/scripts/glm35_q40/prepare_q40_overlay_inputs.py" \
  "$overlay_work/vllm"

if [[ -d "$q40_root" ]]; then
  backup="${q40_root}.backup.$(date -u +%Y%m%dT%H%M%SZ)"
  mv "$q40_root" "$backup"
  echo "Archived previous Q40 bundle at $backup"
fi
mkdir -p "$q40_root"
install -m 0755 "$source_root/scripts/download_exl3_r7.py" \
  "$q40_root/download_exl3_r7.py"
python3 "$source_root/scripts/glm35_q40/q40_exact_state_overlay.py" \
  --source "$overlay_work/vllm/vllm/model_executor/layers/quantization/exl3.py" \
  --output "$q40_root/exl3.py"
python3 "$source_root/scripts/glm35_q40/q40_exact_state_attestation_overlay.py" \
  --source "$overlay_work/vllm/vllm/v1/worker/gpu/model_runner.py" \
  --output "$q40_root/model_runner.py" \
  --image-id "$image_id" \
  --checkpoint-revision "$model_revision"

python3 - "$q40_root" "$image_id" "$model_revision" "$revision" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
files = {}
for name in ("download_exl3_r7.py", "exl3.py", "model_runner.py"):
    path = root / name
    files[name] = {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }
manifest = {
    "schema": "spark-deployment-sparkring-q40/v1",
    "image_id": sys.argv[2],
    "model_revision": sys.argv[3],
    "sparkring_revision": sys.argv[4],
    "files": files,
}
(root / "manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

echo "Built $image ($image_id) from SparkRing $revision"
echo "Prepared image-bound Q40 bundle at $q40_root"
