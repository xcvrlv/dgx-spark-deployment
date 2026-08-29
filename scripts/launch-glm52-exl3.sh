#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

export RECIPE_FILE="${RECIPE_FILE:-$root_dir/recipes/glm-5.2-exl3-r7.env}"
export NODE_SCRIPT="${NODE_SCRIPT:-$root_dir/scripts/glm52-exl3-node.sh}"
export IMAGE_DOCKERFILE="${IMAGE_DOCKERFILE:-Dockerfile.glm52-exl3-sm121}"
export DEPLOYMENT_SLUG="${DEPLOYMENT_SLUG:-glm52-exl3}"
export DEPLOYMENT_LABEL="${DEPLOYMENT_LABEL:-GLM-5.2 EXL3 R7}"

exec bash "$root_dir/scripts/launch-glm53-flash.sh" "$@"
