#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

export RECIPE_FILE="${RECIPE_FILE:-$root_dir/recipes/glm-5.2-exl3-sparkring-switch.env}"
export NODE_SCRIPT="${NODE_SCRIPT:-$root_dir/scripts/glm52-exl3-node.sh}"
export DEPLOYMENT_SLUG="${DEPLOYMENT_SLUG:-glm52-sparkring-switch}"
export DEPLOYMENT_LABEL="${DEPLOYMENT_LABEL:-GLM-5.2 SparkRing-compatible switched fabric}"
export LOCAL_ENV_FILE="${LOCAL_ENV_FILE:-$root_dir/.env.sparkring-switch}"

exec bash "$root_dir/scripts/launch-glm53-flash.sh" "$@"
