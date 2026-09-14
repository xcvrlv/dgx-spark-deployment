#!/usr/bin/env bash
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$here/versions.env"
base="${BASE_IMAGE:-$IMAGE}"
output="${base}-graphs-v1"
[[ $(uname -m) == aarch64 ]]
mkdir -p "$here/.build"
python3 "$here/check-upstream.py" --output "$here/.build/graph-upstream.json"
[[ $(docker image inspect --format '{{.Os}}/{{.Architecture}}' "$base") == linux/arm64 ]]
docker build --platform linux/arm64 --build-arg BASE_IMAGE="$base" -t "$output" -f - "$here" <<'DOCKERFILE'
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY patches/graph_requests.py /opt/ds41/patches/graph_requests.py
RUN package="$(python3 -c "from importlib.metadata import distribution; print(distribution('vllm').locate_file('vllm'))")" \
    && python3 /opt/ds41/patches/graph_requests.py "$package" \
    && python3 /opt/ds41/patches/graph_requests.py "$package" --check \
    && python3 -m py_compile "$package/v1/worker/gpu/cudagraph_utils.py"
LABEL local-inference.graph-requests="ds41-graphs-v1"
DOCKERFILE
printf 'Built %s; enable graph_request_buckets in a separate fleet config.\n' "$output"
