#!/usr/bin/env bash
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$here/versions.env"
base="${BASE_IMAGE:-${IMAGE%-programs-v1}}"
output="$IMAGE"
[[ $(uname -m) == aarch64 ]]
mkdir -p "$here/.build"
python3 "$here/check-upstream.py" --output "$here/.build/roce-dtype-upstream.json"
[[ $(docker image inspect --format '{{.Os}}/{{.Architecture}}' "$base") == linux/arm64 ]]
docker build --platform linux/arm64 --build-arg BASE_IMAGE="$base" -t "$output" -f - "$here" <<'DOCKERFILE'
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY patches/roce_dtype.py patches/roce_programs.py /opt/ds41/patches/
RUN python3 /opt/ds41/patches/roce_dtype.py /opt/b12x/b12x \
    && package="$(python3 -c "from importlib.metadata import distribution; print(distribution('b12x').locate_file('b12x'))")" \
    && python3 /opt/ds41/patches/roce_dtype.py "$package" \
    && python3 /opt/ds41/patches/roce_dtype.py "$package" --check \
    && python3 /opt/ds41/patches/roce_programs.py /opt/b12x/b12x \
    && python3 /opt/ds41/patches/roce_programs.py "$package" \
    && python3 /opt/ds41/patches/roce_programs.py "$package" --check
LABEL local-inference.roce-dtype="v1" local-inference.roce-programs="v1"
DOCKERFILE
printf 'Built %s; update the fleet config image and distribute.\n' "$output"
