#!/usr/bin/env bash
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$here/versions.env"
base="${BASE_IMAGE:-${IMAGE%-programs-v1}}"
output="$IMAGE"
# sha256 of the fixed roce-check.py; update it together with that file. The
# guard makes a stale bake fail the build instead of shipping the old check.
check_sha=a1bc548efbbe7ab9adc612bf8fbde71bc794f86b59802ec1179d13c89931764d
[[ $(uname -m) == aarch64 ]]
mkdir -p "$here/.build"
python3 "$here/check-upstream.py" --output "$here/.build/roce-dtype-upstream.json"
[[ $(docker image inspect --format '{{.Os}}/{{.Architecture}}' "$base") == linux/arm64 ]]
docker build --platform linux/arm64 --build-arg BASE_IMAGE="$base" --build-arg CHECK_SHA="$check_sha" -t "$output" -f - "$here" <<'DOCKERFILE'
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
# Stage-scoped ARG: a pre-FROM ARG is not visible to RUN.
ARG CHECK_SHA
COPY patches/roce_dtype.py patches/roce_programs.py /opt/ds41/patches/
RUN python3 /opt/ds41/patches/roce_dtype.py /opt/b12x/b12x \
    && package="$(python3 -c "from importlib.metadata import distribution; print(distribution('b12x').locate_file('b12x'))")" \
    && python3 /opt/ds41/patches/roce_dtype.py "$package" \
    && python3 /opt/ds41/patches/roce_dtype.py "$package" --check \
    && python3 /opt/ds41/patches/roce_programs.py /opt/b12x/b12x \
    && python3 /opt/ds41/patches/roce_programs.py "$package" \
    && python3 /opt/ds41/patches/roce_programs.py "$package" --check
COPY roce-check.py /opt/ds41/roce-check.py
RUN python3 -c "import hashlib,sys; digest=hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest(); assert digest==sys.argv[2], f'stale roce-check baked into image: {digest}; copy the updated roce-check.py and rebuild'" /opt/ds41/roce-check.py "$CHECK_SHA"
LABEL local-inference.roce-dtype="v1" local-inference.roce-programs="v1"
DOCKERFILE
printf 'Built %s; update the fleet config image and distribute.\n' "$output"
