# syntax=docker/dockerfile:1.7
# First build the pinned upstream ARM64 image with build-image.sh.
ARG UPSTREAM_IMAGE=spark-vllm-ds41:upstream-a6571d0
ARG FABRIC_IMAGE=spark-vllm-glm52-exl3:sparkring-switch-v1
FROM ${FABRIC_IMAGE} AS fabric
FROM ${UPSTREAM_IMAGE}
USER root
ARG VLLM_COMMIT=a6571d0a602ed42525a7a77cf960f39ca7f8d7a8
ARG B12X_COMMIT=00b69ac22e21413622c4ecd98f607a2c3e015161
ARG FABRIC_IMAGE_ID
COPY --from=fabric /opt/sparkring/nccl /opt/sparkring/nccl
RUN test -n "$FABRIC_IMAGE_ID" && test -r /opt/sparkring/nccl/libnccl.so.2 \
    && apt-get update && apt-get install -y --no-install-recommends \
       git build-essential pkg-config liburing-dev libibverbs-dev python3-dev \
    && rm -rf /var/lib/apt/lists/*
RUN git init /opt/b12x-src && cd /opt/b12x-src \
    && git remote add origin https://github.com/local-inference-lab/b12x.git \
    && git fetch --depth 1 origin "$B12X_COMMIT" && git checkout --detach FETCH_HEAD \
    && test "$(git rev-parse HEAD)" = "$B12X_COMMIT" \
    && python3 -m pip install --no-cache-dir .
COPY patches /opt/ds41/patches
# Patch installed packages, after all pip operations. The patch gates both files.
RUN python3 - <<'PY'
import pathlib, subprocess, importlib.util
v = pathlib.Path(importlib.util.find_spec('vllm').origin).parent.parent
b = pathlib.Path(importlib.util.find_spec('b12x').origin).parent.parent
subprocess.run(['python3','/opt/ds41/patches/apply.py',
                '--vllm-root',str(v),'--b12x-root',str(b)],check=True)
PY
COPY serve.py cluster.json preflight.py gpu_smoke.py /opt/ds41/
ENV CUTE_DSL_ARCH=sm_121a TORCH_CUDA_ARCH_LIST=12.1a \
    CUDA_HOME=/usr/local/cuda TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
LABEL local.ds41.vllm=$VLLM_COMMIT local.ds41.b12x=$B12X_COMMIT \
      local.ds41.fp4-engram=ds41-fp4-disk-v2 local.ds41.fabric-parent=$FABRIC_IMAGE_ID
ENTRYPOINT []
CMD ["python3", "/opt/ds41/serve.py", "--rank", "0"]
