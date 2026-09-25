# syntax=docker/dockerfile:1.7
ARG JJ_IMAGE
FROM ${JJ_IMAGE}
ARG VLLM_COMMIT
ARG B12X_COMMIT
USER root
ENV CUDA_HOME=/usr/local/cuda \
    CUTE_DSL_ARCH=sm_121a \
    TORCH_CUDA_ARCH_LIST=12.1a \
    FLASHINFER_CUDA_ARCH_LIST=12.1f \
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    B12X_ROCE_CACHE_DIR=/opt/b12x-roce-cache
RUN test "$(uname -m)" = aarch64 \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential git pkg-config libibverbs-dev ibverbs-providers \
       rdma-core ibverbs-utils iproute2 liburing-dev python3-dev \
       curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --filter=blob:none --no-checkout https://github.com/local-inference-lab/b12x.git /opt/b12x \
    && git -C /opt/b12x checkout --detach "${B12X_COMMIT}" \
    && test "$(git -C /opt/b12x rev-parse HEAD)" = "${B12X_COMMIT}" \
    && python3 -c 'import torch; print("torch==" + torch.__version__)' > /tmp/torch-constraint.txt \
    && python3 -m pip install --no-cache-dir -c /tmp/torch-constraint.txt /opt/b12x pytest
COPY patches/ /opt/ds41/patches/
RUN python3 /opt/ds41/patches/roce_dtype.py /opt/b12x/b12x \
    && package="$(python3 -c "from importlib.metadata import distribution; print(distribution('b12x').locate_file('b12x'))")" \
    && python3 /opt/ds41/patches/roce_dtype.py "$package" \
    && python3 /opt/ds41/patches/roce_dtype.py "$package" --check \
    && python3 /opt/ds41/patches/roce_programs.py /opt/b12x/b12x \
    && python3 /opt/ds41/patches/roce_programs.py "$package" \
    && python3 /opt/ds41/patches/roce_programs.py "$package" --check
LABEL local-inference.roce-dtype="v1" local-inference.roce-programs="v1"

ARG PREFILL_HASHES=1
RUN if [ "$PREFILL_HASHES" = 1 ]; then \
      package="$(python3 -c "from importlib.metadata import distribution; print(distribution('vllm').locate_file('vllm'))")"; \
      python3 /opt/ds41/patches/prefill_hashes.py "$package" \
      && python3 /opt/ds41/patches/prefill_hashes.py "$package" --check; \
    else test "$PREFILL_HASHES" = 0; fi
LABEL local-inference.prefill-hash-overlay="${PREFILL_HASHES}"

# Apply the disk-Engram compatibility properties and the reduced tuning budget
# to source and installed wheels; idempotent over an already-patched base.
RUN python3 /opt/ds41/patches/engram_disk.py /opt/b12x/b12x \
    && package="$(python3 -c "from importlib.metadata import distribution; print(distribution('b12x').locate_file('b12x'))")" \
    && python3 /opt/ds41/patches/engram_disk.py "$package" \
    && python3 /opt/ds41/patches/engram_disk.py "$package" --check \
    && vllm_package="$(python3 -c "from importlib.metadata import distribution; print(distribution('vllm').locate_file('vllm'))")" \
    && python3 /opt/ds41/patches/b12x_tuning.py "$vllm_package" \
    && python3 /opt/ds41/patches/b12x_tuning.py "$vllm_package" --check
LABEL local-inference.b12x-tuning="v1" local-inference.engram-disk="v1"

# World-coordinate the RoCE collective priming in the installed wheel: the
# prepare call primes a real four-rank exchange, so the request must declare
# its collective requirement and cannot be raced per rank.
RUN vllm_package="$(python3 -c "from importlib.metadata import distribution; print(distribution('vllm').locate_file('vllm'))")" \
    && python3 /opt/ds41/patches/roce_collective.py "$vllm_package" \
    && python3 /opt/ds41/patches/roce_collective.py "$vllm_package" --check
LABEL local-inference.roce-collective="v1"

# Apply only the audited communication port to source and installed wheel.
RUN python3 /opt/ds41/patches/roce.py /opt/b12x/b12x \
    && package="$(python3 -c "from importlib.metadata import distribution; print(distribution('b12x').locate_file('b12x'))")" \
    && python3 /opt/ds41/patches/roce.py "$package" \
    && python3 /opt/ds41/patches/roce.py "$package" --check \
    && cc -O2 -std=gnu11 -I"$package/comm/roce" /opt/ds41/patches/probe_roce.c \
       -libverbs -lpthread -o /opt/ds41/patches/probe_roce \
    && /opt/ds41/patches/probe_roce
# Credit the GB10 display reserve to the KV budget and back the KV tensors with
# it. The allocator is AGPL-3.0-only, so it is fetched at a pinned commit and
# verified by SHA-256 instead of being vendored with this repository. Both patches
# are hash-guarded against the vllm pin and carry independent rollback switches.
ARG DISPLAY_KV_COMMIT=878e0eecd893fadc69ad2d58b2df0fabb0fae2ee
RUN set -eu; \
    base="https://raw.githubusercontent.com/coolbho3k/DeepSeek-v4.1-Flash-2x-DGX-Spark/${DISPLAY_KV_COMMIT}"; \
    curl -fsSL "$base/release/runtime/serving/libds41_display_kv.so" -o /opt/ds41/libds41_display_kv.so; \
    echo "bf60fcdf13126ed74363d2a7f0eff208a7318667d75cde323ba11b6b07f8556c  /opt/ds41/libds41_display_kv.so" | sha256sum -c -; \
    curl -fsSL "$base/release/runtime/sources/display_kv.c" -o /opt/ds41/display_kv.c; \
    echo "c689efeab383d33662b9b524e4d34bea5067f0232f85f0df4aceebd1b06a96a4  /opt/ds41/display_kv.c" | sha256sum -c -
RUN vllm_package="$(python3 -c "from importlib.metadata import distribution; print(distribution('vllm').locate_file('vllm'))")" \
    && cp /opt/ds41/libds41_display_kv.so /opt/ds41/patches/ds41_display_kv.py "$vllm_package/v1/worker/" \
    && python3 /opt/ds41/patches/display_kv.py "$vllm_package" \
    && python3 /opt/ds41/patches/display_kv.py "$vllm_package" --check \
    && python3 /opt/ds41/patches/display_kv_credit.py "$vllm_package" \
    && python3 /opt/ds41/patches/display_kv_credit.py "$vllm_package" --check
LABEL local-inference.display-kv="v1"

COPY image-check.py model-check.py roce-check.py display-check.py /opt/ds41/
RUN python3 /opt/ds41/image-check.py \
    && python3 -m pip freeze > /opt/ds41/pip-freeze.txt
LABEL org.opencontainers.image.title="DeepSeek V4.1 Flash JJ TP4 ARM64 SM121" \
      org.opencontainers.image.revision="${VLLM_COMMIT}" \
      local-inference.b12x.commit="${B12X_COMMIT}"
LABEL local-inference.roce-overlay="ds41-roce-v1"
WORKDIR /opt/ds41
ENTRYPOINT []
CMD ["vllm", "--help"]
