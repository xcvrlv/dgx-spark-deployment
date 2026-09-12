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
    && rm -rf /var/lib/apt/lists/*
RUN git clone --filter=blob:none --no-checkout https://github.com/local-inference-lab/b12x.git /opt/b12x \
    && git -C /opt/b12x checkout --detach "${B12X_COMMIT}" \
    && test "$(git -C /opt/b12x rev-parse HEAD)" = "${B12X_COMMIT}" \
    && python3 -c 'import torch; print("torch==" + torch.__version__)' > /tmp/torch-constraint.txt \
    && python3 -m pip install --no-cache-dir -c /tmp/torch-constraint.txt /opt/b12x pytest
COPY image-check.py model-check.py roce-check.py /opt/ds41/
RUN python3 /opt/ds41/image-check.py \
    && python3 -m pip freeze > /opt/ds41/pip-freeze.txt
LABEL org.opencontainers.image.title="DeepSeek V4.1 Flash JJ TP4 ARM64 SM121" \
      org.opencontainers.image.revision="${VLLM_COMMIT}" \
      local-inference.b12x.commit="${B12X_COMMIT}"
WORKDIR /opt/ds41
ENTRYPOINT []
CMD ["vllm", "--help"]
