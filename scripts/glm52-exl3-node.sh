#!/usr/bin/env bash
set -euo pipefail

action="${1:?usage: glm52-exl3-node.sh <prepare|preflight|start|stop|status|verify|logs> <rank> <host-ip> <head-ip>}"
rank="${2:?missing node rank}"
host_ip="${3:?missing host IP}"
head_ip="${4:?missing head IP}"

required_env=(
  IMAGE CONTAINER_NAME MODEL_ID MODEL_REVISION MODEL_HOST_PATH MODEL_MOUNT_HOST_PATH
  MODEL_MOUNT_CONTAINER_PATH MODEL_CONTAINER_PATH CACHE_HOST_PATH LOG_HOST_PATH
  SERVED_MODEL_NAME API_PORT MASTER_PORT MAX_MODEL_LEN MAX_NUM_SEQS
  MAX_NUM_BATCHED_TOKENS GPU_MEMORY_UTILIZATION QUANTIZATION ATTENTION_BACKEND
  LINEAR_BACKEND MOE_BACKEND LOAD_FORMAT ONLINE_QUANT ONLINE_QUANT_CONFIG
  VLLM_EXL3_PREFILL_CAPACITY B12X_PCIE_DMA MTP_SPECULATIVE_TOKENS
  MTP_DRAFT_TP_SIZE MTP_MOE_BACKEND MTP_USE_LOCAL_ARGMAX_REDUCTION
  MTP_DRAFT_SAMPLE_METHOD MTP_REJECTION_SAMPLE_METHOD KV_CACHE_DTYPE
  ENABLE_CHUNKED_PREFILL ENABLE_PREFIX_CACHING ASYNC_SCHEDULING
  DISABLE_CUSTOM_ALL_REDUCE COMPILATION_CONFIG VERIFY_CUDA_GRAPHS ENFORCE_EAGER
  CONTAINER_MEMORY CONTAINER_SHM_SIZE CONTAINER_NOFILE NCCL_IB_GID_INDEX
  NCCL_IB_ADDR_RANGE NCCL_DEBUG PULL_IMAGE ALLOW_UNVERIFIED_MODEL
  INSTANTTENSOR_BACKEND INSTANTTENSOR_COPY INSTANTTENSOR_BUFFER_SIZE
  INSTANTTENSOR_CONCURRENCY INSTANTTENSOR_IO_DEPTH INSTANTTENSOR_CHUNK_SIZE
  INSTANTTENSOR_MAX_FREE_MEM_USAGE
)
for name in "${required_env[@]}"; do
  [[ -n "${!name+x}" ]] || { echo "missing environment variable: $name" >&2; exit 2; }
done

resolve_fabric_iface() {
  if [[ -n "${FABRIC_IFACE:-}" ]]; then
    printf '%s\n' "$FABRIC_IFACE"
    return
  fi
  local iface
  iface="$(ip -o -4 addr show | awk -v wanted="$host_ip" '$4 ~ ("^" wanted "/") { print $2; exit }')"
  [[ -n "$iface" ]] || { echo "no Linux interface owns fabric address $host_ip" >&2; return 1; }
  printf '%s\n' "$iface"
}

resolve_hca() {
  local iface="$1"
  if [[ -n "${NCCL_IB_HCA:-}" ]]; then
    printf '%s\n' "$NCCL_IB_HCA"
    return
  fi
  local hca=""
  if command -v ibdev2netdev >/dev/null 2>&1; then
    hca="$(ibdev2netdev | awk -v wanted="$iface" '$NF == wanted || $(NF-1) == wanted { print $1; exit }')"
  fi
  if [[ -z "$hca" && -d "/sys/class/net/$iface/device/infiniband" ]]; then
    hca="$(find "/sys/class/net/$iface/device/infiniband" -mindepth 1 -maxdepth 1 -printf '%f\n' | head -n 1)"
  fi
  [[ -n "$hca" ]] || { echo "no RDMA HCA is associated with $iface" >&2; return 1; }
  printf '%s\n' "$hca"
}

capture_logs() {
  if docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    mkdir -p "$LOG_HOST_PATH"
    docker logs "$CONTAINER_NAME" >"$LOG_HOST_PATH/$(date -u +%Y%m%dT%H%M%SZ)-rank${rank}.log" 2>&1 || true
  fi
}

prepare_node() {
  command -v docker >/dev/null
  mkdir -p "$MODEL_HOST_PATH" "$CACHE_HOST_PATH" "$LOG_HOST_PATH"
  if [[ "$PULL_IMAGE" == "1" ]]; then
    docker pull "$IMAGE"
  else
    docker image inspect "$IMAGE" >/dev/null
  fi
  docker run --rm --network host --entrypoint bash \
    -v "$MODEL_HOST_PATH:/model" -v "$CACHE_HOST_PATH:/cache" \
    -e HF_HOME=/cache/huggingface "$IMAGE" -lc \
    'hf download "$1" --revision "$2" --local-dir /model' \
    _ "$MODEL_ID" "$MODEL_REVISION"
  test -f "$MODEL_HOST_PATH/config.json"
  test -f "$MODEL_HOST_PATH/model.safetensors.index.json"
  printf '%s\n' "$MODEL_REVISION" >"$MODEL_HOST_PATH/.spark-deployment-revision"
  echo "rank=$rank prepared model=$MODEL_ID revision=$MODEL_REVISION"
}

preflight_node() {
  command -v docker >/dev/null
  command -v ip >/dev/null
  docker info >/dev/null
  docker image inspect "$IMAGE" >/dev/null
  test -f "$MODEL_HOST_PATH/config.json"
  test -f "$MODEL_HOST_PATH/model.safetensors.index.json"
  test -d /dev/infiniband

  python3 - "$MODEL_HOST_PATH" <<'PY'
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
config = json.loads((model_dir / "config.json").read_text())
tail = config.get("hybrid_tr3_tail", {})
if tail.get("format") != "exl3-trellis":
    raise SystemExit(f"unexpected EXL3 format: {tail.get('format')!r}")
if tail.get("tp") != 4:
    raise SystemExit(f"checkpoint is not rank-sliced for TP4: {tail.get('tp')!r}")
if config.get("num_nextn_predict_layers") != 1:
    raise SystemExit("checkpoint does not expose the expected MTP78 layer")
index = json.loads((model_dir / "model.safetensors.index.json").read_text())
shards = sorted(set(index.get("weight_map", {}).values()))
if not shards:
    raise SystemExit("model index has no weight shards")
missing = [name for name in shards if not (model_dir / name).is_file()]
if missing:
    raise SystemExit(f"model is missing {len(missing)} shard(s): {missing[:5]}")
PY

  local iface hca installed_revision image_arch vllm_tree b12x_tree
  iface="$(resolve_fabric_iface)"
  hca="$(resolve_hca "$iface")"
  if [[ -f "$MODEL_HOST_PATH/.spark-deployment-revision" ]]; then
    installed_revision="$(<"$MODEL_HOST_PATH/.spark-deployment-revision")"
    [[ "$installed_revision" == "$MODEL_REVISION" ]] || {
      echo "model revision mismatch: have $installed_revision, want $MODEL_REVISION" >&2
      return 1
    }
  elif [[ "$ALLOW_UNVERIFIED_MODEL" != "1" ]]; then
    echo "model revision marker is missing; run the prepare action" >&2
    return 1
  fi

  image_arch="$(docker image inspect "$IMAGE" --format '{{.Architecture}}')"
  [[ "$image_arch" == "arm64" ]] || { echo "EXL3 image must be arm64, got $image_arch" >&2; return 1; }
  vllm_tree="$(docker image inspect "$IMAGE" --format '{{index .Config.Labels "local-inference.vllm.integration.tree"}}')"
  b12x_tree="$(docker image inspect "$IMAGE" --format '{{index .Config.Labels "local-inference.b12x.integration.tree"}}')"
  [[ "$vllm_tree" == "b0f8c85c7b96497e0148a18230f43d18854ae04a" ]] || {
    echo "unexpected vLLM integration tree: $vllm_tree" >&2; return 1;
  }
  [[ "$b12x_tree" == "cd3ce190f0f1917402cdfd5773724267cc9a63f8" ]] || {
    echo "unexpected B12X integration tree: $b12x_tree" >&2; return 1;
  }

  docker run --rm --gpus all --entrypoint python3 "$IMAGE" -c \
    'import importlib,sys,torch; cap=torch.cuda.get_device_capability(); assert cap == (12, 1), cap; import b12x,vllm; sys.path.insert(0,"/opt/exllamav3"); ext=importlib.import_module("exllamav3_ext"); assert hasattr(ext,"exl3_moe_fused_retile"); print("SM121 EXL3 imports ok")' \
    >/dev/null
  docker run --rm --entrypoint python3 "$IMAGE" -c \
    'from pathlib import Path; import vllm; root=Path(vllm.__file__).parent; assert (root / "model_executor/layers/quantization/exl3.py").is_file(); assert (root / "v1/attention/backends/mla/b12x_mla_sparse.py").is_file()' \
    >/dev/null
  docker run --rm --entrypoint sh \
    -v "$MODEL_MOUNT_HOST_PATH:$MODEL_MOUNT_CONTAINER_PATH:ro" \
    "$IMAGE" -c 'test -f "$1/config.json" && test -f "$1/model.safetensors.index.json"' \
    _ "$MODEL_CONTAINER_PATH" || {
      echo "model is not readable in the container at $MODEL_CONTAINER_PATH" >&2
      return 1
    }

  local serve_help
  serve_help="$(docker run --rm --gpus all --entrypoint vllm "$IMAGE" serve --help=all)"
  for flag in --attention-backend --moe-backend --quantization-config --decode-context-parallel-size --nnodes --node-rank; do
    grep -Fq -- "$flag" <<<"$serve_help" || {
      echo "image does not support required vLLM flag: $flag" >&2
      return 1
    }
  done
  echo "rank=$rank preflight ok host=$host_ip iface=$iface hca=$hca image_arch=$image_arch"
}

start_node() {
  preflight_node
  capture_logs
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  mkdir -p "$CACHE_HOST_PATH" "$LOG_HOST_PATH"

  local iface hca
  iface="$(resolve_fabric_iface)"
  hca="$(resolve_hca "$iface")"
  local headless=()
  [[ "$rank" == "0" ]] || headless=(--headless)

  local serve_args=(
    "$MODEL_CONTAINER_PATH"
    --served-model-name "$SERVED_MODEL_NAME"
    --host 0.0.0.0
    --port "$API_PORT"
    --trust-remote-code
    --tensor-parallel-size 4
    --decode-context-parallel-size 1
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --quantization "$QUANTIZATION"
    --attention-backend "$ATTENTION_BACKEND"
    --moe-backend "$MOE_BACKEND"
    --load-format "$LOAD_FORMAT"
    --kv-cache-dtype "$KV_CACHE_DTYPE"
    --tool-call-parser glm47
    --enable-auto-tool-choice
    --reasoning-parser glm45
    --default-chat-template-kwargs '{"reasoning_effort":"high"}'
    --enable-prompt-tokens-details
    --enable-force-include-usage
    --enable-request-id-headers
    --distributed-executor-backend mp
    --nnodes 4
    --node-rank "$rank"
    --master-addr "$head_ip"
    --master-port "$MASTER_PORT"
  )
  [[ "$LINEAR_BACKEND" == "auto" ]] || serve_args+=(--linear-backend "$LINEAR_BACKEND")
  [[ "$ONLINE_QUANT" == "none" ]] || serve_args+=(--quantization-config "$ONLINE_QUANT_CONFIG")
  [[ "$ENFORCE_EAGER" == "1" ]] && serve_args+=(--enforce-eager)
  [[ -z "$COMPILATION_CONFIG" ]] || serve_args+=(--compilation-config "$COMPILATION_CONFIG")
  [[ "$ENABLE_CHUNKED_PREFILL" == "1" ]] && serve_args+=(--enable-chunked-prefill)
  [[ "$ENABLE_PREFIX_CACHING" == "1" ]] && serve_args+=(--enable-prefix-caching)
  if [[ "$ASYNC_SCHEDULING" == "1" ]]; then
    serve_args+=(--async-scheduling)
  else
    serve_args+=(--no-async-scheduling)
  fi
  [[ "$DISABLE_CUSTOM_ALL_REDUCE" == "1" ]] && serve_args+=(--disable-custom-all-reduce)
  if (( MTP_SPECULATIVE_TOKENS > 0 )); then
    local local_argmax=false
    [[ "$MTP_USE_LOCAL_ARGMAX_REDUCTION" == "1" ]] && local_argmax=true
    serve_args+=(--speculative-config "{\"model\":\"$MODEL_CONTAINER_PATH\",\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_SPECULATIVE_TOKENS,\"draft_tensor_parallel_size\":$MTP_DRAFT_TP_SIZE,\"moe_backend\":\"$MTP_MOE_BACKEND\",\"use_local_argmax_reduction\":$local_argmax,\"draft_sample_method\":\"$MTP_DRAFT_SAMPLE_METHOD\",\"rejection_sample_method\":\"$MTP_REJECTION_SAMPLE_METHOD\"}")
  fi
  serve_args+=("${headless[@]}")

  docker run --gpus all -d \
    --name "$CONTAINER_NAME" --restart no --init \
    --network host --ipc host --shm-size "$CONTAINER_SHM_SIZE" \
    --memory "$CONTAINER_MEMORY" --memory-swap "$CONTAINER_MEMORY" \
    --ulimit memlock=-1:-1 --ulimit "nofile=$CONTAINER_NOFILE:$CONTAINER_NOFILE" \
    --cap-add IPC_LOCK --device /dev/infiniband:/dev/infiniband \
    -v "$MODEL_MOUNT_HOST_PATH:$MODEL_MOUNT_CONTAINER_PATH:ro" \
    -v "$CACHE_HOST_PATH:/cache" \
    -e "VLLM_HOST_IP=$host_ip" \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 \
    -e TORCH_USE_RTLD_GLOBAL=1 \
    -e CUDA_VISIBLE_DEVICES=0 -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
    -e TORCH_CUDA_ARCH_LIST=12.1a -e CUTE_DSL_ARCH=sm_121a \
    -e CMAKE_CUDA_ARCHITECTURES=121 -e FLASHINFER_CUDA_ARCH_LIST=12.1f \
    -e FLASHINFER_DISABLE_VERSION_CHECK=1 -e CUDA_DEVICE_MAX_CONNECTIONS=32 \
    -e VLLM_USE_AOT_COMPILE=1 -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
    -e VLLM_USE_MEGA_AOT_ARTIFACT=1 -e VLLM_MEMORY_PROFILE_INCLUDE_ATTN=1 \
    -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
    -e VLLM_USE_FLASHINFER_SAMPLER=1 -e VLLM_USE_V2_MODEL_RUNNER=1 \
    -e VLLM_USE_B12X_WO_PROJECTION=1 -e VLLM_USE_B12X_MHC=1 \
    -e VLLM_USE_B12X_FP8_GEMM=1 -e VLLM_USE_B12X_MOE=1 \
    -e VLLM_USE_B12X_SPARSE_INDEXER=1 -e VLLM_B12X_ABSORB_BMM=0 \
    -e B12X_MOE_FORCE_A8=0 -e B12X_MOE_FORCE_A16=1 \
    -e B12X_MLA_SM120_UNIFIED=1 -e "B12X_PCIE_DMA=$B12X_PCIE_DMA" \
    -e VLLM_USE_B12X_PCIE_DMA=0 -e VLLM_ENABLE_PCIE_ALLREDUCE=0 \
    -e VLLM_PCIE_DMA_FP8=0 -e B12X_PCIE_DMA_FP8=0 \
    -e VLLM_DCP_GLOBAL_TOPK=1 -e VLLM_DCP_SHARD_DRAFT=1 \
    -e VLLM_EXL3_ONLINE_TRELLIS_BITS=6 \
    -e VLLM_EXL3_EXT_PATH=/opt/exllamav3 \
    -e VLLM_EXL3_ENCODER_SOURCE=/opt/exllamav3-python/exllamav3 \
    -e VLLM_EXL3_ONLINE_CACHE_DIR=/cache/exl3-online \
    -e VLLM_EXL3_ONLINE_CACHE_MODE=readwrite \
    -e "VLLM_EXL3_PREFILL_CAPACITY=$VLLM_EXL3_PREFILL_CAPACITY" \
    -e VLLM_ENGINE_READY_TIMEOUT_S=7200 -e TORCHINDUCTOR_COMPILE_THREADS=1 \
    -e OMP_NUM_THREADS=16 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e SAFETENSORS_FAST_GPU=1 -e "INSTANTTENSOR_BACKEND=$INSTANTTENSOR_BACKEND" \
    -e "INSTANTTENSOR_COPY=$INSTANTTENSOR_COPY" \
    -e "INSTANTTENSOR_BUFFER_SIZE=$INSTANTTENSOR_BUFFER_SIZE" \
    -e "INSTANTTENSOR_CONCURRENCY=$INSTANTTENSOR_CONCURRENCY" \
    -e "INSTANTTENSOR_IO_DEPTH=$INSTANTTENSOR_IO_DEPTH" \
    -e "INSTANTTENSOR_CHUNK_SIZE=$INSTANTTENSOR_CHUNK_SIZE" \
    -e "INSTANTTENSOR_MAX_FREE_MEM_USAGE=$INSTANTTENSOR_MAX_FREE_MEM_USAGE" \
    -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e "NCCL_IB_HCA=$hca" \
    -e "NCCL_IB_GID_INDEX=$NCCL_IB_GID_INDEX" -e NCCL_IB_ROCE_VERSION_NUM=2 \
    -e NCCL_IB_ADDR_FAMILY=AF_INET -e "NCCL_IB_ADDR_RANGE=$NCCL_IB_ADDR_RANGE" \
    -e "NCCL_SOCKET_IFNAME=$iface" -e "GLOO_SOCKET_IFNAME=$iface" \
    -e "TP_SOCKET_IFNAME=$iface" -e "MN_IF_NAME=$iface" \
    -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=0 -e NCCL_IB_MERGE_NICS=0 \
    -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 \
    -e "NCCL_DEBUG=$NCCL_DEBUG" -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    -e XDG_CACHE_HOME=/cache -e VLLM_CACHE_ROOT=/cache/vllm \
    -e TRITON_CACHE_DIR=/cache/triton -e TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor \
    -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions \
    -e FLASHINFER_WORKSPACE_BASE=/cache/flashinfer \
    -e CUTE_DSL_CACHE_DIR=/cache/cute-dsl -e B12X_CUTE_COMPILE_CACHE_DIR=/cache/b12x-cute \
    "$IMAGE" "${serve_args[@]}" >/dev/null

  sleep 2
  docker ps --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME" || {
    docker logs "$CONTAINER_NAME" >&2 || true
    return 1
  }
  echo "rank=$rank launched host=$host_ip iface=$iface hca=$hca"
}

verify_node() {
  local state oom restarts fatal_lines
  state="$(docker inspect -f '{{.State.Status}}' "$CONTAINER_NAME")"
  oom="$(docker inspect -f '{{.State.OOMKilled}}' "$CONTAINER_NAME")"
  restarts="$(docker inspect -f '{{.RestartCount}}' "$CONTAINER_NAME")"
  [[ "$state" == "running" ]] || { echo "rank=$rank state=$state" >&2; return 1; }
  [[ "$oom" == "false" ]] || { echo "rank=$rank was OOM-killed" >&2; return 1; }
  [[ "$restarts" == "0" ]] || { echo "rank=$rank restart_count=$restarts" >&2; return 1; }
  fatal_lines="$(docker logs --since "${VERIFY_LOG_WINDOW:-30m}" "$CONTAINER_NAME" 2>&1 \
    | grep -Ei 'CUDA error|OutOfMemory|NCCL[^[:cntrl:]]*(error|failed)|Traceback \(most recent call last\)' || true)"
  [[ -z "$fatal_lines" ]] || { printf 'rank=%s fatal log lines:\n%s\n' "$rank" "$fatal_lines" >&2; return 1; }
  if [[ "$rank" == "0" && "$VERIFY_CUDA_GRAPHS" == "1" ]]; then
    docker logs "$CONTAINER_NAME" 2>&1 \
      | grep -Eiq 'Capturing CUDA graph|CUDA graph capture|Graph capturing finished' || {
        echo "rank=0 has no evidence that CUDA graph capture ran" >&2
        return 1
      }
  fi
  echo "rank=$rank verify ok state=$state oom=$oom restarts=$restarts"
}

case "$action" in
  prepare) prepare_node ;;
  preflight) preflight_node ;;
  start) start_node ;;
  stop)
    capture_logs
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
    echo "rank=$rank stopped"
    ;;
  status) docker ps -a --filter "name=^/${CONTAINER_NAME}$" --format 'rank='"$rank"' {{.Names}} {{.Status}}' ;;
  running)
    state="$(docker inspect -f '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null || true)"
    [[ "$state" == "running" ]] || { echo "rank=$rank container_state=${state:-missing}" >&2; exit 1; }
    ;;
  verify) verify_node ;;
  logs) docker logs --tail "${LOG_TAIL_LINES:-200}" "$CONTAINER_NAME" ;;
  *) echo "unknown node action: $action" >&2; exit 2 ;;
esac
