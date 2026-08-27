#!/usr/bin/env bash
set -euo pipefail

action="${1:?usage: glm53-node.sh <prepare|preflight|start|stop|status|verify|logs> <rank> <host-ip> <head-ip>}"
rank="${2:?missing node rank}"
host_ip="${3:?missing host IP}"
head_ip="${4:?missing head IP}"

required_env=(
  IMAGE CONTAINER_NAME MODEL_ID MODEL_REVISION MODEL_HOST_PATH MODEL_MOUNT_HOST_PATH
  MODEL_MOUNT_CONTAINER_PATH MODEL_CONTAINER_PATH CACHE_HOST_PATH LOG_HOST_PATH
  SERVED_MODEL_NAME API_PORT
  MASTER_PORT MAX_MODEL_LEN MAX_NUM_SEQS GPU_MEMORY_UTILIZATION BLOCK_SIZE
  MAX_NUM_BATCHED_TOKENS QUANTIZATION LINEAR_BACKEND MOE_BACKEND
  KDA_PREFILL_BACKEND LOAD_FORMAT MTP_SPECULATIVE_TOKENS KV_CACHE_DTYPE
  ENABLE_CHUNKED_PREFILL ENABLE_PREFIX_CACHING ASYNC_SCHEDULING
  DISABLE_CUSTOM_ALL_REDUCE KERNEL_CONFIG
  ENFORCE_EAGER CONTAINER_MEMORY CONTAINER_SHM_SIZE CONTAINER_NOFILE NCCL_IB_GID_INDEX
  NCCL_IB_ADDR_RANGE NCCL_DEBUG PULL_IMAGE ALLOW_UNVERIFIED_MODEL
  INSTANTTENSOR_BACKEND INSTANTTENSOR_BUFFER_SIZE INSTANTTENSOR_CONCURRENCY
  INSTANTTENSOR_IO_DEPTH INSTANTTENSOR_CHUNK_SIZE
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
  [[ -n "$iface" ]] || {
    echo "no Linux interface owns fabric address $host_ip" >&2
    return 1
  }
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
  [[ -n "$hca" ]] || {
    echo "no RDMA HCA is associated with $iface" >&2
    return 1
  }
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

  docker run --rm --network host \
    --entrypoint bash \
    -v "$MODEL_HOST_PATH:/model" \
    -v "$CACHE_HOST_PATH:/cache" \
    -e HF_HOME=/cache/huggingface \
    "$IMAGE" -lc \
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
index = json.loads((model_dir / "model.safetensors.index.json").read_text())
shards = sorted(set(index.get("weight_map", {}).values()))
if not shards:
    raise SystemExit("model index has no weight shards")
missing = [name for name in shards if not (model_dir / name).is_file()]
if missing:
    raise SystemExit(f"model is missing {len(missing)} shard(s): {missing[:5]}")
PY

  local iface hca
  iface="$(resolve_fabric_iface)"
  hca="$(resolve_hca "$iface")"

  if [[ -f "$MODEL_HOST_PATH/.spark-deployment-revision" ]]; then
    local installed_revision
    installed_revision="$(<"$MODEL_HOST_PATH/.spark-deployment-revision")"
    [[ "$installed_revision" == "$MODEL_REVISION" ]] || {
      echo "model revision mismatch: have $installed_revision, want $MODEL_REVISION" >&2
      return 1
    }
  elif [[ "$ALLOW_UNVERIFIED_MODEL" != "1" ]]; then
    echo "model revision marker is missing; run the prepare action" >&2
    return 1
  fi

  docker run --rm --gpus all --entrypoint python3 "$IMAGE" -c \
    'import torch; cap=torch.cuda.get_device_capability(); assert cap == (12, 1), cap; print("SM%d%d" % cap)' \
    >/dev/null

  docker run --rm --entrypoint sh \
    -v "$MODEL_MOUNT_HOST_PATH:$MODEL_MOUNT_CONTAINER_PATH:ro" \
    "$IMAGE" -c 'test -f "$1/config.json" && test -f "$1/model.safetensors.index.json"' \
    _ "$MODEL_CONTAINER_PATH" || {
      echo "model is not readable in the container at $MODEL_CONTAINER_PATH" >&2
      echo "check MODEL_MOUNT_HOST_PATH, MODEL_MOUNT_CONTAINER_PATH, and MODEL_CONTAINER_PATH" >&2
      return 1
    }

  local serve_help
  serve_help="$(docker run --rm --gpus all --entrypoint vllm "$IMAGE" serve --help=all)"
  for flag in --linear-backend --moe-backend --kda-prefill-backend --kv-cache-memory-bytes; do
    grep -Fq -- "$flag" <<<"$serve_help" || {
      echo "image does not support required vLLM flag: $flag" >&2
      return 1
    }
  done
  echo "rank=$rank preflight ok host=$host_ip iface=$iface hca=$hca"
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
    --chat-template-content-format string
    --tensor-parallel-size 4
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --block-size "$BLOCK_SIZE"
    --quantization "$QUANTIZATION"
    --linear-backend "$LINEAR_BACKEND"
    --moe-backend "$MOE_BACKEND"
    --load-format "$LOAD_FORMAT"
    --kda-prefill-backend "$KDA_PREFILL_BACKEND"
    --kernel-config "$KERNEL_CONFIG"
    --tool-call-parser glm47
    --enable-auto-tool-choice
    --reasoning-parser glm45
    --distributed-executor-backend mp
    --nnodes 4
    --node-rank "$rank"
    --master-addr "$head_ip"
    --master-port "$MASTER_PORT"
  )
  [[ "$ENFORCE_EAGER" == "1" ]] && serve_args+=(--enforce-eager)
  [[ "$ENABLE_CHUNKED_PREFILL" == "1" ]] && serve_args+=(--enable-chunked-prefill)
  [[ "$ENABLE_PREFIX_CACHING" == "1" ]] && serve_args+=(--enable-prefix-caching)
  [[ "$ASYNC_SCHEDULING" == "1" ]] && serve_args+=(--async-scheduling)
  [[ "$DISABLE_CUSTOM_ALL_REDUCE" == "1" ]] && serve_args+=(--disable-custom-all-reduce)
  [[ "$KV_CACHE_DTYPE" == "auto" ]] || serve_args+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
  [[ -z "${KV_CACHE_MEMORY_BYTES:-}" ]] || serve_args+=(--kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES")
  if (( MTP_SPECULATIVE_TOKENS > 0 )); then
    serve_args+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_SPECULATIVE_TOKENS,\"draft_tensor_parallel_size\":4,\"use_local_argmax_reduction\":false,\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"standard\"}")
  fi
  serve_args+=("${headless[@]}")

  docker run --gpus all -d \
    --name "$CONTAINER_NAME" --restart no --init \
    --network host --ipc host --shm-size "$CONTAINER_SHM_SIZE" \
    --memory "$CONTAINER_MEMORY" --memory-swap "$CONTAINER_MEMORY" \
    --ulimit memlock=-1:-1 --ulimit "nofile=$CONTAINER_NOFILE:$CONTAINER_NOFILE" \
    --cap-add IPC_LOCK \
    --device /dev/infiniband:/dev/infiniband \
    -v "$MODEL_MOUNT_HOST_PATH:$MODEL_MOUNT_CONTAINER_PATH:ro" \
    -v "$CACHE_HOST_PATH:/cache" \
    -e "VLLM_HOST_IP=$host_ip" \
    -e HF_HOME=/cache/huggingface \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e VLLM_NO_USAGE_STATS=1 -e TORCH_USE_RTLD_GLOBAL=1 \
    -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
    -e VLLM_USE_AOT_COMPILE=1 -e TORCHINDUCTOR_COMPILE_THREADS=1 \
    -e OMP_NUM_THREADS=16 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1f \
    -e CUTE_DSL_ARCH=sm_121a -e CMAKE_CUDA_ARCHITECTURES=121 \
    -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
    -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 \
    -e "NCCL_IB_HCA=$hca" -e "NCCL_IB_GID_INDEX=$NCCL_IB_GID_INDEX" \
    -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET \
    -e "NCCL_IB_ADDR_RANGE=$NCCL_IB_ADDR_RANGE" \
    -e "NCCL_SOCKET_IFNAME=$iface" -e "GLOO_SOCKET_IFNAME=$iface" \
    -e "TP_SOCKET_IFNAME=$iface" -e "MN_IF_NAME=$iface" \
    -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=0 -e NCCL_IB_MERGE_NICS=0 \
    -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 \
    -e "NCCL_DEBUG=$NCCL_DEBUG" -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    -e XDG_CACHE_HOME=/cache -e TRITON_CACHE_DIR=/cache/triton \
    -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions \
    -e VLLM_CACHE_ROOT=/cache/vllm -e FLASHINFER_WORKSPACE_BASE=/cache/flashinfer \
    -e "INSTANTTENSOR_BACKEND=$INSTANTTENSOR_BACKEND" \
    -e "INSTANTTENSOR_BUFFER_SIZE=$INSTANTTENSOR_BUFFER_SIZE" \
    -e "INSTANTTENSOR_CONCURRENCY=$INSTANTTENSOR_CONCURRENCY" \
    -e "INSTANTTENSOR_IO_DEPTH=$INSTANTTENSOR_IO_DEPTH" \
    -e "INSTANTTENSOR_CHUNK_SIZE=$INSTANTTENSOR_CHUNK_SIZE" \
    -e "INSTANTTENSOR_MAX_FREE_MEM_USAGE=$INSTANTTENSOR_MAX_FREE_MEM_USAGE" \
    "$IMAGE" "${serve_args[@]}" >/dev/null

  sleep 2
  docker ps --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME" || {
    docker logs "$CONTAINER_NAME" >&2 || true
    return 1
  }
  echo "rank=$rank launched host=$host_ip iface=$iface hca=$hca"
}

verify_node() {
  local state oom restarts
  state="$(docker inspect -f '{{.State.Status}}' "$CONTAINER_NAME")"
  oom="$(docker inspect -f '{{.State.OOMKilled}}' "$CONTAINER_NAME")"
  restarts="$(docker inspect -f '{{.RestartCount}}' "$CONTAINER_NAME")"
  [[ "$state" == "running" ]] || { echo "rank=$rank state=$state" >&2; return 1; }
  [[ "$oom" == "false" ]] || { echo "rank=$rank was OOM-killed" >&2; return 1; }
  [[ "$restarts" == "0" ]] || { echo "rank=$rank restart_count=$restarts" >&2; return 1; }

  local fatal_lines
  fatal_lines="$(docker logs --since "${VERIFY_LOG_WINDOW:-30m}" "$CONTAINER_NAME" 2>&1 \
    | grep -Ei 'CUDA error|OutOfMemory|NCCL[^[:cntrl:]]*(error|failed)|Traceback \(most recent call last\)' || true)"
  [[ -z "$fatal_lines" ]] || {
    echo "rank=$rank fatal log lines:" >&2
    printf '%s\n' "$fatal_lines" >&2
    return 1
  }
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
  status)
    docker ps -a --filter "name=^/${CONTAINER_NAME}$" --format 'rank='"$rank"' {{.Names}} {{.Status}}'
    ;;
  running)
    state="$(docker inspect -f '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null || true)"
    [[ "$state" == "running" ]] || {
      echo "rank=$rank container_state=${state:-missing}" >&2
      exit 1
    }
    ;;
  verify) verify_node ;;
  logs)
    docker logs --tail "${LOG_TAIL_LINES:-200}" "$CONTAINER_NAME"
    ;;
  *) echo "unknown node action: $action" >&2; exit 2 ;;
esac
