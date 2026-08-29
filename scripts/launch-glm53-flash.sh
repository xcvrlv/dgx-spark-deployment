#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
inventory_file="${INVENTORY_FILE:-$root_dir/sparks.env}"
recipe_file="${RECIPE_FILE:-$root_dir/recipes/glm-5.3-flash-nvfp4.env}"
profile_file="${PROFILE_FILE:-}"
node_script="${NODE_SCRIPT:-$root_dir/scripts/glm53-node.sh}"
image_dockerfile="${IMAGE_DOCKERFILE:-Dockerfile.glm53-sm121}"
deployment_slug="${DEPLOYMENT_SLUG:-glm53}"
deployment_label="${DEPLOYMENT_LABEL:-GLM-5.3 Flash}"
local_env_file="${LOCAL_ENV_FILE:-$root_dir/.env}"

[[ -f "$inventory_file" ]] || { echo "inventory not found: $inventory_file" >&2; exit 2; }
[[ -f "$recipe_file" ]] || { echo "recipe not found: $recipe_file" >&2; exit 2; }
if [[ -n "$profile_file" && "$profile_file" != /* ]]; then
  profile_file="$root_dir/$profile_file"
fi
[[ -z "$profile_file" || -f "$profile_file" ]] || {
  echo "performance profile not found: $profile_file" >&2
  exit 2
}

set -a
# shellcheck disable=SC1090
source "$inventory_file"
# shellcheck disable=SC1090
source "$recipe_file"
# A profile changes only performance-sensitive settings. Local .env values are
# sourced last so node paths and deliberate experiment overrides still win.
if [[ -n "$profile_file" ]]; then
  # shellcheck disable=SC1090
  source "$profile_file"
fi
[[ -f "$local_env_file" ]] && source "$local_env_file"
MODEL_MOUNT_HOST_PATH="${MODEL_MOUNT_HOST_PATH:-$MODEL_HOST_PATH}"
MODEL_MOUNT_CONTAINER_PATH="${MODEL_MOUNT_CONTAINER_PATH:-$MODEL_CONTAINER_PATH}"
MODEL_PROVISIONING="${MODEL_PROVISIONING:-download}"
PREPARE_IMAGE="${PREPARE_IMAGE:-1}"
IMAGE_BUILD_SCRIPT="${IMAGE_BUILD_SCRIPT:-}"
SPARKRING_UPSTREAM_COMMIT="${SPARKRING_UPSTREAM_COMMIT:-}"
RUNTIME_CONTRACT="${RUNTIME_CONTRACT:-local-inference}"
Q40_ENABLED="${Q40_ENABLED:-0}"
Q40_HOST_PATH="${Q40_HOST_PATH:-}"
Q40_EXL3_SHA256="${Q40_EXL3_SHA256:-}"
Q40_CHECKPOINT_REVISION="${Q40_CHECKPOINT_REVISION:-$MODEL_REVISION}"
MODEL_CONFIG_SHA256="${MODEL_CONFIG_SHA256:-}"
MODEL_INDEX_SHA256="${MODEL_INDEX_SHA256:-}"
MODEL_SHARD_COUNT="${MODEL_SHARD_COUNT:-}"
MODEL_INDEX_TOTAL_SIZE="${MODEL_INDEX_TOTAL_SIZE:-}"
DECODE_CONTEXT_PARALLEL_SIZE="${DECODE_CONTEXT_PARALLEL_SIZE:-1}"
DCP_COMM_BACKEND="${DCP_COMM_BACKEND:-}"
DCP_KV_CACHE_INTERLEAVE_SIZE="${DCP_KV_CACHE_INTERLEAVE_SIZE:-}"
HF_OVERRIDES="${HF_OVERRIDES:-}"
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-}"
KV_FP8_ROPE="${KV_FP8_ROPE:-0}"
VLLM_NVFP4_MLA_DYNAMIC_SCALE="${VLLM_NVFP4_MLA_DYNAMIC_SCALE:-0}"
VLLM_USE_B12X_DCP_A2A="${VLLM_USE_B12X_DCP_A2A:-0}"
VLLM_B12X_MLA_CKV_GATHER="${VLLM_B12X_MLA_CKV_GATHER:-0}"
VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS="${VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS:-0}"
VLLM_SPARK_MAX_QUERY_ROWS="${VLLM_SPARK_MAX_QUERY_ROWS:-8}"
VLLM_SPARK_MTP_MODE_ID="${VLLM_SPARK_MTP_MODE_ID:-target-only}"
VLLM_SPARK_MTP_TOKENS="${VLLM_SPARK_MTP_TOKENS:-$MTP_SPECULATIVE_TOKENS}"
VLLM_SPARK_SHARED_CAPTURE_STREAM="${VLLM_SPARK_SHARED_CAPTURE_STREAM:-0}"
UMA_DROP_CACHES_BEFORE_START="${UMA_DROP_CACHES_BEFORE_START:-0}"
VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
CPU_THREADS_PER_PROCESS="${CPU_THREADS_PER_PROCESS:-16}"
NETWORK_TOPOLOGY="${NETWORK_TOPOLOGY:-switched-single-rail}"
NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-0}"
NCCL_IB_MERGE_NICS="${NCCL_IB_MERGE_NICS:-0}"
NCCL_IB_SUBNET_AWARE_ROUTING="${NCCL_IB_SUBNET_AWARE_ROUTING:-0}"
NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-}"
NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-}"
NCCL_ALGO="${NCCL_ALGO:-}"
set +a

action="${1:-start}"
selected_rank="${2:-}"
[[ "${TP_NODE_COUNT:-}" == "4" ]] || { echo "this recipe requires TP_NODE_COUNT=4" >&2; exit 2; }
[[ "$CLUSTER_RAIL" =~ ^CX[01]$ ]] || { echo "CLUSTER_RAIL must be CX0 or CX1" >&2; exit 2; }
[[ "$BUILD_IMAGE" =~ ^[01]$ && "$PULL_IMAGE" =~ ^[01]$ && "$PREPARE_IMAGE" =~ ^[01]$ ]] || {
  echo "BUILD_IMAGE, PULL_IMAGE, and PREPARE_IMAGE must be 0 or 1" >&2
  exit 2
}
[[ "$BUILD_IMAGE" != "1" || "$PULL_IMAGE" != "1" ]] || {
  echo "BUILD_IMAGE and PULL_IMAGE cannot both be enabled" >&2
  exit 2
}
for boolean_name in ENFORCE_EAGER VERIFY_CUDA_GRAPHS MTP_USE_LOCAL_ARGMAX_REDUCTION Q40_ENABLED UMA_DROP_CACHES_BEFORE_START; do
  [[ "${!boolean_name}" =~ ^[01]$ ]] || {
    echo "$boolean_name must be 0 or 1" >&2
    exit 2
  }
done
if [[ "$Q40_ENABLED" == "1" && ! "$Q40_CHECKPOINT_REVISION" =~ ^[0-9a-f]{40}$ ]]; then
  echo "Q40_CHECKPOINT_REVISION must be 40 lowercase hex characters" >&2
  exit 2
fi
for boolean_name in NCCL_CROSS_NIC NCCL_IB_MERGE_NICS NCCL_IB_SUBNET_AWARE_ROUTING; do
  [[ "${!boolean_name}" =~ ^[01]$ ]] || {
    echo "$boolean_name must be 0 or 1" >&2
    exit 2
  }
done
[[ "$MTP_SPECULATIVE_TOKENS" =~ ^[0-9]+$ ]] || {
  echo "MTP_SPECULATIVE_TOKENS must be a non-negative integer" >&2
  exit 2
}
[[ "$VLLM_WORKER_MULTIPROC_METHOD" =~ ^(spawn|forkserver)$ ]] || {
  echo "VLLM_WORKER_MULTIPROC_METHOD must be spawn or forkserver" >&2
  exit 2
}
[[ "$CPU_THREADS_PER_PROCESS" =~ ^[1-9][0-9]*$ ]] || {
  echo "CPU_THREADS_PER_PROCESS must be a positive integer" >&2
  exit 2
}
if [[ "$MTP_USE_LOCAL_ARGMAX_REDUCTION" == "1" && "$MTP_DRAFT_SAMPLE_METHOD" != "greedy" ]]; then
  echo "local argmax reduction requires MTP_DRAFT_SAMPLE_METHOD=greedy" >&2
  exit 2
fi
if (( MTP_SPECULATIVE_TOKENS > 0 )); then
  case "$MTP_MOE_BACKEND" in
    b12x|marlin|triton|batched_triton|flashinfer_trtllm|flashinfer_cutlass|aiter) ;;
    *)
      echo "unsupported MTP_MOE_BACKEND=$MTP_MOE_BACKEND" >&2
      exit 2
      ;;
  esac
fi
if [[ "$NETWORK_TOPOLOGY" == "switch-star-dual-rail" ]]; then
  [[ -z "$NCCL_IB_HCA" || "$NCCL_IB_HCA" == *,* ]] || {
    echo "set NCCL_IB_HCA to two comma-separated devices or leave it empty for per-node discovery" >&2
    exit 2
  }
  [[ "$NCCL_CROSS_NIC" == "1" ]] || {
    echo "switch-star-dual-rail requires NCCL_CROSS_NIC=1" >&2
    exit 2
  }
  [[ -z "$NCCL_ALGO" ]] || {
    echo "the switched profile leaves NCCL_ALGO unset for topology-aware selection" >&2
    exit 2
  }
fi
if [[ "$RUNTIME_CONTRACT" == "sparkring-r7-switch-q40" ]]; then
  contract_values=(
    "DECODE_CONTEXT_PARALLEL_SIZE=4" "DCP_COMM_BACKEND=ag_rs"
    "DCP_KV_CACHE_INTERLEAVE_SIZE=1" "MTP_SPECULATIVE_TOKENS=4"
    "MTP_DRAFT_TP_SIZE=4" "MTP_MOE_BACKEND=b12x"
    "MAX_MODEL_LEN=1048576" "MAX_NUM_SEQS=16"
    "MAX_NUM_BATCHED_TOKENS=4096" "BLOCK_SIZE=64"
    "KV_CACHE_DTYPE=fp8" "KV_CACHE_MEMORY_BYTES=16489130435"
    "KV_FP8_ROPE=0" "VLLM_NVFP4_MLA_DYNAMIC_SCALE=0"
    "VLLM_WORKER_MULTIPROC_METHOD=forkserver" "CPU_THREADS_PER_PROCESS=4"
    "VLLM_EXL3_PREFILL_CAPACITY=4096" "VLLM_SPARK_MAX_QUERY_ROWS=40"
    "MAX_CUDAGRAPH_CAPTURE_SIZE=40"
  )
  for contract_value in "${contract_values[@]}"; do
    contract_name="${contract_value%%=*}"
    expected_value="${contract_value#*=}"
    [[ "${!contract_name}" == "$expected_value" ]] || {
      echo "SparkRing serving contract requires $contract_value (got ${!contract_name})" >&2
      exit 2
    }
  done
fi
[[ -z "$profile_file" ]] || echo "Using performance profile: $profile_file"
case "$action" in
  preflight|start|verify|benchmark)
    echo "Requested action config: action=$action target_moe=$MOE_BACKEND mtp_moe=$MTP_MOE_BACKEND mtp_tokens=$MTP_SPECULATIVE_TOKENS eager=$ENFORCE_EAGER"
    ;;
esac
ssh_options=(-o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=4)
[[ -n "${SPARK_SSH_KEY:-}" ]] && ssh_options+=(-i "$SPARK_SSH_KEY")

inventory_value() {
  local node="$1"
  local suffix="$2"
  local variable="SPARK_${node}_${suffix}"
  [[ -n "${!variable:-}" ]] || { echo "missing inventory value: $variable" >&2; return 1; }
  printf '%s\n' "${!variable}"
}

ssh_target() {
  local management_ip="$1"
  if [[ -n "${SPARK_SSH_USER:-}" ]]; then
    printf '%s@%s\n' "$SPARK_SSH_USER" "$management_ip"
  else
    printf '%s\n' "$management_ip"
  fi
}

target_for_rank() {
  local rank="$1"
  local node=$((rank + 1))
  local management_ip
  management_ip="$(inventory_value "$node" MANAGEMENT_IP)"
  ssh_target "$management_ip"
}

remote_node() {
  local node_action="$1"
  local rank="$2"
  local node=$((rank + 1))
  local management_ip fabric_ip fabric_ips head_ip target remote_command
  management_ip="$(inventory_value "$node" MANAGEMENT_IP)"
  fabric_ip="$(inventory_value "$node" "${CLUSTER_RAIL}_IP")"
  fabric_ips="$(inventory_value "$node" CX0_IP),$(inventory_value "$node" CX1_IP)"
  head_ip="$(inventory_value 1 "${CLUSTER_RAIL}_IP")"
  target="$(ssh_target "$management_ip")"

  local assignments=(
    "IMAGE=$IMAGE" "CONTAINER_NAME=$CONTAINER_NAME"
    "MODEL_ID=$MODEL_ID" "MODEL_REVISION=$MODEL_REVISION"
    "MODEL_PROVISIONING=$MODEL_PROVISIONING"
    "MODEL_CONFIG_SHA256=$MODEL_CONFIG_SHA256" "MODEL_INDEX_SHA256=$MODEL_INDEX_SHA256"
    "MODEL_SHARD_COUNT=$MODEL_SHARD_COUNT" "MODEL_INDEX_TOTAL_SIZE=$MODEL_INDEX_TOTAL_SIZE"
    "MODEL_HOST_PATH=$MODEL_HOST_PATH" "MODEL_MOUNT_HOST_PATH=$MODEL_MOUNT_HOST_PATH"
    "MODEL_MOUNT_CONTAINER_PATH=$MODEL_MOUNT_CONTAINER_PATH"
    "MODEL_CONTAINER_PATH=$MODEL_CONTAINER_PATH"
    "CACHE_HOST_PATH=$CACHE_HOST_PATH" "LOG_HOST_PATH=$LOG_HOST_PATH"
    "SERVED_MODEL_NAME=$SERVED_MODEL_NAME" "API_PORT=$API_PORT"
    "MASTER_PORT=$MASTER_PORT" "MAX_MODEL_LEN=$MAX_MODEL_LEN"
    "RUNTIME_CONTRACT=$RUNTIME_CONTRACT" "Q40_ENABLED=$Q40_ENABLED"
    "SPARKRING_UPSTREAM_COMMIT=$SPARKRING_UPSTREAM_COMMIT"
    "Q40_HOST_PATH=$Q40_HOST_PATH" "Q40_EXL3_SHA256=$Q40_EXL3_SHA256"
    "Q40_CHECKPOINT_REVISION=$Q40_CHECKPOINT_REVISION"
    "MAX_NUM_SEQS=$MAX_NUM_SEQS" "MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS"
    "GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION" "BLOCK_SIZE=$BLOCK_SIZE"
    "QUANTIZATION=$QUANTIZATION" "LINEAR_BACKEND=$LINEAR_BACKEND"
    "MOE_BACKEND=$MOE_BACKEND" "KDA_PREFILL_BACKEND=$KDA_PREFILL_BACKEND"
    "LOAD_FORMAT=$LOAD_FORMAT" "ENABLE_CHUNKED_PREFILL=$ENABLE_CHUNKED_PREFILL"
    "ENABLE_PREFIX_CACHING=$ENABLE_PREFIX_CACHING" "ASYNC_SCHEDULING=$ASYNC_SCHEDULING"
    "DISABLE_CUSTOM_ALL_REDUCE=$DISABLE_CUSTOM_ALL_REDUCE" "KERNEL_CONFIG=$KERNEL_CONFIG"
    "COMPILATION_CONFIG=$COMPILATION_CONFIG" "VERIFY_CUDA_GRAPHS=$VERIFY_CUDA_GRAPHS"
    "MAX_CUDAGRAPH_CAPTURE_SIZE=$MAX_CUDAGRAPH_CAPTURE_SIZE"
    "DECODE_CONTEXT_PARALLEL_SIZE=$DECODE_CONTEXT_PARALLEL_SIZE"
    "DCP_COMM_BACKEND=$DCP_COMM_BACKEND"
    "DCP_KV_CACHE_INTERLEAVE_SIZE=$DCP_KV_CACHE_INTERLEAVE_SIZE"
    "HF_OVERRIDES=$HF_OVERRIDES"
    "MTP_SPECULATIVE_TOKENS=$MTP_SPECULATIVE_TOKENS"
    "MTP_DRAFT_TP_SIZE=$MTP_DRAFT_TP_SIZE"
    "MTP_MOE_BACKEND=$MTP_MOE_BACKEND"
    "MTP_USE_LOCAL_ARGMAX_REDUCTION=$MTP_USE_LOCAL_ARGMAX_REDUCTION"
    "MTP_DRAFT_SAMPLE_METHOD=$MTP_DRAFT_SAMPLE_METHOD"
    "MTP_REJECTION_SAMPLE_METHOD=$MTP_REJECTION_SAMPLE_METHOD"
    "KV_CACHE_DTYPE=$KV_CACHE_DTYPE" "KV_CACHE_MEMORY_BYTES=${KV_CACHE_MEMORY_BYTES:-}"
    "KV_FP8_ROPE=$KV_FP8_ROPE"
    "VLLM_NVFP4_MLA_DYNAMIC_SCALE=$VLLM_NVFP4_MLA_DYNAMIC_SCALE"
    "VLLM_USE_B12X_DCP_A2A=$VLLM_USE_B12X_DCP_A2A"
    "VLLM_B12X_MLA_CKV_GATHER=$VLLM_B12X_MLA_CKV_GATHER"
    "VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS=$VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS"
    "VLLM_SPARK_MAX_QUERY_ROWS=$VLLM_SPARK_MAX_QUERY_ROWS"
    "VLLM_SPARK_MTP_MODE_ID=$VLLM_SPARK_MTP_MODE_ID"
    "VLLM_SPARK_MTP_TOKENS=$VLLM_SPARK_MTP_TOKENS"
    "VLLM_SPARK_SHARED_CAPTURE_STREAM=$VLLM_SPARK_SHARED_CAPTURE_STREAM"
    "UMA_DROP_CACHES_BEFORE_START=$UMA_DROP_CACHES_BEFORE_START"
    "VLLM_WORKER_MULTIPROC_METHOD=$VLLM_WORKER_MULTIPROC_METHOD"
    "CPU_THREADS_PER_PROCESS=$CPU_THREADS_PER_PROCESS"
    "ENFORCE_EAGER=$ENFORCE_EAGER" "CONTAINER_MEMORY=$CONTAINER_MEMORY"
    "CONTAINER_SHM_SIZE=$CONTAINER_SHM_SIZE" "CONTAINER_NOFILE=$CONTAINER_NOFILE"
    "FABRIC_IFACE=${FABRIC_IFACE:-}" "FABRIC_IPS=$fabric_ips"
    "NCCL_IB_HCA=${NCCL_IB_HCA:-}" "NCCL_IB_GID_INDEX=$NCCL_IB_GID_INDEX"
    "NCCL_IB_ADDR_RANGE=$NCCL_IB_ADDR_RANGE" "NCCL_DEBUG=$NCCL_DEBUG"
    "NETWORK_TOPOLOGY=$NETWORK_TOPOLOGY" "NCCL_CROSS_NIC=$NCCL_CROSS_NIC"
    "NCCL_IB_MERGE_NICS=$NCCL_IB_MERGE_NICS"
    "NCCL_IB_SUBNET_AWARE_ROUTING=$NCCL_IB_SUBNET_AWARE_ROUTING"
    "NCCL_MIN_NCHANNELS=$NCCL_MIN_NCHANNELS" "NCCL_MAX_NCHANNELS=$NCCL_MAX_NCHANNELS"
    "NCCL_ALGO=$NCCL_ALGO"
    "PULL_IMAGE=$PULL_IMAGE" "ALLOW_UNVERIFIED_MODEL=$ALLOW_UNVERIFIED_MODEL"
    "INSTANTTENSOR_BACKEND=$INSTANTTENSOR_BACKEND"
    "INSTANTTENSOR_COPY=${INSTANTTENSOR_COPY:-1}"
    "INSTANTTENSOR_BUFFER_SIZE=$INSTANTTENSOR_BUFFER_SIZE"
    "INSTANTTENSOR_CONCURRENCY=$INSTANTTENSOR_CONCURRENCY"
    "INSTANTTENSOR_IO_DEPTH=$INSTANTTENSOR_IO_DEPTH"
    "INSTANTTENSOR_CHUNK_SIZE=$INSTANTTENSOR_CHUNK_SIZE"
    "INSTANTTENSOR_MAX_FREE_MEM_USAGE=$INSTANTTENSOR_MAX_FREE_MEM_USAGE"
    "ATTENTION_BACKEND=${ATTENTION_BACKEND:-}"
    "ONLINE_QUANT=${ONLINE_QUANT:-none}"
    "ONLINE_QUANT_CONFIG=${ONLINE_QUANT_CONFIG:-}"
    "VLLM_EXL3_PREFILL_CAPACITY=${VLLM_EXL3_PREFILL_CAPACITY:-}"
    "B12X_PCIE_DMA=${B12X_PCIE_DMA:-0}"
    "VERIFY_LOG_WINDOW=${VERIFY_LOG_WINDOW:-30m}"
    "LOG_TAIL_LINES=${LOG_TAIL_LINES:-200}"
  )
  printf -v remote_command '%q ' env "${assignments[@]}" bash -s -- "$node_action" "$rank" "$fabric_ip" "$head_ip"
  ssh "${ssh_options[@]}" "$target" "$remote_command" <"$node_script"
}

prepare_image() {
  if [[ "$BUILD_IMAGE" != "1" ]]; then
    echo "Skipping local image build (BUILD_IMAGE=$BUILD_IMAGE)."
    return
  fi

  command -v tar >/dev/null
  local head_target build_command save_command rank worker_target
  head_target="$(target_for_rank 0)"
  if [[ -n "$IMAGE_BUILD_SCRIPT" ]]; then
    local build_script_path build_assignments
    build_script_path="$IMAGE_BUILD_SCRIPT"
    [[ "$build_script_path" == /* ]] || build_script_path="$root_dir/$build_script_path"
    [[ -f "$build_script_path" ]] || {
      echo "image build script not found: $build_script_path" >&2
      return 1
    }
    printf -v build_assignments '%q ' env \
      "IMAGE=$IMAGE" "MODEL_REVISION=$MODEL_REVISION" \
      "Q40_CHECKPOINT_REVISION=$Q40_CHECKPOINT_REVISION" \
      "SPARKRING_UPSTREAM_COMMIT=${SPARKRING_UPSTREAM_COMMIT:-}" \
      "SPARKRING_BASE_IMAGE=${SPARKRING_BASE_IMAGE:-}" \
      "SPARKRING_BASE_IMAGE_LICENSES=${SPARKRING_BASE_IMAGE_LICENSES:-}" \
      "SPARKRING_BUILD_ROOT=${SPARKRING_BUILD_ROOT:-/var/tmp/sparkring-r7-build}" \
      "Q40_HOST_PATH=$Q40_HOST_PATH"
    echo "Building $IMAGE on rank 0 from pinned SparkRing source..."
    ssh "${ssh_options[@]}" "$head_target" "${build_assignments}bash -s" <"$build_script_path"
  else
    printf -v build_command 'docker build --pull=false -f %q -t %q -' "$image_dockerfile" "$IMAGE"
    echo "Building $IMAGE on rank 0 from the pinned SM121 Dockerfile..."
    tar -C "$root_dir/docker" -cf - . | ssh "${ssh_options[@]}" "$head_target" "$build_command"
  fi

  printf -v save_command 'docker save %q' "$IMAGE"
  for rank in 1 2 3; do
    worker_target="$(target_for_rank "$rank")"
    echo "Streaming $IMAGE from rank 0 to rank $rank..."
    ssh "${ssh_options[@]}" "$head_target" "$save_command" \
      | ssh "${ssh_options[@]}" "$worker_target" 'docker load >/dev/null'
  done

  if [[ "$Q40_ENABLED" == "1" ]]; then
    [[ "$Q40_HOST_PATH" == /* && "$Q40_HOST_PATH" != "/" ]] || {
      echo "Q40_HOST_PATH must be a specific absolute path" >&2
      return 1
    }
    local q40_parent archive_command extract_command bundle_stamp staging_path backup_path
    q40_parent="$(dirname -- "$Q40_HOST_PATH")"
    bundle_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    printf -v archive_command 'test -f %q/manifest.json && tar -C %q -cf - .' \
      "$Q40_HOST_PATH" "$Q40_HOST_PATH"
    for rank in 1 2 3; do
      worker_target="$(target_for_rank "$rank")"
      staging_path="${Q40_HOST_PATH}.incoming.${bundle_stamp}"
      backup_path="${Q40_HOST_PATH}.backup.${bundle_stamp}"
      printf -v extract_command \
        'test ! -e %q && test ! -e %q && mkdir -p %q && tar -C %q -xf - && test -f %q/manifest.json && { test ! -e %q || mv %q %q; } && mv %q %q' \
        "$staging_path" "$backup_path" "$staging_path" "$staging_path" "$staging_path" \
        "$Q40_HOST_PATH" "$Q40_HOST_PATH" "$backup_path" \
        "$staging_path" "$Q40_HOST_PATH"
      echo "Streaming the image-bound Q40 bundle from rank 0 to rank $rank..."
      ssh "${ssh_options[@]}" "$head_target" "$archive_command" \
        | ssh "${ssh_options[@]}" "$worker_target" "$extract_command"
    done
  fi
}

run_all_parallel() {
  local node_action="$1" failures=0 pids=() rank
  for rank in 0 1 2 3; do
    remote_node "$node_action" "$rank" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || failures=$((failures + 1))
  done
  (( failures == 0 )) || { echo "$node_action failed on $failures node(s)" >&2; return 1; }
}

stop_all() {
  run_all_parallel stop
}

benchmark_head() {
  local head_management output_file profile_name
  head_management="$(inventory_value 1 MANAGEMENT_IP)"
  profile_name=baseline
  [[ -z "$profile_file" ]] || profile_name="$(basename "$profile_file")"
  if [[ "$RUNTIME_CONTRACT" == "sparkring-r7-switch-q40" ]]; then
    echo "NOTICE: this benchmark is a deployment diagnostic, not SparkRing's normalized reference harness." >&2
    echo "Do not compare its output directly with the published SparkRing table or make a speed claim from it." >&2
  fi
  mkdir -p "$root_dir/benchmarks/results"
  output_file="$root_dir/benchmarks/results/$(date -u +%Y%m%dT%H%M%SZ)-${deployment_slug}.json"
  python3 "$root_dir/scripts/benchmark-glm53.py" \
    --base-url "http://${head_management}:${API_PORT}/v1" \
    --model "$SERVED_MODEL_NAME" \
    --runs "$BENCHMARK_RUNS" \
    --warmups "$BENCHMARK_WARMUPS" \
    --prefill-words "$BENCHMARK_PREFILL_WORDS" \
    --decode-tokens "$BENCHMARK_DECODE_TOKENS" \
    --profile "$profile_name" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --mtp-tokens "$MTP_SPECULATIVE_TOKENS" \
    --compilation-config "$COMPILATION_CONFIG" \
    --output "$output_file"
}

start_all() {
  echo "Stopping every old rank before rendezvous..."
  stop_all

  local rank
  for rank in 3 2 1; do
    remote_node start "$rank"
    sleep "$START_STAGGER_SECONDS"
  done
  remote_node start 0

  local head_management deadline
  head_management="$(inventory_value 1 MANAGEMENT_IP)"
  deadline=$((SECONDS + HEALTH_TIMEOUT_SECONDS))
  echo "Waiting up to ${HEALTH_TIMEOUT_SECONDS}s for http://${head_management}:${API_PORT}/health ..."
  until curl --silent --show-error --fail --max-time 5 "http://${head_management}:${API_PORT}/health" >/dev/null 2>&1; do
    if ! run_all_parallel running; then
      echo "one or more ranks exited before the health endpoint became ready" >&2
      run_all_parallel status >&2 || true
      return 1
    fi
    if (( SECONDS >= deadline )); then
      echo "health check timed out; recent head logs follow" >&2
      remote_node logs 0 >&2 || true
      return 1
    fi
    sleep 10
  done
  if [[ "$RUN_SMOKE_TEST" == "1" ]]; then
    BASE_URL="http://${head_management}:${API_PORT}/v1" \
      SERVED_MODEL_NAME="$SERVED_MODEL_NAME" \
      EXPECTED_MAX_MODEL_LEN="$MAX_MODEL_LEN" \
      bash "$root_dir/scripts/smoke-test-glm53.sh"
  fi
  run_all_parallel verify
  echo "$deployment_label is ready at http://${head_management}:${API_PORT}/v1"
}

case "$action" in
  build)
    prepare_image
    ;;
  prepare)
    if [[ "$PREPARE_IMAGE" == "1" ]]; then
      prepare_image
    else
      echo "PREPARE_IMAGE=0: using the existing runtime image and local model files; no build or pull will run."
    fi
    run_all_parallel prepare
    ;;
  preflight) run_all_parallel preflight ;;
  start) start_all ;;
  stop) stop_all ;;
  status) run_all_parallel status ;;
  verify)
    run_all_parallel verify
    head_management="$(inventory_value 1 MANAGEMENT_IP)"
    BASE_URL="http://${head_management}:${API_PORT}/v1" \
      SERVED_MODEL_NAME="$SERVED_MODEL_NAME" \
      EXPECTED_MAX_MODEL_LEN="$MAX_MODEL_LEN" \
      bash "$root_dir/scripts/smoke-test-glm53.sh"
    ;;
  benchmark) benchmark_head ;;
  logs)
    if [[ -n "$selected_rank" ]]; then
      [[ "$selected_rank" =~ ^[0-3]$ ]] || { echo "rank must be 0-3" >&2; exit 2; }
      remote_node logs "$selected_rank"
    else
      run_all_parallel logs
    fi
    ;;
  *)
    echo "usage: $0 [build|prepare|preflight|start|stop|status|verify|benchmark|logs [0-3]]" >&2
    exit 2
    ;;
esac
