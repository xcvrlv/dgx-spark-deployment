#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
inventory_file="${INVENTORY_FILE:-$root_dir/sparks.env}"
recipe_file="${RECIPE_FILE:-$root_dir/recipes/glm-5.3-flash-nvfp4.env}"
profile_file="${PROFILE_FILE:-}"
node_script="$root_dir/scripts/glm53-node.sh"

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
[[ -f "$root_dir/.env" ]] && source "$root_dir/.env"
MODEL_MOUNT_HOST_PATH="${MODEL_MOUNT_HOST_PATH:-$MODEL_HOST_PATH}"
MODEL_MOUNT_CONTAINER_PATH="${MODEL_MOUNT_CONTAINER_PATH:-$MODEL_CONTAINER_PATH}"
set +a

action="${1:-start}"
selected_rank="${2:-}"
[[ "${TP_NODE_COUNT:-}" == "4" ]] || { echo "this recipe requires TP_NODE_COUNT=4" >&2; exit 2; }
[[ "$CLUSTER_RAIL" =~ ^CX[01]$ ]] || { echo "CLUSTER_RAIL must be CX0 or CX1" >&2; exit 2; }
[[ "$BUILD_IMAGE" =~ ^[01]$ && "$PULL_IMAGE" =~ ^[01]$ ]] || {
  echo "BUILD_IMAGE and PULL_IMAGE must be 0 or 1" >&2
  exit 2
}
[[ "$BUILD_IMAGE" != "1" || "$PULL_IMAGE" != "1" ]] || {
  echo "BUILD_IMAGE and PULL_IMAGE cannot both be enabled" >&2
  exit 2
}
for boolean_name in ENFORCE_EAGER VERIFY_CUDA_GRAPHS MTP_USE_LOCAL_ARGMAX_REDUCTION; do
  [[ "${!boolean_name}" =~ ^[01]$ ]] || {
    echo "$boolean_name must be 0 or 1" >&2
    exit 2
  }
done
[[ "$MTP_SPECULATIVE_TOKENS" =~ ^[0-9]+$ ]] || {
  echo "MTP_SPECULATIVE_TOKENS must be a non-negative integer" >&2
  exit 2
}
if [[ "$MTP_USE_LOCAL_ARGMAX_REDUCTION" == "1" && "$MTP_DRAFT_SAMPLE_METHOD" != "greedy" ]]; then
  echo "local argmax reduction requires MTP_DRAFT_SAMPLE_METHOD=greedy" >&2
  exit 2
fi
if (( MTP_SPECULATIVE_TOKENS > 0 )); then
  case "$MTP_MOE_BACKEND" in
    marlin|triton|batched_triton|flashinfer_trtllm|flashinfer_cutlass|aiter) ;;
    *)
      echo "unsupported MTP_MOE_BACKEND=$MTP_MOE_BACKEND" >&2
      exit 2
      ;;
  esac
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
  local management_ip fabric_ip head_ip target remote_command
  management_ip="$(inventory_value "$node" MANAGEMENT_IP)"
  fabric_ip="$(inventory_value "$node" "${CLUSTER_RAIL}_IP")"
  head_ip="$(inventory_value 1 "${CLUSTER_RAIL}_IP")"
  target="$(ssh_target "$management_ip")"

  local assignments=(
    "IMAGE=$IMAGE" "CONTAINER_NAME=$CONTAINER_NAME"
    "MODEL_ID=$MODEL_ID" "MODEL_REVISION=$MODEL_REVISION"
    "MODEL_HOST_PATH=$MODEL_HOST_PATH" "MODEL_MOUNT_HOST_PATH=$MODEL_MOUNT_HOST_PATH"
    "MODEL_MOUNT_CONTAINER_PATH=$MODEL_MOUNT_CONTAINER_PATH"
    "MODEL_CONTAINER_PATH=$MODEL_CONTAINER_PATH"
    "CACHE_HOST_PATH=$CACHE_HOST_PATH" "LOG_HOST_PATH=$LOG_HOST_PATH"
    "SERVED_MODEL_NAME=$SERVED_MODEL_NAME" "API_PORT=$API_PORT"
    "MASTER_PORT=$MASTER_PORT" "MAX_MODEL_LEN=$MAX_MODEL_LEN"
    "MAX_NUM_SEQS=$MAX_NUM_SEQS" "MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS"
    "GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION" "BLOCK_SIZE=$BLOCK_SIZE"
    "QUANTIZATION=$QUANTIZATION" "LINEAR_BACKEND=$LINEAR_BACKEND"
    "MOE_BACKEND=$MOE_BACKEND" "KDA_PREFILL_BACKEND=$KDA_PREFILL_BACKEND"
    "LOAD_FORMAT=$LOAD_FORMAT" "ENABLE_CHUNKED_PREFILL=$ENABLE_CHUNKED_PREFILL"
    "ENABLE_PREFIX_CACHING=$ENABLE_PREFIX_CACHING" "ASYNC_SCHEDULING=$ASYNC_SCHEDULING"
    "DISABLE_CUSTOM_ALL_REDUCE=$DISABLE_CUSTOM_ALL_REDUCE" "KERNEL_CONFIG=$KERNEL_CONFIG"
    "COMPILATION_CONFIG=$COMPILATION_CONFIG" "VERIFY_CUDA_GRAPHS=$VERIFY_CUDA_GRAPHS"
    "MTP_SPECULATIVE_TOKENS=$MTP_SPECULATIVE_TOKENS"
    "MTP_DRAFT_TP_SIZE=$MTP_DRAFT_TP_SIZE"
    "MTP_MOE_BACKEND=$MTP_MOE_BACKEND"
    "MTP_USE_LOCAL_ARGMAX_REDUCTION=$MTP_USE_LOCAL_ARGMAX_REDUCTION"
    "MTP_DRAFT_SAMPLE_METHOD=$MTP_DRAFT_SAMPLE_METHOD"
    "MTP_REJECTION_SAMPLE_METHOD=$MTP_REJECTION_SAMPLE_METHOD"
    "KV_CACHE_DTYPE=$KV_CACHE_DTYPE" "KV_CACHE_MEMORY_BYTES=${KV_CACHE_MEMORY_BYTES:-}"
    "ENFORCE_EAGER=$ENFORCE_EAGER" "CONTAINER_MEMORY=$CONTAINER_MEMORY"
    "CONTAINER_SHM_SIZE=$CONTAINER_SHM_SIZE" "CONTAINER_NOFILE=$CONTAINER_NOFILE"
    "FABRIC_IFACE=${FABRIC_IFACE:-}"
    "NCCL_IB_HCA=${NCCL_IB_HCA:-}" "NCCL_IB_GID_INDEX=$NCCL_IB_GID_INDEX"
    "NCCL_IB_ADDR_RANGE=$NCCL_IB_ADDR_RANGE" "NCCL_DEBUG=$NCCL_DEBUG"
    "PULL_IMAGE=$PULL_IMAGE" "ALLOW_UNVERIFIED_MODEL=$ALLOW_UNVERIFIED_MODEL"
    "INSTANTTENSOR_BACKEND=$INSTANTTENSOR_BACKEND"
    "INSTANTTENSOR_BUFFER_SIZE=$INSTANTTENSOR_BUFFER_SIZE"
    "INSTANTTENSOR_CONCURRENCY=$INSTANTTENSOR_CONCURRENCY"
    "INSTANTTENSOR_IO_DEPTH=$INSTANTTENSOR_IO_DEPTH"
    "INSTANTTENSOR_CHUNK_SIZE=$INSTANTTENSOR_CHUNK_SIZE"
    "INSTANTTENSOR_MAX_FREE_MEM_USAGE=$INSTANTTENSOR_MAX_FREE_MEM_USAGE"
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
  printf -v build_command 'docker build --pull=false -f Dockerfile.glm53-sm121 -t %q -' "$IMAGE"
  echo "Building $IMAGE on rank 0 from the pinned SM121 Dockerfile..."
  tar -C "$root_dir/docker" -cf - . | ssh "${ssh_options[@]}" "$head_target" "$build_command"

  printf -v save_command 'docker save %q' "$IMAGE"
  for rank in 1 2 3; do
    worker_target="$(target_for_rank "$rank")"
    echo "Streaming $IMAGE from rank 0 to rank $rank..."
    ssh "${ssh_options[@]}" "$head_target" "$save_command" \
      | ssh "${ssh_options[@]}" "$worker_target" 'docker load >/dev/null'
  done
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
  mkdir -p "$root_dir/benchmarks/results"
  output_file="$root_dir/benchmarks/results/$(date -u +%Y%m%dT%H%M%SZ)-glm53.json"
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
  echo "Preflighting all four Sparks..."
  run_all_parallel preflight
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
  echo "GLM-5.3 Flash is ready at http://${head_management}:${API_PORT}/v1"
}

case "$action" in
  build)
    prepare_image
    ;;
  prepare)
    prepare_image
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
