#!/usr/bin/env bash
set -euo pipefail

action="${1:?usage: glm52-exl3-node.sh <prepare|preflight|start|stop|status|verify|logs> <rank> <host-ip> <head-ip>}"
rank="${2:?missing node rank}"
host_ip="${3:?missing host IP}"
head_ip="${4:?missing head IP}"

required_env=(
  IMAGE CONTAINER_NAME MODEL_ID MODEL_REVISION MODEL_HOST_PATH MODEL_MOUNT_HOST_PATH
  MODEL_MOUNT_CONTAINER_PATH MODEL_CONTAINER_PATH CACHE_HOST_PATH LOG_HOST_PATH
  MODEL_PROVISIONING
  MODEL_CONFIG_SHA256 MODEL_INDEX_SHA256 MODEL_SHARD_COUNT MODEL_INDEX_TOTAL_SIZE
  RUNTIME_CONTRACT SPARKRING_UPSTREAM_COMMIT Q40_ENABLED Q40_HOST_PATH Q40_EXL3_SHA256
  Q40_CHECKPOINT_REVISION
  SERVED_MODEL_NAME API_PORT MASTER_PORT MAX_MODEL_LEN MAX_NUM_SEQS
  MAX_NUM_BATCHED_TOKENS GPU_MEMORY_UTILIZATION QUANTIZATION ATTENTION_BACKEND
  LINEAR_BACKEND MOE_BACKEND LOAD_FORMAT ONLINE_QUANT ONLINE_QUANT_CONFIG
  VLLM_EXL3_PREFILL_CAPACITY B12X_PCIE_DMA MTP_SPECULATIVE_TOKENS
  MTP_DRAFT_TP_SIZE MTP_MOE_BACKEND MTP_USE_LOCAL_ARGMAX_REDUCTION
  MTP_DRAFT_SAMPLE_METHOD MTP_REJECTION_SAMPLE_METHOD KV_CACHE_DTYPE
  DECODE_CONTEXT_PARALLEL_SIZE DCP_COMM_BACKEND DCP_KV_CACHE_INTERLEAVE_SIZE
  HF_OVERRIDES MAX_CUDAGRAPH_CAPTURE_SIZE KV_FP8_ROPE
  VLLM_NVFP4_MLA_DYNAMIC_SCALE VLLM_USE_B12X_DCP_A2A
  VLLM_B12X_MLA_CKV_GATHER VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS
  VLLM_SPARK_MAX_QUERY_ROWS VLLM_SPARK_MTP_MODE_ID VLLM_SPARK_MTP_TOKENS
  VLLM_SPARK_SHARED_CAPTURE_STREAM UMA_DROP_CACHES_BEFORE_START
  VLLM_UMA_USE_MEM_AVAILABLE
  VLLM_WORKER_MULTIPROC_METHOD CPU_THREADS_PER_PROCESS
  ENABLE_CHUNKED_PREFILL ENABLE_PREFIX_CACHING ASYNC_SCHEDULING
  DISABLE_CUSTOM_ALL_REDUCE COMPILATION_CONFIG VERIFY_CUDA_GRAPHS ENFORCE_EAGER
  CONTAINER_MEMORY CONTAINER_SHM_SIZE CONTAINER_NOFILE NCCL_IB_GID_INDEX
  NCCL_IB_ADDR_RANGE NCCL_DEBUG NETWORK_TOPOLOGY FABRIC_IPS NCCL_CROSS_NIC
  NCCL_IB_MERGE_NICS NCCL_IB_SUBNET_AWARE_ROUTING NCCL_MIN_NCHANNELS
  NCCL_MAX_NCHANNELS NCCL_ALGO PULL_IMAGE ALLOW_UNVERIFIED_MODEL
  INSTANTTENSOR_BACKEND INSTANTTENSOR_COPY INSTANTTENSOR_BUFFER_SIZE
  INSTANTTENSOR_CONCURRENCY INSTANTTENSOR_IO_DEPTH INSTANTTENSOR_CHUNK_SIZE
  INSTANTTENSOR_MAX_FREE_MEM_USAGE
)
for name in "${required_env[@]}"; do
  [[ -n "${!name+x}" ]] || { echo "missing environment variable: $name" >&2; exit 2; }
done

effective_model_host_path="$MODEL_HOST_PATH"
effective_model_mount_host_path="$MODEL_MOUNT_HOST_PATH"
effective_model_mount_container_path="$MODEL_MOUNT_CONTAINER_PATH"
effective_model_container_path="$MODEL_CONTAINER_PATH"

resolve_model_paths() {
  effective_model_host_path="$MODEL_HOST_PATH"
  effective_model_mount_host_path="$MODEL_MOUNT_HOST_PATH"
  effective_model_mount_container_path="$MODEL_MOUNT_CONTAINER_PATH"
  effective_model_container_path="$MODEL_CONTAINER_PATH"

  if [[ -f "$MODEL_HOST_PATH/config.json" && -f "$MODEL_HOST_PATH/model.safetensors.index.json" ]]; then
    if [[ "$MODEL_CONTAINER_PATH" == "auto" ]]; then
      effective_model_container_path="$MODEL_MOUNT_CONTAINER_PATH"
    fi
    return
  fi

  [[ -d "$MODEL_HOST_PATH/snapshots" ]] || {
    echo "local model is neither a ready model directory nor a Hugging Face cache root: $MODEL_HOST_PATH" >&2
    return 1
  }

  local revision="" candidate=""
  if [[ -f "$MODEL_HOST_PATH/refs/main" ]]; then
    revision="$(tr -d '\r\n' <"$MODEL_HOST_PATH/refs/main")"
    candidate="$MODEL_HOST_PATH/snapshots/$revision"
    if [[ ! -f "$candidate/config.json" || ! -f "$candidate/model.safetensors.index.json" ]]; then
      echo "refs/main points to an incomplete local snapshot: $candidate" >&2
      return 1
    fi
  else
    candidate="$(find "$MODEL_HOST_PATH/snapshots" -mindepth 2 -maxdepth 2 \
      -name config.json -printf '%T@ %h\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
    [[ -n "$candidate" && -f "$candidate/model.safetensors.index.json" ]] || {
      echo "no complete local snapshot found under $MODEL_HOST_PATH/snapshots" >&2
      return 1
    }
    revision="$(basename -- "$candidate")"
    echo "refs/main is absent; using newest complete local snapshot $revision" >&2
  fi

  effective_model_host_path="$candidate"
  effective_model_container_path="$MODEL_MOUNT_CONTAINER_PATH/snapshots/$revision"
}

interface_for_ip() {
  local wanted_ip="$1" iface
  iface="$(ip -o -4 addr show | awk -v wanted="$wanted_ip" '$4 ~ ("^" wanted "/") { print $2; exit }')"
  [[ -n "$iface" ]] || { echo "no Linux interface owns fabric address $wanted_ip" >&2; return 1; }
  printf '%s\n' "$iface"
}

resolve_fabric_iface() {
  if [[ -n "${FABRIC_IFACE:-}" ]]; then
    printf '%s\n' "$FABRIC_IFACE"
    return
  fi
  interface_for_ip "$host_ip"
}

resolve_hca_for_iface() {
  local iface="$1"
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

resolve_hca() {
  local primary_iface="$1"
  if [[ -n "${NCCL_IB_HCA:-}" ]]; then
    printf '%s\n' "$NCCL_IB_HCA"
    return
  fi

  local -a fabric_ips=() resolved_hcas=()
  local fabric_ip iface hca existing
  if [[ "$NETWORK_TOPOLOGY" == "switch-star-dual-rail" ]]; then
    IFS=',' read -r -a fabric_ips <<<"$FABRIC_IPS"
  else
    fabric_ips=("$host_ip")
  fi
  for fabric_ip in "${fabric_ips[@]}"; do
    iface="$(interface_for_ip "$fabric_ip")"
    hca="$(resolve_hca_for_iface "$iface")"
    existing=0
    local known_hca
    for known_hca in "${resolved_hcas[@]}"; do
      [[ "$known_hca" == "$hca" ]] && existing=1
    done
    (( existing == 1 )) || resolved_hcas+=("$hca")
  done
  if [[ "$NETWORK_TOPOLOGY" == "switch-star-dual-rail" && "${#resolved_hcas[@]}" -ne 2 ]]; then
    echo "dual-rail discovery expected two distinct RDMA HCAs, found: ${resolved_hcas[*]}" >&2
    return 1
  fi
  local joined
  joined="$(IFS=,; echo "${resolved_hcas[*]}")"
  printf '%s\n' "$joined"
}

validate_hcas() {
  local hca_list="$1" raw_hca hca
  local -a configured_hcas=()
  IFS=',' read -r -a configured_hcas <<<"$hca_list"
  for raw_hca in "${configured_hcas[@]}"; do
    hca="${raw_hca%%:*}"
    [[ -d "/sys/class/infiniband/$hca" ]] || {
      echo "configured RDMA HCA is absent: $hca" >&2
      return 1
    }
  done
}

validate_fabric_mtus() {
  local -a fabric_ips=()
  local fabric_ip iface iface_mtu
  if [[ "$NETWORK_TOPOLOGY" == "switch-star-dual-rail" ]]; then
    IFS=',' read -r -a fabric_ips <<<"$FABRIC_IPS"
  else
    fabric_ips=("$host_ip")
  fi
  for fabric_ip in "${fabric_ips[@]}"; do
    iface="$(interface_for_ip "$fabric_ip")"
    iface_mtu="$(<"/sys/class/net/$iface/mtu")"
    (( iface_mtu >= 9000 )) || {
      echo "fabric interface $iface ($fabric_ip) has MTU $iface_mtu; switched profile requires at least 9000" >&2
      return 1
    }
  done
}

capture_logs() {
  if docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    mkdir -p "$LOG_HOST_PATH"
    docker logs "$CONTAINER_NAME" >"$LOG_HOST_PATH/$(date -u +%Y%m%dT%H%M%SZ)-rank${rank}.log" 2>&1 || true
  fi
}

reclaim_uma_page_cache() {
  [[ "$UMA_DROP_CACHES_BEFORE_START" == "1" ]] || return 0

  echo "Reclaiming host page cache before vLLM's GB10 UMA memory admission check..."
  sync
  if [[ -w /proc/sys/vm/drop_caches ]]; then
    printf '3\n' >/proc/sys/vm/drop_caches
  elif docker run --rm --privileged --pid host --entrypoint sh "$IMAGE" \
    -c 'sync; echo 3 > /proc/sys/vm/drop_caches'; then
    :
  elif command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
    printf '3\n' | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
  else
    echo "UMA cache reclaim failed through Docker and sudo; run 'sudo sh -c \"sync; echo 3 > /proc/sys/vm/drop_caches\"' on this node or set UMA_DROP_CACHES_BEFORE_START=0" >&2
    return 1
  fi
  awk '/^(MemFree|MemAvailable|Cached):/ { printf "%s %s %s\n", $1, $2, $3 }' /proc/meminfo
}

prepare_uma_memory_overlay() {
  uma_utils_overlay=""
  [[ "$VLLM_UMA_USE_MEM_AVAILABLE" == "1" ]] || return 0

  local image_id overlay_dir source_path temp_path
  image_id="$(docker image inspect "$IMAGE" --format '{{.Id}}')"
  overlay_dir="$CACHE_HOST_PATH/overlays/${image_id#sha256:}"
  source_path=/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/utils.py
  uma_utils_overlay="$overlay_dir/utils.py"
  mkdir -p "$overlay_dir"
  temp_path="$(mktemp "$overlay_dir/utils.py.source.XXXXXX")"
  docker run --rm --entrypoint cat "$IMAGE" "$source_path" >"$temp_path"

  python3 - "$temp_path" "$uma_utils_overlay" <<'PY'
import sys
from pathlib import Path

source_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
source = source_path.read_text(encoding="utf-8")
marker = "GB10 UMA startup admission"

if marker not in source:
    import_anchor = "import math\n"
    if source.count(import_anchor) != 1:
        raise SystemExit("refusing UMA overlay: pinned utils.py import anchor drifted")
    source = source.replace(import_anchor, "import math\nimport os\n", 1)

    old = '''    if init_snapshot.free_memory < requested_memory:
        raise ValueError(
            f"Free memory on device {init_snapshot.device_} "
            f"({format_gib(init_snapshot.free_memory)}/"
'''
    new = '''    available_memory = init_snapshot.free_memory
    if os.getenv("VLLM_UMA_USE_MEM_AVAILABLE", "0") == "1":
        try:
            with open("/proc/meminfo", encoding="utf-8") as meminfo:
                meminfo_kib = {
                    parts[0].rstrip(":"): int(parts[1])
                    for line in meminfo
                    if len(parts := line.split()) >= 2 and parts[1].isdigit()
                }
            mem_available_kib = meminfo_kib["MemAvailable"]
            available_memory = mem_available_kib * 1024
            cgroup_current = 0
            try:
                with open(
                    "/sys/fs/cgroup/memory.current", encoding="utf-8"
                ) as current_file:
                    cgroup_current = int(current_file.read().strip())
            except (OSError, ValueError):
                pass
            logger.warning(
                "GB10 UMA startup admission: CUDA MemFree=%s GiB, "
                "Linux MemAvailable=%s GiB, Cached=%s GiB, "
                "SReclaimable=%s GiB, Shmem=%s GiB, cgroup_current=%s GiB; "
                "using MemAvailable",
                format_gib(init_snapshot.free_memory),
                format_gib(available_memory),
                format_gib(meminfo_kib.get("Cached", 0) * 1024),
                format_gib(meminfo_kib.get("SReclaimable", 0) * 1024),
                format_gib(meminfo_kib.get("Shmem", 0) * 1024),
                format_gib(cgroup_current),
            )
        except (KeyError, OSError, ValueError):
            logger.exception(
                "GB10 UMA MemAvailable lookup failed; falling back to CUDA MemFree"
            )

    if available_memory < requested_memory:
        raise ValueError(
            f"Free memory on device {init_snapshot.device_} "
            f"({format_gib(available_memory)}/"
'''
    if source.count(old) != 1:
        raise SystemExit("refusing UMA overlay: pinned request_memory block drifted")
    source = source.replace(old, new, 1)

output_path.write_text(source, encoding="utf-8")
PY
  rm -f -- "$temp_path"
  chmod 0444 "$uma_utils_overlay"
  echo "Prepared image-bound GB10 UMA memory overlay: $uma_utils_overlay"
}

prepare_node() {
  command -v docker >/dev/null
  mkdir -p "$CACHE_HOST_PATH" "$LOG_HOST_PATH"
  if [[ "$MODEL_PROVISIONING" == "local" ]]; then
    docker image inspect "$IMAGE" >/dev/null
    resolve_model_paths
    test -f "$effective_model_host_path/config.json"
    test -f "$effective_model_host_path/model.safetensors.index.json"
    echo "rank=$rank local model ready at $effective_model_host_path (offline; no files downloaded)"
    return
  fi

  [[ "$MODEL_PROVISIONING" == "download" ]] || {
    echo "unsupported MODEL_PROVISIONING=$MODEL_PROVISIONING" >&2
    return 1
  }
  mkdir -p "$MODEL_HOST_PATH"
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
  local -a cuda_compat_env=()
  if [[ "$RUNTIME_CONTRACT" == "sparkring-r7-switch-q40" ]]; then
    cuda_compat_env=(
      -e LD_PRELOAD=/usr/local/cuda/compat/libcuda.so.1
      -e VLLM_NCCL_SO_PATH=/opt/sparkring/nccl/libnccl.so.2
    )
  fi
  command -v docker >/dev/null
  command -v ip >/dev/null
  docker info >/dev/null
  docker image inspect "$IMAGE" >/dev/null
  resolve_model_paths
  test -f "$effective_model_host_path/config.json"
  test -f "$effective_model_host_path/model.safetensors.index.json"
  test -d /dev/infiniband

  python3 - "$effective_model_host_path" "$MODEL_CONFIG_SHA256" "$MODEL_INDEX_SHA256" \
    "$MODEL_SHARD_COUNT" "$MODEL_INDEX_TOTAL_SIZE" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
expected_config_sha, expected_index_sha = sys.argv[2:4]
expected_shards, expected_total_size = sys.argv[4:6]
config_path = model_dir / "config.json"
index_path = model_dir / "model.safetensors.index.json"
if expected_config_sha and hashlib.sha256(config_path.read_bytes()).hexdigest() != expected_config_sha:
    raise SystemExit("model config SHA-256 mismatch")
if expected_index_sha and hashlib.sha256(index_path.read_bytes()).hexdigest() != expected_index_sha:
    raise SystemExit("model index SHA-256 mismatch")
config = json.loads(config_path.read_text())
tail = config.get("hybrid_tr3_tail", {})
if tail.get("tp") not in (None, 4):
    raise SystemExit(f"checkpoint declares incompatible TP slicing: {tail.get('tp')!r}")
formats = {
    str(tail.get("format", "")).lower(),
    str(config.get("quantization_config", {}).get("quant_method", "")).lower(),
}
if not any("exl" in value for value in formats):
    print("warning: config has no recognized EXL marker; deferring compatibility to vLLM", file=sys.stderr)
index = json.loads(index_path.read_text())
shards = sorted(set(index.get("weight_map", {}).values()))
if not shards:
    raise SystemExit("model index has no weight shards")
if expected_shards and len(shards) != int(expected_shards):
    raise SystemExit(f"model shard count mismatch: expected {expected_shards}, got {len(shards)}")
reported_total = index.get("metadata", {}).get("total_size")
if expected_total_size and reported_total != int(expected_total_size):
    raise SystemExit(
        f"model index total_size mismatch: expected {expected_total_size}, got {reported_total}"
    )
missing = [name for name in shards if not (model_dir / name).is_file()]
if missing:
    raise SystemExit(f"model is missing {len(missing)} shard(s): {missing[:5]}")
PY

  local iface hca installed_revision image_arch vllm_tree b12x_tree image_id
  iface="$(resolve_fabric_iface)"
  hca="$(resolve_hca "$iface")"
  validate_hcas "$hca"
  validate_fabric_mtus
  if [[ "$MODEL_PROVISIONING" == "local" ]]; then
    : # The operator owns local model identity and lifecycle.
  elif [[ -f "$MODEL_HOST_PATH/.spark-deployment-revision" ]]; then
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
  image_id="$(docker image inspect "$IMAGE" --format '{{.Id}}')"
  case "$RUNTIME_CONTRACT" in
    local-inference)
      vllm_tree="$(docker image inspect "$IMAGE" --format '{{index .Config.Labels "local-inference.vllm.integration.tree"}}')"
      b12x_tree="$(docker image inspect "$IMAGE" --format '{{index .Config.Labels "local-inference.b12x.integration.tree"}}')"
      [[ "$vllm_tree" == "b0f8c85c7b96497e0148a18230f43d18854ae04a" ]] || {
        echo "unexpected vLLM integration tree: $vllm_tree" >&2; return 1;
      }
      [[ "$b12x_tree" == "cd3ce190f0f1917402cdfd5773724267cc9a63f8" ]] || {
        echo "unexpected B12X integration tree: $b12x_tree" >&2; return 1;
      }
      ;;
    sparkring-r7-switch-q40)
      local image_revision
      image_revision="$(docker image inspect "$IMAGE" --format '{{index .Config.Labels "org.opencontainers.image.revision"}}')"
      [[ -n "$SPARKRING_UPSTREAM_COMMIT" && "$image_revision" == "$SPARKRING_UPSTREAM_COMMIT" ]] || {
        echo "SparkRing image revision mismatch: have $image_revision, want $SPARKRING_UPSTREAM_COMMIT" >&2
        return 1
      }
      docker run --rm --gpus all "${cuda_compat_env[@]}" "$IMAGE" /bin/true >/dev/null
      ;;
    *) echo "unsupported RUNTIME_CONTRACT=$RUNTIME_CONTRACT" >&2; return 1 ;;
  esac

  docker run --rm --gpus all --entrypoint python3 \
    "${cuda_compat_env[@]}" -e "GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION" \
    -e "UMA_DROP_CACHES_BEFORE_START=$UMA_DROP_CACHES_BEFORE_START" \
    "$IMAGE" -c '
import os
import torch

free_bytes, total_bytes = torch.cuda.mem_get_info()
utilization = float(os.environ["GPU_MEMORY_UTILIZATION"])
required_bytes = total_bytes * utilization
properties = torch.cuda.get_device_properties(0)
profile_declares_uma = os.environ["UMA_DROP_CACHES_BEFORE_START"] == "1"
integrated = profile_declares_uma or bool(getattr(properties, "integrated", False))
mem_available_bytes = 0
with open("/proc/meminfo", encoding="utf-8") as meminfo:
    for line in meminfo:
        if line.startswith("MemAvailable:"):
            mem_available_bytes = int(line.split()[1]) * 1024
            break
available_bytes = mem_available_bytes if integrated and mem_available_bytes else free_bytes
available_source = "MemAvailable" if integrated and mem_available_bytes else "CUDA MemFree"
gib = 1024 ** 3
print(
    f"GPU memory health: cuda_free={free_bytes / gib:.2f} GiB "
    f"mem_available={mem_available_bytes / gib:.2f} GiB "
    f"required={required_bytes / gib:.2f} GiB integrated={integrated} "
    f"decision_source={available_source}"
)
if available_bytes < required_bytes:
    raise SystemExit(
        "GPU memory health failed: stop competing GPU workloads or repair/reboot the node"
    )
'

  if [[ "$Q40_ENABLED" == "1" ]]; then
    test -f "$Q40_HOST_PATH/exl3.py"
    test -f "$Q40_HOST_PATH/model_runner.py"
    test -f "$Q40_HOST_PATH/manifest.json"
    python3 - "$Q40_HOST_PATH" "$image_id" "$Q40_CHECKPOINT_REVISION" "$Q40_EXL3_SHA256" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = json.loads((root / "manifest.json").read_text())
if manifest.get("schema") != "spark-deployment-sparkring-q40/v1":
    raise SystemExit("Q40 manifest schema mismatch")
if manifest.get("image_id") != sys.argv[2]:
    raise SystemExit("Q40 bundle was generated for a different image ID")
if manifest.get("model_revision") != sys.argv[3]:
    raise SystemExit("Q40 bundle was generated for a different model revision")
for name in ("exl3.py", "model_runner.py"):
    digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
    if manifest.get("files", {}).get(name, {}).get("sha256") != digest:
        raise SystemExit(f"Q40 {name} hash mismatch")
if sys.argv[4] and manifest["files"]["exl3.py"]["sha256"] != sys.argv[4]:
    raise SystemExit("Q40 EXL3 contract hash mismatch")
PY
  fi

  docker run --rm --gpus all --entrypoint python3 "${cuda_compat_env[@]}" "$IMAGE" -c \
    'import importlib,sys,torch; cap=torch.cuda.get_device_capability(); assert cap == (12, 1), cap; import b12x,vllm; sys.path[:0]=["/opt/exllamav3","/opt/exllamav3-python"]; ext=importlib.import_module("exllamav3_ext"); assert hasattr(ext,"exl3_moe_fused_retile"); print("SM121 EXL3 imports ok")' \
    >/dev/null
  docker run --rm --gpus all --entrypoint python3 "${cuda_compat_env[@]}" "$IMAGE" -c \
    'from pathlib import Path; import vllm; root=Path(vllm.__file__).parent; assert (root / "model_executor/layers/quantization/exl3.py").is_file(); assert (root / "v1/attention/backends/mla/b12x_mla_sparse.py").is_file()' \
    >/dev/null
  docker run --rm --entrypoint sh \
    -v "$effective_model_mount_host_path:$effective_model_mount_container_path:ro" \
    "$IMAGE" -c 'test -f "$1/config.json" && test -f "$1/model.safetensors.index.json"' \
    _ "$effective_model_container_path" || {
      echo "model is not readable in the container at $effective_model_container_path" >&2
      return 1
    }

  local serve_help
  serve_help="$(docker run --rm --gpus all --entrypoint vllm \
    "${cuda_compat_env[@]}" "$IMAGE" serve --help=all)"
  for flag in --attention-backend --moe-backend --quantization-config --decode-context-parallel-size --nnodes --node-rank --block-size; do
    grep -Fq -- "$flag" <<<"$serve_help" || {
      echo "image does not support required vLLM flag: $flag" >&2
      return 1
    }
  done
  if (( DECODE_CONTEXT_PARALLEL_SIZE > 1 )); then
    for flag in --dcp-comm-backend --dcp-kv-cache-interleave-size; do
      grep -Fq -- "$flag" <<<"$serve_help" || {
        echo "image does not support required DCP flag: $flag" >&2
        return 1
      }
    done
  fi
  for configured_flag in \
    "${HF_OVERRIDES:+--hf-overrides}" \
    "${MAX_CUDAGRAPH_CAPTURE_SIZE:+--max-cudagraph-capture-size}" \
    "${KV_CACHE_MEMORY_BYTES:+--kv-cache-memory-bytes}"; do
    [[ -z "$configured_flag" ]] && continue
    grep -Fq -- "$configured_flag" <<<"$serve_help" || {
      echo "image does not support configured vLLM flag: $configured_flag" >&2
      return 1
    }
  done
  echo "rank=$rank preflight ok host=$host_ip iface=$iface hca=$hca image_arch=$image_arch"
}

start_node() {
  resolve_model_paths
  capture_logs
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  mkdir -p "$CACHE_HOST_PATH" "$LOG_HOST_PATH"
  local uma_utils_overlay
  prepare_uma_memory_overlay
  reclaim_uma_page_cache

  local iface hca
  iface="$(resolve_fabric_iface)"
  hca="$(resolve_hca "$iface")"
  local headless=() extra_mounts=() extra_env=() nccl_tuning_env=() memory_args=()
  [[ "$rank" == "0" ]] || headless=(--headless)
  if [[ -n "$CONTAINER_MEMORY" && "$CONTAINER_MEMORY" != "0" ]]; then
    memory_args=(--memory "$CONTAINER_MEMORY" --memory-swap "$CONTAINER_MEMORY")
  fi

  if [[ "$Q40_ENABLED" == "1" ]]; then
    local receipt_path
    receipt_path="$CACHE_HOST_PATH/jit/q40-exact-state-serving-v1-rank${rank}.json"
    if [[ -f "$receipt_path" ]]; then
      mv "$receipt_path" "${receipt_path}.backup.$(date -u +%Y%m%dT%H%M%SZ)"
    fi
    extra_mounts+=(
      -v "$Q40_HOST_PATH/exl3.py:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/exl3.py:ro"
      -v "$Q40_HOST_PATH/model_runner.py:/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/model_runner.py:ro"
    )
    extra_env+=(
      -e PYTHONPATH=/opt/sparkring-r7-tvm-ffi:/opt/spark-vllm
      -e XDG_CACHE_HOME=/cache/jit
      -e VLLM_CACHE_ROOT=/cache/jit/vllm-q40-exact-state-v1
      -e VLLM_EXL3_EXT_PATH=/opt/exllamav3-python
      -e "SPARK_Q40_EXACT_STATE_ATTEST_PATH=/cache/jit/q40-exact-state-serving-v1-rank${rank}.json"
      -e "SPARK_Q40_EXACT_STATE_EXPECTED_EXL3_SHA256=$Q40_EXL3_SHA256"
      -e "SPARK_Q40_EXACT_STATE_IMAGE_ID=$(docker image inspect "$IMAGE" --format '{{.Id}}')"
      -e "SPARK_Q40_EXACT_STATE_CHECKPOINT=$Q40_CHECKPOINT_REVISION"
      -e LD_PRELOAD=/usr/local/cuda/compat/libcuda.so.1:/opt/sparkring/nccl/libnccl.so.2
      -e VLLM_NCCL_SO_PATH=/opt/sparkring/nccl/libnccl.so.2
    )
  fi
  if [[ "$VLLM_UMA_USE_MEM_AVAILABLE" == "1" ]]; then
    extra_mounts+=(
      -v "$uma_utils_overlay:/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/utils.py:ro"
    )
    extra_env+=(-e VLLM_UMA_USE_MEM_AVAILABLE=1)
  fi
  [[ -z "$NCCL_ALGO" ]] || nccl_tuning_env+=(-e "NCCL_ALGO=$NCCL_ALGO")
  [[ -z "$NCCL_MIN_NCHANNELS" ]] || nccl_tuning_env+=(-e "NCCL_MIN_NCHANNELS=$NCCL_MIN_NCHANNELS")
  [[ -z "$NCCL_MAX_NCHANNELS" ]] || nccl_tuning_env+=(-e "NCCL_MAX_NCHANNELS=$NCCL_MAX_NCHANNELS")
  [[ -z "$NCCL_IB_ADDR_RANGE" ]] || nccl_tuning_env+=(-e "NCCL_IB_ADDR_RANGE=$NCCL_IB_ADDR_RANGE")

  local serve_args=(
    "$effective_model_container_path"
    --served-model-name "$SERVED_MODEL_NAME"
    --host 0.0.0.0
    --port "$API_PORT"
    --trust-remote-code
    --tensor-parallel-size 4
    --decode-context-parallel-size "$DECODE_CONTEXT_PARALLEL_SIZE"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --block-size "$BLOCK_SIZE"
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
  [[ -z "$DCP_COMM_BACKEND" ]] || serve_args+=(--dcp-comm-backend "$DCP_COMM_BACKEND")
  [[ -z "$DCP_KV_CACHE_INTERLEAVE_SIZE" ]] || serve_args+=(--dcp-kv-cache-interleave-size "$DCP_KV_CACHE_INTERLEAVE_SIZE")
  [[ -z "$HF_OVERRIDES" ]] || serve_args+=(--hf-overrides "$HF_OVERRIDES")
  [[ "$ONLINE_QUANT" == "none" ]] || serve_args+=(--quantization-config "$ONLINE_QUANT_CONFIG")
  [[ "$ENFORCE_EAGER" == "1" ]] && serve_args+=(--enforce-eager)
  [[ -z "$COMPILATION_CONFIG" ]] || serve_args+=(--compilation-config "$COMPILATION_CONFIG")
  [[ -z "$MAX_CUDAGRAPH_CAPTURE_SIZE" ]] || serve_args+=(--max-cudagraph-capture-size "$MAX_CUDAGRAPH_CAPTURE_SIZE")
  [[ -z "${KV_CACHE_MEMORY_BYTES:-}" ]] || serve_args+=(--kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES")
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
    if [[ "$RUNTIME_CONTRACT" == "sparkring-r7-switch-q40" ]]; then
      serve_args+=(--speculative-config "{\"model\":\"$effective_model_container_path\",\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_SPECULATIVE_TOKENS,\"draft_tensor_parallel_size\":$MTP_DRAFT_TP_SIZE,\"quantization\":\"exl3\",\"moe_backend\":\"$MTP_MOE_BACKEND\",\"attention_backend\":\"$ATTENTION_BACKEND\",\"use_local_argmax_reduction\":$local_argmax,\"draft_sample_method\":\"$MTP_DRAFT_SAMPLE_METHOD\"}")
    else
      serve_args+=(--speculative-config "{\"model\":\"$effective_model_container_path\",\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_SPECULATIVE_TOKENS,\"draft_tensor_parallel_size\":$MTP_DRAFT_TP_SIZE,\"moe_backend\":\"$MTP_MOE_BACKEND\",\"use_local_argmax_reduction\":$local_argmax,\"draft_sample_method\":\"$MTP_DRAFT_SAMPLE_METHOD\",\"rejection_sample_method\":\"$MTP_REJECTION_SAMPLE_METHOD\"}")
    fi
  fi
  serve_args+=("${headless[@]}")
  if [[ "$RUNTIME_CONTRACT" == "sparkring-r7-switch-q40" ]]; then
    serve_args=(serve "${serve_args[@]}")
  fi

  docker run --gpus all -d \
    --name "$CONTAINER_NAME" --restart no --init \
    --network host --ipc host --shm-size "$CONTAINER_SHM_SIZE" \
    "${memory_args[@]}" \
    --ulimit memlock=-1:-1 --ulimit "nofile=$CONTAINER_NOFILE:$CONTAINER_NOFILE" \
    --cap-add IPC_LOCK --device /dev/infiniband:/dev/infiniband \
    -v "$effective_model_mount_host_path:$effective_model_mount_container_path:ro" \
    -v "$CACHE_HOST_PATH:/cache" \
    "${extra_mounts[@]}" \
    -e "VLLM_HOST_IP=$host_ip" \
    -e "VLLM_WORKER_MULTIPROC_METHOD=$VLLM_WORKER_MULTIPROC_METHOD" \
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
    -e B12X_DENSE_SPLITK_TURBO=1 -e B12X_W4A16_TC_DECODE=1 \
    -e B12X_W4A8_TINY_DECODE=1 -e MOE_MODE=a16 \
    -e B12X_MOE_FORCE_A8=0 -e B12X_MOE_FORCE_A16=1 \
    -e B12X_MLA_SM120_UNIFIED=1 -e "B12X_PCIE_DMA=$B12X_PCIE_DMA" \
    -e VLLM_USE_B12X_PCIE_DMA=0 -e VLLM_ENABLE_PCIE_ALLREDUCE=0 \
    -e VLLM_PCIE_DMA_FP8=0 -e B12X_PCIE_DMA_FP8=0 \
    -e VLLM_DCP_GLOBAL_TOPK=1 -e VLLM_DCP_SHARD_DRAFT=1 \
    -e "VLLM_USE_B12X_DCP_A2A=$VLLM_USE_B12X_DCP_A2A" \
    -e "VLLM_SPARK_MAX_QUERY_ROWS=$VLLM_SPARK_MAX_QUERY_ROWS" \
    -e "VLLM_SPARK_MTP_MODE_ID=$VLLM_SPARK_MTP_MODE_ID" \
    -e "VLLM_SPARK_MTP_TOKENS=$VLLM_SPARK_MTP_TOKENS" \
    -e "VLLM_SPARK_SHARED_CAPTURE_STREAM=$VLLM_SPARK_SHARED_CAPTURE_STREAM" \
    -e "KV_FP8_ROPE=$KV_FP8_ROPE" \
    -e "VLLM_NVFP4_MLA_DYNAMIC_SCALE=$VLLM_NVFP4_MLA_DYNAMIC_SCALE" \
    -e "VLLM_B12X_MLA_CKV_GATHER=$VLLM_B12X_MLA_CKV_GATHER" \
    -e "VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS=$VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS" \
    -e VLLM_EXL3_ONLINE_TRELLIS_BITS=6 \
    -e VLLM_EXL3_EXT_PATH=/opt/exllamav3 \
    -e VLLM_EXL3_ENCODER_SOURCE=/opt/exllamav3-python/exllamav3 \
    -e VLLM_EXL3_ONLINE_CACHE_DIR=/cache/exl3-online \
    -e VLLM_EXL3_ONLINE_CACHE_MODE=readwrite \
    -e "ONLINE_QUANT=$ONLINE_QUANT" \
    -e "VLLM_EXL3_PREFILL_CAPACITY=$VLLM_EXL3_PREFILL_CAPACITY" \
    -e VLLM_ENGINE_READY_TIMEOUT_S=7200 -e TORCHINDUCTOR_COMPILE_THREADS=1 \
    -e "OMP_NUM_THREADS=$CPU_THREADS_PER_PROCESS" \
    -e "OPENBLAS_NUM_THREADS=$CPU_THREADS_PER_PROCESS" \
    -e "MKL_NUM_THREADS=$CPU_THREADS_PER_PROCESS" \
    -e "NUMEXPR_NUM_THREADS=$CPU_THREADS_PER_PROCESS" \
    -e "RAYON_NUM_THREADS=$CPU_THREADS_PER_PROCESS" \
    -e TOKENIZERS_PARALLELISM=false -e MALLOC_ARENA_MAX=2 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e SAFETENSORS_FAST_GPU=1 -e "INSTANTTENSOR_BACKEND=$INSTANTTENSOR_BACKEND" \
    -e "INSTANTTENSOR_COPY=$INSTANTTENSOR_COPY" \
    -e "INSTANTTENSOR_BUFFER_SIZE=$INSTANTTENSOR_BUFFER_SIZE" \
    -e "INSTANTTENSOR_CONCURRENCY=$INSTANTTENSOR_CONCURRENCY" \
    -e "INSTANTTENSOR_IO_DEPTH=$INSTANTTENSOR_IO_DEPTH" \
    -e "INSTANTTENSOR_CHUNK_SIZE=$INSTANTTENSOR_CHUNK_SIZE" \
    -e "INSTANTTENSOR_MAX_FREE_MEM_USAGE=$INSTANTTENSOR_MAX_FREE_MEM_USAGE" \
    -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e "NCCL_IB_HCA=$hca" \
    -e "NCCL_IB_GID_INDEX=$NCCL_IB_GID_INDEX" -e NCCL_IB_ROCE_VERSION_NUM=2 \
    -e NCCL_IB_ADDR_FAMILY=AF_INET \
    -e "NCCL_SOCKET_IFNAME=$iface" -e "GLOO_SOCKET_IFNAME=$iface" \
    -e "TP_SOCKET_IFNAME=$iface" -e "MN_IF_NAME=$iface" \
    -e NCCL_NVLS_ENABLE=0 -e "NCCL_CROSS_NIC=$NCCL_CROSS_NIC" \
    -e "NCCL_IB_MERGE_NICS=$NCCL_IB_MERGE_NICS" \
    -e "NCCL_IB_SUBNET_AWARE_ROUTING=$NCCL_IB_SUBNET_AWARE_ROUTING" \
    -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 \
    -e "NCCL_DEBUG=$NCCL_DEBUG" -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    -e XDG_CACHE_HOME=/cache -e VLLM_CACHE_ROOT=/cache/vllm \
    -e TRITON_CACHE_DIR=/cache/triton -e TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor \
    -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions \
    -e FLASHINFER_WORKSPACE_BASE=/cache/flashinfer \
    -e CUTE_DSL_CACHE_DIR=/cache/cute-dsl -e B12X_CUTE_COMPILE_CACHE_DIR=/cache/b12x-cute \
    "${extra_env[@]}" "${nccl_tuning_env[@]}" \
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
  if [[ "$Q40_ENABLED" == "1" ]]; then
    local receipt_path image_id
    receipt_path="$CACHE_HOST_PATH/jit/q40-exact-state-serving-v1-rank${rank}.json"
    image_id="$(docker image inspect "$IMAGE" --format '{{.Id}}')"
    python3 - "$receipt_path" "$rank" "$image_id" "$Q40_CHECKPOINT_REVISION" "$Q40_EXL3_SHA256" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"Q40 runtime attestation is missing: {path}")
receipt = json.loads(path.read_text())
expected = {
    "schema": "sparkring-q40-exact-state-runtime-attestation/v1",
    "status": "live-runtime-attested",
    "scope": "target-mixed-exact-q40-only",
    "dcp_rank": int(sys.argv[2]),
    "image_id": sys.argv[3],
    "checkpoint_revision": sys.argv[4],
}
for key, value in expected.items():
    if receipt.get(key) != value:
        raise SystemExit(f"Q40 receipt {key} mismatch: expected {value!r}, got {receipt.get(key)!r}")
if receipt.get("sources", {}).get("exl3", {}).get("sha256") != sys.argv[5]:
    raise SystemExit("Q40 receipt EXL3 hash mismatch")
if set(receipt.get("gates", {}).values()) != {"pass"}:
    raise SystemExit("one or more Q40 runtime gates did not pass")
PY
  fi
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
