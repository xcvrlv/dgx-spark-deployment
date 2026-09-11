#!/usr/bin/env bash
# ds41-flash-vlspeed-up.sh: DeepSeek-V4.1-Flash on the fleet's four DGX Sparks,
# served by the joe-spark-patches dsv41 stack (vLLM PR #56214 wheel overlay,
# TP=4, Engram rows on NVMe, CUDA graphs), pointed at the fleet's
# MXFP4+FP4-Engram hybrid checkpoint.
#
# This is a wrapper around launch/vlspeed-tp4-4node-up.sh, not a fork: the
# upstream launcher is verbatim in this tree for diffing against upstream.
# Structurally the two are identical; the differences are:
#   - the node config is read from ds4.1/cluster.json (the fleet's four-node
#     definition: hosts, CX0/CX1 addresses, HCAs, GID index, paths, ports) --
#     no RFC 5737 placeholders to edit;
#   - the NCCL socket interface is discovered from each node's CX0 IP, as in
#     ds4.1/cluster.py -- no foreign interface name copied;
#   - fabric identities follow the ds4.1 cluster: NCCL on the f0 HCA pair
#     rocep1s0f0,roceP2p1s0f0, GID 3 (RoCEv2), VLLM_HOST_IP = the node's CX0
#     IP, rendezvous on CX0;
#   - the checkpoint is /model (the hybrid, mounted read-only from
#     model_path), not the official fp8 snapshot;
#   - the image defaults to vlspeed-eng:5 = vlspeed-eng:4 +
#     patch/engram-fp4-disk.py, without which the hybrid dies at weight load
#     (docs/INTEGRATION.md);
#   - the container is vlspeed-tp4, distinct from the ds4.1 stack's
#     ds41-flash-tp4. Co-tenancy fails startup anyway: stop the ds4.1 service
#     before bringing this one up.
#
# Serving defaults are the upstream measured 2026-09-11 serve: CTX=1048576,
# GPU_UTIL=0.78, MAXSEQS=8, MAXBATCH=8192, MOE_BACKEND=b12x, B12X_A16=1,
# DSPARK=5. The ds4.1 cluster.json's 0.80 utilization and 16 sequences are
# tuned for the ds4.1 stack, not measured with this one; override with
# GPU_UTIL/MAXSEQS if wanted. CTX is mapped from cluster.json because it
# agrees with the measured serve.
#
# The row files are built by the serve itself at weight load
# (_stage_disk_shard -> build_row_file), fp4 rows included -- roughly
# 12.2 GiB per layer per rank on the local SSD under TABLE_HOST. Nothing needs
# pre-building; tests/build_real_engram_table_fp4.py can build and verify them
# stand-alone.
#
# Usage:  ds41-flash-vlspeed-up.sh            # bring up
#         DRYRUN=1 ds41-flash-vlspeed-up.sh   # print per-rank scripts, change nothing
#         CONFIG=FILE ds41-flash-vlspeed-up.sh  # alternative settings file
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONFIG="${CONFIG:-$HERE/../../ds4.1/cluster.json}"
[ -f "$CONFIG" ] || { echo ">>> FAIL: node config not found: $CONFIG"; exit 1; }

# --- Resolve the node config (ds4.1/cluster.json) into shell arrays. --------
cfg_out="$(python3 - "$CONFIG" <<'PY'
import json, shlex, sys

c = json.load(open(sys.argv[1]))
if len(c["nodes"]) != 4:
    raise SystemExit(">>> FAIL: the wrapper requires the four-node config")
lines = [
    "MODEL_PATH=" + shlex.quote(c["model_path"]),
    "SERVED=" + shlex.quote(c.get("served_model_name", "DeepSeek-V4.1-Flash")),
    "PORT=" + shlex.quote(str(c["port"])),
    "MASTER_PORT=" + shlex.quote(str(c["master_port"])),
    "CTX=" + shlex.quote(str(c["max_model_len"])),
    "CACHE_PATH=" + shlex.quote(c["cache_path"]),
    "HCAS=" + shlex.quote(",".join(c["hcas"])),
    "GID=" + shlex.quote(str(c["gid_index"])),
    "NODE_HOST=( " + " ".join(shlex.quote(n["host"]) for n in c["nodes"]) + " )",
    "NODE_CX0=( " + " ".join(shlex.quote(n["cx0"]) for n in c["nodes"]) + " )",
    "NODE_CX1=( " + " ".join(shlex.quote(n["cx1"]) for n in c["nodes"]) + " )",
]
print("\n".join(lines))
PY
)" || { echo ">>> FAIL: cannot read node config: $CONFIG"; exit 1; }
mapfile -t CFG <<< "$cfg_out"
for line in "${CFG[@]}"; do eval "$line"; done

IMAGE="${IMAGE:-vlspeed-eng:5}"
NAME="${NAME:-vlspeed-tp4}"
TABLE_HOST="${TABLE_HOST:-$CACHE_PATH/vlspeed-table}"
SNAP="${SNAP:-/model}"

TP="${TP:-4}"
GPU_UTIL="${GPU_UTIL:-0.78}"        # upstream measured; cluster.json says 0.80 for ds4.1
MAXSEQS="${MAXSEQS:-8}"             # upstream measured; cluster.json says 16 for ds4.1
MAXBATCH="${MAXBATCH:-8192}"
EAGER="${EAGER:-0}"                 # 1 = --enforce-eager, skips graph capture
CGMODE="${CGMODE:-FULL_AND_PIECEWISE}"
CG_SIZES="${CG_SIZES-}"             # comma list; empty = derive from DSPARK/MAXSEQS
VERIFY="${VERIFY:-0}"               # N = verify the first N engram lookups, eager only
PRESTAGE="${PRESTAGE:-1}"           # 0 = disable the prestage, for an A/B
DISK_THREADS="${DISK_THREADS:-12}"
DISK_ODIRECT="${DISK_ODIRECT:-true}"
EXTRA="${EXTRA:---enable-auto-tool-choice --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41}"
MOE_BACKEND="${MOE_BACKEND:-b12x}"  # b12x; empty = DeepGEMM
# The docker memory cap is not free at 1M (upstream finding 3). Same rule as
# the upstream launcher: cleared above CTX 262144 unless the caller set it.
MEM_LIMIT_SET="${MEM_LIMIT+1}"
MEM_LIMIT="${MEM_LIMIT-112g}"
[ -n "$MEM_LIMIT_SET" ] || [ "$CTX" -le 262144 ] || MEM_LIMIT=""
OOM_ADJ="${OOM_ADJ:-500}"
POLL_N="${POLL_N:-120}"
DEFRAG="${DEFRAG:-1}"
# DSPARK=k enables V4.1 speculative decoding; k must be a multiple of 5. Five
# is the checkpoint's qualified setup (ds4.1/serve.py enforces it).
DSPARK="${DSPARK:-5}"
# Traffic class and GDR level are the upstream dsv41 fabric tunings. The
# ds4.1 stack's validated fabric env for this cluster sets neither; export
# NCCL_IB_TC= / NCCL_NET_GDR_LEVEL= (empty) to restore the NCCL defaults.
NCCL_IB_TC="${NCCL_IB_TC-104}"
NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL-5}"

SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=8 -i $SSH_KEY"
SSH_USER="${SSH_USER:-${USER}}"
SSH_USER="${SSH_USER:-juho}"

sshto()   { if [ "$1" = 0 ] && [ "$HEAD_LOCAL" = 1 ]; then bash -c "$2"; else ssh $SSH_OPTS "${SSH_USER}@${NODE_HOST[$1]}" "$2"; fi; }
sshpipe() { if [ "$1" = 0 ] && [ "$HEAD_LOCAL" = 1 ]; then bash -c 'cat > /tmp/vlspeed-launch.sh && bash /tmp/vlspeed-launch.sh'; else ssh $SSH_OPTS "${SSH_USER}@${NODE_HOST[$1]}" 'cat > /tmp/vlspeed-launch.sh && bash /tmp/vlspeed-launch.sh'; fi; }

# --- Discover each node's socket interface from its CX0 IP (ds4.1 rule). -----
# Rank 0 runs locally when this box owns the head's CX0 address, as in
# ds4.1/cluster.py; remote ranks are reached over SSH. DRYRUN tolerates a
# failed discovery and prints (undiscovered) instead of aborting.
json_ifs() {
  python3 -c '
import json, sys
for item in json.load(sys.stdin):
    for a in item.get("addr_info", []):
        if a.get("local"):
            print(item["ifname"], a["local"])
'
}

NODE_IFNAME=()
HEAD_LOCAL=0
local_map="$(ip -j -4 addr show 2>/dev/null | json_ifs)"
for i in 0 1 2 3; do
  if printf '%s\n' "$local_map" | awk -v ip="${NODE_CX0[$i]}" '$2 == ip {found=1} END {exit !found}'; then
    NODE_IFNAME[$i]="$(printf '%s\n' "$local_map" | awk -v ip="${NODE_CX0[$i]}" '$2 == ip {print $1; exit}')"
    [ "$i" = 0 ] && HEAD_LOCAL=1
  else
    NODE_IFNAME[$i]="$(sshto "$i" 'ip -j -4 addr show' 2>/dev/null | json_ifs | awk -v ip="${NODE_CX0[$i]}" '$2 == ip {print $1; exit}')" || true
  fi
  if [ -z "${NODE_IFNAME[$i]}" ]; then
    if [ "${DRYRUN:-0}" = 1 ]; then
      NODE_IFNAME[$i]="(undiscovered)"
      [ "$i" = 0 ] && HEAD_LOCAL=0
    else
      echo ">>> FAIL: no interface owns ${NODE_CX0[$i]} on ${NODE_HOST[$i]} (fabric mismatch)"; exit 1
    fi
  fi
done

NCCL_MODE="-e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=$HCAS \
 -e NCCL_IB_GID_INDEX=$GID -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_TC=$NCCL_IB_TC -e NCCL_NET_GDR_LEVEL=$NCCL_NET_GDR_LEVEL \
 -e NCCL_CROSS_NIC=1 -e NCCL_NET_PLUGIN=none -e NCCL_IB_SUBNET_AWARE_ROUTING=1 -e NCCL_IB_MERGE_NICS=0 \
 -e NCCL_CUMEM_ENABLE=0 -e NCCL_WIN_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_IB_TIMEOUT=22 \
 -e NCCL_IB_RETRY_CNT=7 -e NCCL_SOCKET_IFNAME=IFNAME_PLACEHOLDER -e GLOO_SOCKET_IFNAME=IFNAME_PLACEHOLDER \
 -e NCCL_NVLS_ENABLE=0 -e NCCL_DEBUG=${NCCL_DEBUG:-WARN}"

ARCH_ENV="-e CUTE_DSL_ARCH=sm_121a -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
 -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

EAGER_ARG=""; GRAPH_ARG=""; GRAPH_ENV=""
if [ "$EAGER" = 1 ]; then
  EAGER_ARG="--enforce-eager"
else
  if [ -z "$CG_SIZES" ]; then
    if [ -n "$DSPARK" ]; then
      CG_SIZES=$( { seq "$DSPARK" "$DSPARK" $((DSPARK * MAXSEQS));
                    seq $((DSPARK + 1)) $((DSPARK + 1)) $(((DSPARK + 1) * MAXSEQS)); } \
                  | sort -n -u | paste -sd, - )
    else
      CG_SIZES=$(seq 1 "$MAXSEQS" | paste -sd, -)
    fi
  fi
  GRAPH_ARG='--compilation-config "{\"cudagraph_mode\":\"'"$CGMODE"'\",\"cudagraph_capture_sizes\":['"$CG_SIZES"']}"'
  GRAPH_ENV="-e VLLM_USE_BREAKABLE_CUDAGRAPH=1"
fi
VERIFY_ENV="-e VL41_ENGRAM_PRESTAGE_VERIFY=$VERIFY -e VL41_ENGRAM_PRESTAGE=$PRESTAGE"
MOE_ARG=""; [ -n "$MOE_BACKEND" ] && MOE_ARG="--moe-backend $MOE_BACKEND"
MEM_ARG="--oom-score-adj $OOM_ADJ"
[ -n "$MEM_LIMIT" ] && MEM_ARG="--memory $MEM_LIMIT --memory-swap $MEM_LIMIT $MEM_ARG"
B12X_ENV="-e B12X_COMPILE_CACHE_DIR=/cache/b12x/compile -e B12X_ROCE_CACHE_DIR=/cache/b12x/roce"
B12X_A16="${B12X_A16:-1}"
[ -n "$B12X_A16" ] && B12X_ENV="$B12X_ENV -e VLLM_B12X_MOE_FP4_FORCE_A16=$B12X_A16"
SPEC_ARG=""
[ -n "$DSPARK" ] && SPEC_ARG='--speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":'"$DSPARK"',\"enable_adaptive_verification\":false}"'

runscript() {
  local r="$1"
  local idx="$r"
  local ifname="${NODE_IFNAME[$idx]}"
  [ "$r" != 0 ] && hl="--headless"
  local nccl="${NCCL_MODE//IFNAME_PLACEHOLDER/$ifname}"
  cat <<EOF
docker rm -f $NAME >/dev/null 2>&1 || true
mkdir -p $TABLE_HOST
docker run -d --name $NAME --network host --ipc host --shm-size 32g --gpus all \\
  --cap-add IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=1048576:1048576 \\
  --device /dev/infiniband:/dev/infiniband --restart no --init $MEM_ARG \\
  -v $MODEL_PATH:/model:ro \\
  -v $TABLE_HOST:/table \\
  -v $CACHE_PATH:/cache \\
  -v \$HOME/.cache/huggingface:/cache/huggingface \\
  -v $CACHE_PATH/tlcache:/root/.tilelang \\
  -e HF_HOME=/cache/huggingface -e HF_HUB_OFFLINE=1 -e VLLM_CACHE_ROOT=/cache/vllm-cache \\
  -e VLLM_HOST_IP=${NODE_CX0[$idx]} -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \\
  $nccl $ARCH_ENV $GRAPH_ENV $VERIFY_ENV $B12X_ENV \\
  -e VLLM_USE_FLASHINFER_SAMPLER=0 -e MAX_JOBS=2 -e FLASHINFER_NVCC_THREADS=1 \\
  -e TRITON_CACHE_DIR=/cache/triton \\
  --entrypoint /bin/bash $IMAGE -lc '
    exec vllm serve $SNAP \\
      --served-model-name $SERVED \\
      --host 0.0.0.0 --port $PORT \\
      --tensor-parallel-size $TP \\
      --gpu-memory-utilization $GPU_UTIL \\
      --max-model-len $CTX --max-num-seqs $MAXSEQS --max-num-batched-tokens $MAXBATCH \\
      --engram-config "{\\"table_path\\":\\"/table\\",\\"disk_read_threads\\":$DISK_THREADS,\\"disk_direct_io\\":$DISK_ODIRECT}" \\
      $SPEC_ARG \\
      --tokenizer-mode deepseek_v41 \\
      $EAGER_ARG $GRAPH_ARG $MOE_ARG --enable-chunked-prefill \\
      --distributed-executor-backend mp \\
      --nnodes 4 --node-rank $r --master-addr ${NODE_CX0[0]} --master-port $MASTER_PORT $hl $EXTRA
  '
EOF
}

if [ "${DRYRUN:-0}" = 1 ]; then
  echo "### DRYRUN IMAGE=$IMAGE TP=$TP CTX=$CTX UTIL=$GPU_UTIL EAGER=$EAGER CG=$CGMODE[$CG_SIZES] DSPARK=$DSPARK MOE=${MOE_BACKEND:-auto} MEM=${MEM_LIMIT:-none}"
  echo "### config: $CONFIG"
  echo "### model: $MODEL_PATH (mounted read-only at $SNAP)"
  echo "### table: $TABLE_HOST (row files are built by the serve at weight load, fp4 included)"
  for i in 0 1 2 3; do
    echo "### node $i: host=${NODE_HOST[$i]} cx0=${NODE_CX0[$i]} cx1=${NODE_CX1[$i]} ifname=${NODE_IFNAME[$i]} head_local=$([ "$i" = 0 ] && [ "$HEAD_LOCAL" = 1 ] && echo yes || echo no)"
  done
  for r in 0 1 2 3; do echo "===== rank $r -> ${NODE_HOST[$r]} ====="; runscript "$r"; echo; done
  exit 0
fi

echo ">>> [1/3] clearing $NAME on all 4 boxes (image=$IMAGE eager=$EAGER cg=$CGMODE[$CG_SIZES] dspark=$DSPARK ctx=$CTX seqs=$MAXSEQS moe=${MOE_BACKEND:-auto} mem=${MEM_LIMIT:-none})"
for i in 0 1 2 3; do sshto "$i" "docker rm -f $NAME >/dev/null 2>&1 || true"; done
sleep 2
# Do this while the weights are released, or it reclaims nothing. Upstream
# finding 4: b12x weight prep needs contiguous host pages, and the other ranks
# then block with no timeout in _init_message_queues, so the boot reads as hung.
# DEFRAG=0 skips it.
if [ "$DEFRAG" = 1 ]; then
  echo ">>> [1b/3] dropping caches + compacting memory on all 4 boxes"
  for i in 0 1 2 3; do
    sshto "$i" "sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory' 2>/dev/null || echo '    WARN: defrag failed on node $i'"
  done
  sleep 5
fi
echo ">>> [2/3] starting workers (3,2,1) then head (0 -> ${NODE_HOST[0]})"
for r in 3 2 1; do echo "    rank $r -> ${NODE_HOST[$r]}"; runscript "$r" | sshpipe "$r"; sleep 4; done
echo "    rank 0 (head) -> ${NODE_HOST[0]}"; runscript 0 | sshpipe 0
echo ">>> [3/3] polling head:$PORT"
for i in $(seq 1 "$POLL_N"); do
  sleep 15
  if sshto 0 "curl -s --max-time 4 localhost:$PORT/v1/models 2>/dev/null" | grep -q "$SERVED"; then
    echo ">>> OK: $SERVED UP after ~$((i*15))s"; exit 0
  fi
  hs=$(sshto 0 "docker inspect -f '{{.State.Status}}' $NAME 2>/dev/null" 2>/dev/null || true)
  [ "$hs" = exited ] && { echo ">>> FAIL: head exited at ~$((i*15))s"; exit 1; }
done
echo ">>> FAIL: timed out (~$((POLL_N*15/60))min)"; exit 2
