# Karmic Kraken serving migration — 2026-09-25

## Pinned source and scope

| Source | Checked head | Role |
| --- | --- | --- |
| [vLLM `dev/karmic-kraken`](https://github.com/local-inference-lab/vllm/commit/1794dcf18454900263e0c66711af8ea4a1283ac1) | `1794dcf18454900263e0c66711af8ea4a1283ac1` | New serving base and image revision. |
| [vLLM `dev/jovian-judgement`](https://github.com/local-inference-lab/vllm/commit/8e1f1e587f8d24faf606f334a1c4bdaaa6bd4368) | `8e1f1e587f8d24faf606f334a1c4bdaaa6bd4368` | Required legacy-head comparison; no JJ source is imported. The previous image pin was `5bca5a58d970216bd46be82575e824c6e424c465`. |
| [b12x `master`](https://github.com/local-inference-lab/b12x/commit/a7d7d29b2ef8869086e0ceaa787321f17544e3c9) | `a7d7d29b2ef8869086e0ceaa787321f17544e3c9` | Kernel and RoCEnante runtime. Previous pin was `92cd3800932539c947c9a8e06123fe5f36c9eae4`. |

Rechecked these three upstream heads on 2026-09-26; they still match the pins.
Upstream has not superseded the preparation diagnostics or resolved the
CUDA-free-memory race budget on unified memory.

The new [Dockerfile](Dockerfile.karmic) copies only the local RoCE transport
patch and its native probe. Karmic/b12x already contain the prior RoCE dtype
name fix, attached program metadata and world-coordinated collective
preparation, so the older local patches for those are superseded. The b12x
proxy still had the same four optional opportunities: inline tiny payloads,
rotated peer posting, skip empty send-CQ polls, and initialize protocol bytes
without touching payload bytes. [The rebased patch](patches/roce_karmic.py) is
guarded by hashes of both exact upstream inputs and outputs. All four switches
remain independent through `roce_optimizations` in the fleet config.

The upstream RoCE audit also found the CPU/Gloo setup path that saves about
3.4 GiB of unified memory per rank, the fail-stop health contract, dual-rail
striping, size-gated dispatch and `B12X_ROCE_TRAFFIC_CLASS`. These are present
upstream. No further local RoCE source change was justified without fleet
evidence. The traffic class stays at its upstream default until the switch
and NIC QoS mapping is known. Existing `NCCL_P2P_DISABLE=1` is retained for
this one-GPU-per-node fleet.

Karmic pins CUTLASS DSL 4.7.1 while the current b12x package metadata pins
4.6.2. The image installs b12x without dependency resolution and checks that
Karmic's 4.7.1 stays installed. This is a known upstream compatibility gap;
only the Spark GPU build and four-rank run can establish that the two trees
work together. The image does not alter either project's source dependency
declaration or silently downgrade Karmic.

No graph capture, graph memory profiling, shape profiling, prefill hash,
general tuning, disk Engram, or display-KV source overlay is in this image.
The display-reserve experiment remains in the older Dockerfile and working
tree for separate review.

## Unified memory and KV budget

Karmic's CUDA platform detects integrated GPUs and its `MemorySnapshot`
uses `psutil.virtual_memory().available` on UMA, accounting for reclaimable
host page cache that `cudaMemGetInfo` can miss. Upstream also releases cached
device allocations under high UMA pressure. The recipe limits b12x compiler
concurrency with `B12X_COMPILE_WORKERS=4`, because compiler processes share
physical memory with the GPU. It leaves `--kv-cache-memory-bytes` unset, so
vLLM profiles startup allocations and sizes KV automatically at
`--gpu-memory-utilization 0.88`. The explicit display-KV credit is disabled.

The target limits are **16 sequences**, **1,048,576 tokens per sequence** and
**4,096 batched tokens**. These are admission limits, not a claim that sixteen
simultaneous 1M-token requests fit the available KV pages. This profile uses
DSpark K5 with adaptive verification and lets Karmic choose its graph capture
defaults. Capacity and throughput require measurement on the four Spark nodes.

## Build and start

On Spark 1 after copying this directory and confirming the local model and
cache paths in the fleet config:

```bash
cd ds41-vllm
python3 configure-karmic.py --from-config cluster-r38-c8.json --output fleet.karmic.json
python3 fleet.py --config cluster-r38-c8.json stop
bash build-image.sh
```

Continue only after the build reports `Built spark-vllm-ds41:...`. The image
check links native Engram storage against the CUDA toolkit's driver stub during
the image build. The GPU check after the build loads the real host driver.

```bash
python3 fleet.py --config fleet.karmic.json share
python3 fleet.py --config fleet.karmic.json plan
python3 fleet.py --config fleet.karmic.json preflight
python3 fleet.py --config fleet.karmic.json start
```

While `start` is waiting for the model, use another terminal on Spark 1 to
follow logs for any rank (0–3). Press Ctrl-C to stop following; the serving
container keeps running.

```bash
python3 fleet.py --config fleet.karmic.json logs --rank 0 --follow
```

### Diagnose b12x autotune startup

The pinned b12x preparation session first selects a configuration, then plans
and submits compilation, primes the selected plan, and only races candidates
for requests with multiple configurations. `norm.vision` has one fixed
configuration, so a repeated `selecting norm.vision` line with zero prepared
and zero compiled candidates is before its GPU benchmark. The dashboard shows
only global rank 0 and does not identify the active request within that family.

For the next start, add `"b12x_preparation_trace": true` and
`"b12x_hang_dump": true` to `fleet.karmic.json`. These only enable upstream
diagnostics. The trace writes `job-*.jsonl` and `coordinator-*.jsonl` in the
host cache's `b12x-preparation-trace` directory. To capture a stalled worker's
Python stacks on Spark 1, find its current PID and signal that PID inside the
container:

```bash
docker exec ds41-jj-0 ps -eo pid,cmd | grep 'VLLM::Worker_TP0'
WORKER_PID=$(docker exec ds41-jj-0 ps -eo pid,cmd | awk '$2 == "VLLM::Worker_TP0" { print $1; exit }')
docker exec ds41-jj-0 kill -USR1 "$WORKER_PID"
docker logs ds41-jj-0 --since 2m 2>&1 | tail -n 1000
docker exec ds41-jj-0 sh -c 'tail -n 5 /cache/b12x-preparation-trace/job-*.jsonl'
```

Only send `SIGUSR1` if `b12x_hang_dump` was enabled when the container started.
The stack dump is written to the container log. The trace records request
names and accumulated time in `compile_plan`, `cache_lookup`, and other steps.

The later candidate race uses half of `cuda.mem_get_info().free` as its default
temporary-memory budget. On GB10 that can miss CPU and compiler-worker use of
the same physical memory. Lowering `gpu_memory_utilization` can increase this
race budget if it leaves more CUDA memory free. In the weights stage shown
above, preparation precedes KV memory profiling, so changing the KV utilization
limit may have no effect on that stage at all. The race budget does not affect
the fixed `norm.vision` selection. A smaller race budget still visits every
candidate, but changes the race batches and may change the winning kernel;
compare measured inference latency before adopting a cap.

For a fresh fleet config, use `cluster-karmic-c16.json`. If `fleet.karmic.json`
was generated before the K5 update, set its `draft_tokens` to `5` before
sharing or starting. The migration helper
keeps operator paths, checkpoint revision and node addresses while replacing
the serving limits and removing old experiment controls. `preflight` checks
both exact source revision labels, both RoCE HCAs, GPU imports, disk Engram
checkpoint headers and the local filesystem. `start` also qualifies four-rank
RoCEnante unless `fabric_check` is explicitly changed.

## Validation status

The pinned source was inspected, the rebased patch was applied to its exact
b12x snapshot and checked for repeat application and source-drift rejection.
Local deployment and Karmic patch tests pass. A Linux ARM64 image build, native
verbs/GPU checks, four-node fabric test and 1M serving capacity test remain to
be run on the Spark fleet; this Windows workspace cannot execute those checks.
