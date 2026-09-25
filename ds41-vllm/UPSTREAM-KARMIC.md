# Karmic Kraken serving migration — 2026-09-25

## Pinned source and scope

| Source | Checked head | Role |
| --- | --- | --- |
| [vLLM `dev/karmic-kraken`](https://github.com/local-inference-lab/vllm/commit/1794dcf18454900263e0c66711af8ea4a1283ac1) | `1794dcf18454900263e0c66711af8ea4a1283ac1` | New serving base and image revision. |
| [vLLM `dev/jovian-judgement`](https://github.com/local-inference-lab/vllm/commit/8e1f1e587f8d24faf606f334a1c4bdaaa6bd4368) | `8e1f1e587f8d24faf606f334a1c4bdaaa6bd4368` | Required legacy-head comparison; no JJ source is imported. The previous image pin was `5bca5a58d970216bd46be82575e824c6e424c465`. |
| [b12x `master`](https://github.com/local-inference-lab/b12x/commit/a7d7d29b2ef8869086e0ceaa787321f17544e3c9) | `a7d7d29b2ef8869086e0ceaa787321f17544e3c9` | Kernel and RoCEnante runtime. Previous pin was `92cd3800932539c947c9a8e06123fe5f36c9eae4`. |

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
simultaneous 1M-token requests fit the available KV pages. This profile starts
without speculative drafting and lets Karmic choose its graph capture defaults.
Capacity and throughput require measurement on the four Spark nodes.

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

For a fresh fleet config, use `cluster-karmic-c16.json`. The migration helper
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
