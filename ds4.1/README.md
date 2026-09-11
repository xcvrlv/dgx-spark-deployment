# DeepSeek V4.1 Flash on the four Sparks

This directory prepares a native multi-node vLLM deployment for the existing
MXFP4 + FP4 Engram hybrid at:

```text
/home/juho/.cache/huggingface/hub/DeepSeek-V4.1-Flash-MXFP4-FP4-Engram
```

It uses **TP4 with expert sharding, DCP1, five-token DSpark, 16 sequences,
1,048,576-token maximum context, and breakable decode CUDA graphs**. Engram
tables stay on each host's SSD; only requested rows are staged. The model is
mounted read-only. Nothing downloads or rewrites checkpoint weights.

This is a source-checked candidate. The image build, GPU smoke, full model
loading, TP4 collectives, and throughput have not been run on the cluster here.
`start` requires GPU preflight on every node and checks actual graph-capture logs
after serving becomes healthy. It does not silently substitute eager execution,
smaller context, or resident Engram tables if the requested setup fails.

Local validation completed: seven CPU contract tests passed; the patch applied to
the downloaded pinned sources, passed compilation and an idempotent reapply,
and rejected a deliberately changed source. CUDA kernels cannot be exercised
on the Windows development host; `preflight` performs that check on the Sparks.

## Run on Spark 1

Copy this directory to the head Spark (or update the repository there).
Use the existing passwordless SSH and Docker setup. Review `cluster.json`:
the account defaults to `juho`; all paths are absolute and identical on peers.
The native launcher runs rank 0 locally; execute it on **192.168.0.1**.

```bash
# Print all four commands without contacting the cluster.
python3 ds4.1/cluster.py plan

# Build the pinned ARM64 vLLM sources, install B12x, apply the FP4 disk adapter.
# This is a substantial source build, not an overlay of the old GLM runtime.
python3 ds4.1/cluster.py build

# Stream the resulting image to peers over CX0 and compare image IDs.
python3 ds4.1/cluster.py copy-image

# Optional stand-alone qualification before launch.
python3 ds4.1/cluster.py preflight

# Sync launch files, run preflight, start ranks 3/2/1 then 0, wait, verify.
python3 ds4.1/cluster.py start

python3 ds4.1/cluster.py status
python3 ds4.1/cluster.py logs
python3 ds4.1/cluster.py verify
python3 ds4.1/cluster.py stop
```

For startup failures, collect complete logs **before stopping or recreating
containers** with `python3 ds4.1/collect-logs.py`. Run this on the head Spark;
it saves all four ranks' timestamped logs and container exit/OOM state into a
new local `ds41-logs-*` directory over CX0. Collection continues if a peer fails.
The ordinary `cluster.py logs` command shows only the last 150 lines, which can
omit the original worker exception and leave only the head's cancellation trace.

`start` includes preflight; running it separately first is optional. Leave the
Sparks free for this workload. An occupied API port aborts before starting any
DS41 rank. Existing GLM services are not stopped by this launcher. If a launch
fails, inspect the retained DS41 containers with `logs`, then `stop` before
retrying. Startup/compilation can take a long time; the readiness timeout is two
hours. The endpoint is `http://10.3.10.1:8000/v1` on the management network
(the server binds all interfaces), or `http://192.168.0.1:8000/v1` on CX0.

`--node 0` through `--node 3` restricts an operational action to the local host
and that rank; the fleet launcher normally supplies it remotely. For example,
run `python3 ds4.1/cluster.py logs --node 0` on the head for head logs only.
An alternative settings file can be selected with `--config FILE`.

## Network and runtime choices

| Setting | Initial value / reason |
|---|---|
| SSH, image transfer, rendezvous | `192.168.0.1–4`, CX0 |
| NCCL/RoCEnante HCA pair | `rocep1s0f0,roceP2p1s0f0`, as in GLM v20/v21 |
| RoCE GID | 3; validated as RoCEv2 and matched to each node's CX0/CX1 interfaces |
| Socket interface | Discovered from the node's CX0 IP; no foreign interface name copied |
| NCCL | Existing switched SparkRing library, copied from the local fabric image |
| Small TP collectives | Current B12x RoCEnante, 2 MB all-reduce / 16 MB all-gather bounds |
| NCCL channels | Four, cross-NIC on, merged NICs off, subnet-aware routing on |
| PCIe all-reduce | Disabled for the four separate hosts |
| DSpark | Five drafts, TP4, greedy drafts, standard rejection, adaptive verification |
| Attention | `B12X_MLA_SPARSE_DSV41`; native V4.1 cache layout, no generic NVFP4 KV override |
| Graphs | `VLLM_USE_BREAKABLE_CUDAGRAPH=1`, `FULL_DECODE_ONLY`, capacity through 96 token rows |
| Prefill | Chunked, 4096-token scheduler budget; prefix caching enabled |
| Allocation budget | 0.80 initially, with automatic KV sizing after profiling |

The 1M setting is the maximum **per request**, not a promise that sixteen 1M
requests fit simultaneously. Actual concurrency at that length depends on the
profiled KV capacity, SWA state and transient buffers. The context is explicit
instead of upstream's `auto`, so insufficient memory fails visibly.

The graph buckets preserve depths 1–6 and the Spark launch's step-4 grid through
96 (=16×6). These are **breakable CUDA graphs**, not torch.compile: the branch
sets compilation mode to NONE while retaining graph capture/replay. Disk I/O
and row preparation happen before graph replay; the graph consumes stable BF16
rows. Prefill/mixed batches may remain eager in FULL_DECODE_ONLY mode. Debug
logging is intentional for initial qualification so actual captures can be verified.

SSD io_uring syscalls are blocked by Docker's default seccomp profile. This
candidate uses `seccomp=unconfined` and `IPC_LOCK` for the disk reader; it does
not require a privileged container. The GPU preflight exercises the native
reader in precisely this container configuration. Hosts need both CX links,
the listed HCAs/GIDs, NVIDIA Container Toolkit, Docker, Python 3, rsync and SSH.

## Why there is an FP4 patch

At the pinned source revisions, upstream's disk Engram loader accepts
`float8_e4m3fn[rows,256]` plus `E8M0[rows,8]`. The hybrid has
`uint8[rows,128]` packed E2M1 plus the same-shaped scale plane. Setting
`table_memory=disk` alone would fail the source-shape/dtype checks.

The adapter:

1. Recognizes `engram_dtype=fp4`, block size 32 and `ue8m0` from the checkpoint.
2. Registers 128-byte rows with the existing bounded io_uring reader, preserving
   the tensor byte offsets, TP row ownership and original recomputed scale plane.
3. Decodes E2M1 nibbles directly in the existing lookup kernel and applies the
   original E8M0 scales to produce BF16 output. This is not another quantization
   step. It preserves the values of the downloaded FP4 checkpoint.
4. Retains the existing row masks, scale special cases and reduction path.

The v2 adapter removes v1's intermediate FP8 buffer (24 MiB per Engram layer,
48 MiB per rank at 4096 tokens), its write/read traffic, and one kernel launch
per table preparation. Packed I/O remains 128 weight bytes plus eight scale
bytes per row. The existing BF16 output is still necessary for model computation.
Construction warms the fused kernel before capture and JIT monitoring, using
only a one-token temporary output. The cache transaction retains upstream's stream
synchronization around disk reads and consumption. The patch explicitly rejects
resident FP4 mode because this implementation is for the requested SSD path.

The patcher hashes all source inputs and outputs, checks them before edits,
and supports an idempotent second application and an exact v1-to-v2 upgrade.
It fails on a different ABI.
Source gates are in `patches/source-hashes.json`.

The image is now `spark-vllm-ds41:mx-fp4-engram-v2`. Re-run `build` and
`copy-image`, then `stop`/`start` this service to switch versions. Docker can
reuse the unchanged upstream build layers. No checkpoint download/conversion
is needed. These are structural savings, not measured speedup claims: the
GPU preflight compares fused lookup against the v1 expansion-plus-FP8 path,
including extreme E8M0 scales, before full-model launch.

## Earlier patch assessment

| Earlier work | Applicability here |
|---|---|
| Switched SparkRing NCCL + fabric interface/GID settings | Carried over; final image records the actual local fabric-image ID. Preflight imports Torch with that library loaded. |
| RoCEnante backport and transport tuning | Use current upstream RoCEnante. Do not reapply the old backport or copy old kernel switches without checking current implementations. |
| EXL3 mixed K3/K4, K6 rotations, fused outputs, FC1 tail split, paired FC2 | Not selected: V4.1 uses native MXFP4 `w4a8_mx`, expert sharding, different expert geometry and a different numerical recipe. |
| GLM indexer workspace right-sizing/coalescing | No direct transplant. V4.1 has CSA2 Full/Reindex/Reuse and its own buffers/plans; its top-k buffer is already sized by scheduler token capacity. Profile this path before proposing an equivalent patch. |
| GLM DCP gathers, CKV borrowing/in-place updates | Not applicable to initial DCP1; the CED/global-KV ownership also differs. |
| GLM MTP argmax and feedback changes | Not transplanted into DSpark's Markov/confidence-head path. |
| GLM instanttensor loader overlay | Not used. Lazy safetensors plus file-source descriptors is the inspected SSD Engram path. |
| Repeated host page-cache flusher | Not enabled. Assess startup memory first; do not impose the old GLM load-window flusher on an SSD-backed workload by default. |

## Provenance and validation limits

- vLLM: `dev/jovian-judgement` at `a6571d0a602ed42525a7a77cf960f39ca7f8d7a8`.
  `dev/jovian-justice` did not resolve when inspected on 2026-09-11.
- B12x master: `00b69ac22e21413622c4ecd98f607a2c3e015161`.
- Fabric parent: `spark-vllm-glm52-exl3:sparkring-switch-v1`, already used by GLM.
  Override `FABRIC_IMAGE` for the build if the qualified fabric image has another name.
- Build inputs use upstream's Dockerfile and dependency pins. Container base tags
  and dependency ranges can still change; source commits and patch content are
  pinned, and `copy-image` enforces the same resulting image ID on all nodes.
- Build base image: upstream's default build base
  (`pytorch/manylinux2_28-builder:cuda13.0-*`) is published for linux/amd64
  only, so an arm64 build from it dies with "exec format error" inside the
  `base` stage. The build overrides it with
  `pytorch/manylinuxaarch64-builder:cuda13.0-b8b5f17a7d9ccfc25bbc5cf17b3fcea12964a042`,
  the CUDA-enabled aarch64 builder upstream CI pins for arm64 CUDA image
  builds, and fails visibly if that tag loses its arm64 variant. Override
  `DS41_BUILD_BASE_IMAGE` to select another builder.
- GPU preflight checks every packed byte, signed zero, low-nibble-first order,
  real SSD reads with unaligned offsets, scales, duplicate/missing rows, all four
  TP row partitions, changing active/prepared counts and a graph consumer reading
  fresh outputs. It does not establish full-model or multi-host correctness.
- Model preflight compares config/index hashes and image IDs across hosts; it
  validates the two table headers without reading large tensor payloads.
  Full shard hashes were checked by the download/distribution script.
- Post-start checks require healthy containers without OOM/restarts, actual
  breakable capture logs, FP4 disk-table registration and two HTTP generations.
  They do not establish sustained concurrency-16, 1M-context capacity, DSpark
  acceptance rate or numerical parity against upstream FP8 Engrams.

Sources: [V4.1 launch](https://github.com/local-inference-lab/vllm/blob/a6571d0a602ed42525a7a77cf960f39ca7f8d7a8/serve-ds41-flash.sh),
[V4 Spark launch](https://github.com/local-inference-lab/vllm/blob/a6571d0a602ed42525a7a77cf960f39ca7f8d7a8/serve-ds4-flash-spark.sh),
[Engram loader](https://github.com/local-inference-lab/vllm/blob/a6571d0a602ed42525a7a77cf960f39ca7f8d7a8/vllm/models/deepseek_v4_1/common/engram.py),
[B12x disk-table API](https://github.com/local-inference-lab/b12x/blob/00b69ac22e21413622c4ecd98f607a2c3e015161/b12x/sequence/engram/api.py).
