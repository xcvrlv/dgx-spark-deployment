# GLM-5.3 EXL3 SparkRun bring-up

This is a small, independent SparkRun recipe for the locally derived
`spark-vllm-glm52-exl3:sparkring-switch-prefill-v2` image and the existing
local model cache. It does not pull an image, download a model, create a fixed
oversized KV cache, or capture prefill graphs. It enables an experimental MTP3
path for up to sixteen sequences; prefill and mixed batches stay eager.

Build the small overlay once on the head Spark, then stream it to the other
three nodes (replace the worker names with the management hosts used by the
cluster):

```bash
bash sparkrun-glm53-exl3/scripts/build-prefill-image.sh host2 host3 host4
```

The build is hash-gated against the exact `sparkring-switch-v1` vLLM source and
fails without modifying the image if that ABI has drifted.

Run the commands from a Spark that can SSH to all four nodes:

```bash
# Install the CLI only if it is not already installed. This does not alter the
# current cluster configuration.
uvx sparkrun setup install

# One-time host and network validation; this is separate from launching the
# model and does not alter the existing switched fabric.
uvx sparkrun setup check

# Confirm SparkRun sees the cache root and render the exact launch first.
sparkrun recipe validate sparkrun-glm53-exl3/recipes/glm53-exl3-4x-safe.yaml
sparkrun show sparkrun-glm53-exl3/recipes/glm53-exl3-4x-safe.yaml
sparkrun run sparkrun-glm53-exl3/recipes/glm53-exl3-4x-safe.yaml \
  --hosts host1,host2,host3,host4 --dry-run

# The first host is the head. This starts in the background; Ctrl-C only
# detaches from log following.
sparkrun run sparkrun-glm53-exl3/recipes/glm53-exl3-4x-safe.yaml \
  --hosts host1,host2,host3,host4
```

Replace `host1` through `host4` with the management hostnames or IPs already
used by the cluster. SparkRun manages the per-node native-vLLM arguments,
container/image availability, RDMA discovery, lifecycle, and logs. The model
path is deliberately the Hugging Face cache root: the command reads `refs/main`
and the complete root is mounted, so snapshot-to-blob symlinks work without
another copy of the weights.

Before page-cache reclamation, the initialized worker reported only 101.49 GiB
free out of 121.63 GiB. Immediately clearing the cluster page cache raised the
same startup reading to about 112.9 GiB. The current 0.89 startup target is
108.25 GiB, leaving about 4.65 GiB of admission margin after a clean start.
`gpu_memory_utilization` is an allocation budget, not a complete host-RAM
limit: model loading, graph capture,
communication buffers, Python processes, and page cache can consume additional
memory. This recipe fixes the per-node KV pool at 18,000,000,000 bytes. Online
K6 weight savings are deliberately left as additional UMA headroom during the
first correctness and memory A/B instead of being immediately reassigned to
KV. MTP3 verifies four rows per sequence, so
`FULL_DECODE_ONLY` captures `[4,8,12,...,64]` for one through sixteen
uniform speculative-decode requests while leaving prefill and mixed batches
eager. CUDA-graph memory estimation is enabled so KV sizing accounts for graph
headroom during profiling; the explicit KV pool prevents a conservative graph
estimate from silently shrinking the cache. The 32 GiB `/dev/shm` setting is
capacity, not preallocated memory or a container memory limit.

The 4096-token prefill batch is paired with
`VLLM_EXL3_PREFILL_CAPACITY=4096`; increasing only the scheduler limit would
leave the EXL3 scratch path sized for the earlier 1024-token baseline. The
bulk-prefill Trellis planner uses the measured block-size-32 setting and a
128-row chunk. Block size 8 launched but reduced prefill throughput on this
Spark/GLM-5.3 deployment. Block size 16 is unusable because this image lacks
W4A16 register-table key `(256, 1, 32, 2, False)` and fails during the
memory-profile dummy run.

The derived image also carries the sparse-indexer workspace right-sizing from
MiaAI-Lab commit `2022ce5`, enabled by this recipe with
`GLM53_INDEXER_WORKSPACE=rightsize`. At this recipe's 1,048,576-token context,
16-sequence cap, MTP3, and `index_kpool=4`, the stock permanent allocation is
41,943,040 entries (5.156 GiB). The legal per-step bound is 4,194,320 entries
(528.002 MiB), returning an estimated 4.641 GiB per Spark to the allocator.
This does not make indexer kernels faster; it creates UMA/KV/graph headroom and
should make the next 8K-capacity experiment substantially more realistic.
The first launch deliberately retains the already measured 4096/4096 EXL
prefill settings so the memory change is isolated. Confirm all four TP ranks log
`[GLM53_INDEXER_WORKSPACE] builder verified 4194320 entries` before trying
8192/8192 again. Roll back without rebuilding by setting the knob to `stock`.

### MiaAI-Lab PR 77 assessment

Do not replace this image with the Flash image from PR 77. That implementation
loads a uniform-K4 Flash checkpoint into fixed-width stacked expert tensors;
its new CUDA path explicitly accepts only K4 MCG tensors. The full 3.42-bpw
checkpoint has mixed K3/K4 routed experts, BF16 shared experts converted online
to Trellis K6, TP4/DCP4 collectives, and a mixed-bit B12X prefill plan. Our
large-M path already avoids the per-expert FP16 reconstruction fallback that
PR 77 accelerates, so copying its fat-expert kernel would replace a working
mixed one-grid plan with a K4-only side path and provide no demonstrated gain.

PR 77's 7168-token default is also a one-shot result for the 321B Flash model,
not a transferable optimum for the full 755B model. Its stated 8192 ceiling is
caused by that image's Flash indexer shared-memory limit. Our 8192 failure was
instead an earlyoom event while piecewise graphs and the larger EXL scratch
capacity were active. Qualify our next capacity ladder with decode-only graphs:
4096 baseline, 6144, 7168, then 8192 only if the preceding rung is stable.
MTP3's four rows per request remain in the small-M Trellis path
through sixteen sequences by pinning its upper threshold to 64; this prevents
the 52- through 64-row decode graphs from entering the bulk-prefill planner during
capture. The recipe also pins the proven SM121/B12X controls: V2 model runner, B12X sparse
indexer, MTP verification through the sparse decode path, CKV gather disabled,
DCP global top-k with a sharded draft, a 256 MiB sparse-indexer logits bound,
four NCCL channels, and MTP acceptance instrumentation. Async scheduling is
disabled while isolating the repeated worker-exit failure seen during
long-context speculative decode; decode-aware prefill is not enabled.

The explicit `index_topk_pattern` mirrors the checkpoint's 78-entry
`indexer_types` topology. It is retained because this SparkRing/B12X branch
needs the compact schedule metadata for coherent sparse-index reuse even though
the upstream GLM-5.3 config also carries the expanded list.

Online Trellis K6 is enabled for the checkpoint's eligible BF16 linear/shared
expert weights through `ONLINE_QUANT=exl3-b6` and the same quantization ignore
list used by the earlier SparkRing launcher. The mixed K3/K4 routed experts are
loaded directly from the checkpoint and are not re-encoded. A cold launch can
spend more than 15 minutes per rank creating the K6 artifacts; subsequent
launches reuse `.exl3-online-k6` inside the identity-mounted Hugging Face model
cache. Do not delete that directory between benchmarks. Adaptive MTP depths,
decode-aware prefill, and its associated scheduler budgets remain absent.

### Prefill qualification log

| Date | Batched tokens | EXL3 prefill capacity | CUDA graph mode | GPU memory utilization | Result |
| --- | ---: | ---: | --- | ---: | --- |
| 2026-09-01 | 8192 | 8192 | `FULL_AND_PIECEWISE` | 0.89 | Failed before a usable prefill measurement. On the head Spark, available memory fell to 660 MiB (0.53%). At 17:36:14 earlyoom observed 0.69% available memory and 79.98% free swap, then sent SIGTERM to `VLLM::Worker_TP` (PID 2219074). No throughput result; lower capacities remain to be qualified. |

Treat this row as a rejected capacity/graph combination, not as evidence that
8192-token eager or decode-only-graph prefill is unsafe. Change one dimension
at a time in subsequent runs and retain the first capacity which completes the
uncached prefill benchmark without NVIDIA allocation errors or earlyoom action.

Do not copy the v1 `mods/drop-caches` field into this native v2 recipe: the
presence of `mods` selects SparkRun's legacy recipe path. To test whether page
cache is what prevents 0.85 admission, use the supported cluster operation
immediately before one controlled launch:

```bash
# One-time setup if SparkRun has not saved the required sudo permission:
sparkrun setup clear-cache --cluster <cluster-name> --save-sudo

# Controlled A/B launch:
sparkrun setup clear-cache --cluster <cluster-name>
```

Then compare vLLM's `Free memory on device (.../121.63 GiB) on startup` line.
The 0.89 gate needs about 108.25 GiB free. The measured increase from 101.49 to
about 112.9 GiB confirms that reclaimable cache—not the sparse-indexer
workspace—caused the earlier ceiling. Repeat the cache clear before cold
launches; the fixed 18 GB KV pool assumes that clean-start margin.

The recipe intentionally does not set `executor_config.memory_limit`, so
SparkRun does not add Docker's `--memory` cgroup cap. That cap is not a useful
proxy for the dynamically shared GB10 CPU/GPU memory pool and can turn a
recoverable allocation failure into a prematurely killed container. SparkRun's
host setup and lifecycle controls remain responsible for node protection.

### Clean up a failed launch before retrying

SparkRun's native executor is an otherwise-idle `sleep infinity` container;
vLLM itself runs through `docker exec`. If that vLLM command fails after model
loading, the executor can remain alive with its cgroup still charged for model
file cache and shared memory. Stop the SparkRun job before retrying so Docker
removes that executor:

```bash
sparkrun stop glm53-exl3-4x-safe --hosts host1,host2,host3,host4
```

If the job is already failed and the exact executor is still present, first
confirm it has no live vLLM process (`docker top <container>`), then remove
only that named stale container with `docker rm -f <container>`. Do this on
each affected node. Do not use a blanket Docker prune on the DGX Spark.

Confirm the logs report MTP with three speculative tokens, `FULL_DECODE_ONLY`,
capture sizes through 64, and an 18,000,000,000-byte KV pool. Then record KV
capacity, drafted-token acceptance,
and generation throughput at concurrency 1 and 8. Capture `sparkrun logs
glm53-exl3-4x-safe`, `sparkrun status`, host `dmesg`, and `memory.events` on any
failed node. Do not restore the old Q1–Q40 graph capture.

For optional name-based registration, initialize and commit this folder as its
own Git repository, then run `sparkrun registry add <its-git-url>`. The local
file path above is the recommended first launch because it needs no registry
installation.

If no SparkRun cluster has been configured yet, run `sparkrun setup wizard`
interactively before the launch. It can set up the SSH mesh, RDMA detection,
and early-OOM protection, so review its detected networking and sudo changes
instead of accepting them blindly on this existing switched fabric.

### Rank-0 crash capture

The recipe enables Python fatal-signal reporting and reduced CUDA exception
dumps. CUDA evidence is retained under the model root's
`.runtime-diagnostics` directory. The dump excludes global, shared, and local
GPU memory, preventing an exception from writing a model-sized artifact.

On the head Spark, start the host-side signal trace in a second SSH session as
soon as SparkRun creates the executor container:

```bash
bash sparkrun-glm53-exl3/scripts/trace-head-worker-signals.sh
```

The script waits up to 15 minutes for `Worker_TP0_DCP0`, then attaches a
signal-only `strace`. Keep that SSH session open. If SIGTERM or SIGINT is sent
through a normal userspace signal syscall, the trace records its `si_pid` and
`si_uid`. Fatal native signals are also recorded. On failure, collect both
`$HOME/glm53-runtime-diagnostics` from the head and `.runtime-diagnostics`
from each node's model cache before cleanup. An intentional SparkRun stop will
also appear in the signal trace.
