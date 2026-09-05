# GLM-5.3 EXL3 SparkRun bring-up

This is a small, independent SparkRun recipe for the locally derived
`spark-vllm-glm52-exl3:sparkring-switch-prefill-v2` image and the existing
local model cache. It does not pull an image, download a model, create a fixed
oversized KV cache, or capture prefill graphs. It enables an experimental MTP3
path for up to eight sequences; prefill and mixed batches stay eager.

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

# One-time: install SparkRun's narrowly scoped permission to clear page cache.
# This prompts for sudo during setup, then the load-window flusher can run
# non-interactively without granting the serving container host privilege.
sparkrun setup clear-cache --hosts host1,host2,host3,host4 --save-sudo

# Confirm SparkRun sees the cache root and render the exact launch first.
sparkrun recipe validate sparkrun-glm53-exl3/recipes/glm53-exl3-4x-safe.yaml
sparkrun show sparkrun-glm53-exl3/recipes/glm53-exl3-4x-safe.yaml
bash sparkrun-glm53-exl3/scripts/run-with-cache-flusher.sh \
  sparkrun-glm53-exl3/recipes/glm53-exl3-4x-safe.yaml \
  --hosts host1,host2,host3,host4 --dry-run

# The first host is the head. The wrapper starts and verifies an unconditional
# host cache flusher on every node before allowing SparkRun to create a model
# container. --no-follow returns after dispatch without tying flusher lifecycle
# to an interactive log-following session.
bash sparkrun-glm53-exl3/scripts/run-with-cache-flusher.sh \
  sparkrun-glm53-exl3/recipes/glm53-exl3-4x-safe.yaml \
  --hosts host1,host2,host3,host4 --no-follow
```

Replace `host1` through `host4` with the management hostnames or IPs already
used by the cluster. SparkRun manages the per-node native-vLLM arguments,
container/image availability, RDMA discovery, lifecycle, and logs. The model
path is deliberately the Hugging Face cache root: the command reads `refs/main`
and the complete root is mounted, so snapshot-to-blob symlinks work without
another copy of the weights.

Before page-cache reclamation, the initialized worker reported only 101.49 GiB
free out of 121.63 GiB. Immediately clearing the cluster page cache raised the
same startup reading to about 112.9 GiB. The current 0.91 startup target is
110.68 GiB, leaving about 2.22 GiB of admission margin after a clean start.
`gpu_memory_utilization` is an allocation budget, not a complete host-RAM
limit: model loading, graph capture,
communication buffers, Python processes, and page cache can consume additional
memory. This recipe lets vLLM profile the remaining memory and size the KV pool;
record the resulting `GPU KV cache size` from this exact GLM-5.3 build rather
than copying the 200,064-token number from the older GLM-5.2 QuantTrio recipe.
MTP3 verifies four rows per sequence, so
`FULL_DECODE_ONLY` captures `[4,8,12,...,32]` for one through eight
uniform speculative-decode requests while leaving prefill and mixed batches
eager. CUDA-graph memory estimation is enabled so KV sizing accounts for graph
headroom during profiling. The 32 GiB `/dev/shm` setting is capacity, not
preallocated memory or a container memory limit.

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

### Required load-window cache flusher

Do not copy the v1 `mods/drop-caches` field into this native v2 recipe: the
presence of `mods` selects SparkRun's legacy recipe path. A one-time cache drop
is also insufficient for this configuration because reading model shards can
repopulate the cache before CUDA's allocation and admission phases finish.

The launch wrapper uploads `cache_flusher.sh` to each target user's home,
starts it before SparkRun, and requires every node to report ready. The flusher
does an immediate cache drop and repeats it unconditionally every 60 seconds.
It self-expires after 90 minutes so a forgotten process cannot continue
disrupting normal file-cache behavior indefinitely. Its only privileged action
is `sudo -n tee /proc/sys/vm/drop_caches`, matching SparkRun's scoped sudo rule.
This adapts the load-window procedure from the
[GLM-5.2 QuantTrio deployment](https://github.com/tonyd2wild/GLM-5.2-QuantTrio-200K-4x-DGX-Spark--36tok-s)
without copying that model's KV geometry or measured token count.

Inspect or stop the fleet-wide flusher explicitly with:

```bash
bash sparkrun-glm53-exl3/scripts/cache-flusher-cluster.sh \
  status host1,host2,host3,host4

# Run as soon as the API health check succeeds; do not leave repeated cache
# drops active during normal serving.
bash sparkrun-glm53-exl3/scripts/cache-flusher-cluster.sh \
  stop host1,host2,host3,host4
```

The per-node log is
`~/.cache/sparkrun/glm53-cache-flusher/flusher.log`. Compare vLLM's
`Free memory on device (.../121.63 GiB) on startup` line across all ranks. The
0.91 gate needs about 110.68 GiB free. The measured increase from 101.49 to
about 112.9 GiB confirms that reclaimable cache—not the sparse-indexer
workspace—caused the earlier ceiling.

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
capture sizes through 32, and the profiled `GPU KV cache size`. Then record KV
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

## R22 EXL3 + MTP3 comparison candidate

The new candidate recipe is
`recipes/glm53-exl3-dflash2-4x.yaml`. It combines the newer Local Inference
Lab R22 vLLM/B12X stack with the mixed-K3/K4 EXL3 loader required by
[davidsyoung/GLM-5.3-EXL3-TR3-3.42bpw](https://huggingface.co/davidsyoung/GLM-5.3-EXL3-TR3-3.42bpw).
For the current comparison it uses the checkpoint's native MTP heads with
three speculative tokens. The image retains R22's DFlash2 implementation, but
the recipe deliberately does not load the external draft model.

The image keeps the qualified R22 B12X compute tree and composes only the
later RoCEnante communication directory from b12x commit `1a7e3ec`. That adds
dual-rail RoCE striping and size-aware CUDA-graph collectives without importing
the newer, unrelated sparse-MLA and GLM-next cache changes. Eligible decode
all-reduces and all-gathers use `B12X_ROCENANTE`; larger collectives continue
through the switched-fabric NCCL fallback. Both paths are pinned to the two
active RDMA HCAs present on every node, `rocep1s0f0` and `roceP2p1s0f0`, at
GID index 3.

The published Jovian Judgement R22 container is linux/amd64-only. DGX Spark is
linux/arm64, so `Dockerfile.r22-dflash2` rebuilds the exact R22 vLLM commit and
R22 B12X commit on a digest-pinned ARM64 vLLM nightly. It compiles
the EXL3 extension for `12.1a`/`sm_121a`, verifies all upstream Git trees, and
fails the build if the reviewed EXL3 compatibility composition differs from
the reviewed tree. This is a source-equivalent ARM64 port, not a claim that the
x86 image itself supports Spark. The build copies only the already-qualified
`/opt/sparkring/nccl` switched-fabric runtime from the local
`sparkring-switch-v1` image, so that prerequisite image must remain present on
the head node during the build.

This Hugging Face cache root must already exist on every Spark:

```text
/home/juho/.cache/huggingface/hub/models--davidsyoung--GLM-5.3-EXL3-TR3-3.42bpw
```

Build the image on the head Spark and copy the exact image ID to the other
three nodes:

```bash
bash sparkrun-glm53-exl3/scripts/build-r22-dflash2-image.sh host2 host3 host4
```

Then render and dry-run the launch before starting it through the existing
load-window cache-flusher wrapper:

```bash
sparkrun recipe validate \
  sparkrun-glm53-exl3/recipes/glm53-exl3-dflash2-4x.yaml
sparkrun show sparkrun-glm53-exl3/recipes/glm53-exl3-dflash2-4x.yaml
bash sparkrun-glm53-exl3/scripts/run-with-cache-flusher.sh \
  sparkrun-glm53-exl3/recipes/glm53-exl3-dflash2-4x.yaml \
  --hosts host1,host2,host3,host4 --dry-run
bash sparkrun-glm53-exl3/scripts/run-with-cache-flusher.sh \
  sparkrun-glm53-exl3/recipes/glm53-exl3-dflash2-4x.yaml \
  --hosts host1,host2,host3,host4 --no-follow
```

SparkRun validates and identity-mounts only the target model root. The command
resolves its `refs/main` to the immutable snapshot before starting vLLM. The
target runs TP4/DCP4 with full-CKV B12X prefill; its native MTP draft also runs
TP4. MTP3 produces four verification rows per active request, so decode-only
graphs cover four through 32 rows for the eight-sequence limit. Memory
utilization is 0.91, matching the established MTP comparison recipe.

The first cold launch creates a separate `.exl3-online-k6-r22` cache. Keep it
between runs. Do not reuse the old image's `.exl3-online-k6` directory: encoder,
B12X, and loader identities are part of the cache contract.

The v9 image keeps mixed-Trellis route-pack warmup under PyTorch inference
mode. This matches the lifetime of the persistent route workspace created by
the earlier profile pass and permits its in-place reset during final kernel
warmup.

Its build also supplies the pinned QUTLASS source through `QUTLASS_SRC_DIR`.
This prevents QUTLASS from starting a redundant recursive CUTLASS submodule
clone inside the vLLM wheel build; it uses vLLM's already-resolved CUTLASS
headers instead.

This comparison recipe uses vLLM's ordinary safetensors loader. InstantTensor's
zero-copy ring-buffer mode was removed after it segfaulted before loading the
first tensor on the four-Spark mixed-EXL3 path. The recipe also leaves `--dtype`
unset, so vLLM's default `auto` mode follows the checkpoint configuration; the
checkpoint's non-EXL3 carrier tensors and activations remain BF16.

`PYTHONPATH=/opt/exllamav3` is intentional for this composition. vLLM can use
`VLLM_EXL3_EXT_PATH` to load its private extension handle, but R22 B12X imports
`exllamav3_ext` by module name when resolving the `had_r_128` rotations for
online K6/Trellis warmup. The recipe checks that symbol before starting vLLM.

Before promoting this candidate, require all of the following:

- all four nodes report the image ID printed by the build script;
- startup logs identify the V2 runner, EXL3 target, native MTP speculator,
  three speculative tokens, TP4/DCP4, full-CKV gather, decode-only graphs, and
  `B12X_ROCENANTE` in the TP communicator's backend list;
- API health, finite-logprob, reasoning, and tool-call smoke tests pass;
- a long-context generation and a sustained concurrency-8 run complete without
  worker exit, earlyoom action, CUDA allocation failure, or NCCL timeout;
- local measurements isolate the R22/EXL3 prefill and runtime changes against
  the established MTP3 recipe under the same host state.

[R22 and its B12X stack](https://github.com/local-inference-lab/rtx6kpro/blob/master/models/glm-5.3-flash.md)
have published qualification evidence for the newer execution paths, but not
for this exact mixed-EXL3 model on four GB10 nodes. Treat those results as
upstream evidence and keep the established MTP3 recipe as the rollback path.
