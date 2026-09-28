# DS41 performance bundle — 2026-09-28

One image contains the latest native upstream, all retained local RoCE options,
the DSpark prefill and bounded-tuning overlays, and the applicable new ports.
Configuration switches select experiments without rebuilding. This work was
prepared locally: no Spark was contacted, no service was changed, and no fleet
speedup has been measured. The operator builds and restarts after the current
load finishes.

## Checked revisions

Live GitHub API checks are recorded in [performance-upstream.json](performance-upstream.json).
The [final recheck at 11:21 UTC](performance-final-heads.json) confirmed all five
heads were unchanged and both build pins match their latest upstream heads.

| Repository / branch | Checked revision | Disposition |
| --- | --- | --- |
| [vLLM Karmic Kraken](https://github.com/local-inference-lab/vllm/commit/502d6cb5acd2ba2a62ecf58497be558c9d86089f) | `502d6cb5acd2ba2a62ecf58497be558c9d86089f` | New image pin, from `1794dcf`. |
| [vLLM Jovian Judgement](https://github.com/local-inference-lab/vllm/commit/8e1f1e587f8d24faf606f334a1c4bdaaa6bd4368) | `8e1f1e587f8d24faf606f334a1c4bdaaa6bd4368` | Required legacy-head comparison; unchanged. |
| [b12x](https://github.com/local-inference-lab/b12x/commit/d44247b6171f7c2f9787341ae884b537887d7df9) | `d44247b6171f7c2f9787341ae884b537887d7df9` | New image pin, from `a7d7d29`. |
| [SparkRing one-command-installer](https://github.com/FujitsuPolycom/sparkring/tree/8b152d65c701f557f62ae6a9a1c3db90771c1df3) | `8b152d65c701f557f62ae6a9a1c3db90771c1df3` | Native Karmic profile and source-fix comparison. |
| [knapcio DGX Spark TP4](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/tree/e9ec61d276c467c747777ed8b9671dac80455954) | `e9ec61d276c467c747777ed8b9671dac80455954` | Decode implementation comparison; SGLang-specific paths assessed separately. |

The Karmic update contains 36 commits, mainly Kimi work. The V4.1 model files
used by the existing overlays are unchanged. The relevant warmup drift is two
typed local declarations; the bounded-tuning guard admits the new exact hash.
The two b12x commits primarily change PCIe collectives for uneven groups.
TP4 on separate Sparks retains the same native RoCE inputs. The H2D staging,
prefix-hash bound and automatic-GC safeguard remain absent from upstream.

Native dtype/program-metadata fixes and coordinated RoCE preparation already
supersede the older local equivalents. Keep the native implementations. The
original checkpoint revision `fb2764a5cf321eaa5070ca8f9e892818f477c16d`, disk
Engram layout, switched two-rail topology and driver/toolchain recipe remain
the deployment inputs. SparkRing's checkpoint and CUDA/NCCL combination differ,
so its rates are evidence for candidates, not a forecast for this fleet.

## Decode findings

SparkRing's applicable native recipe is its
[current TP4 profile](https://github.com/FujitsuPolycom/sparkring/blob/8b152d65c701f557f62ae6a9a1c3db90771c1df3/profiles/deepseek-v41-flash-tp4/config.json)
and [temperature-1 serving record](https://github.com/FujitsuPolycom/sparkring/blob/8b152d65c701f557f62ae6a9a1c3db90771c1df3/performance/records/images/dev-20260927-h2dstaging-deepseek-v41-tp4-20260927.md).
Its probabilistic draft plus block-verification runs show a substantial
concurrency-dependent improvement over greedy drafting. Its two NVFP4 draft
vocabulary projections are another smaller decode candidate. Both are already
implemented upstream; the launcher now exposes them independently. Target
weights retain their original precision. Draft quantization still needs
acceptance and quality checks on our workload.

The [knapcio adapter inventory](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/blob/e9ec61d276c467c747777ed8b9671dac80455954/docs/adapters.md)
explains several gains that should not be counted as missing native kernels:

| Improvement in that stack | Applicability to this image |
| --- | --- |
| TP4 / EP1 native b12x MoE, including 576-wide expert shards | Already selected by our `b12x` backend; no FlashInfer-CUTLASS fallback needs replacing. Tile sizes belong to native preparation/tuning. |
| L2 weight prefetch at collective/attention windows | Already on by default for SM121 in native V4.1. `l2_prefetch` exposes rollback; no second prefetch stream is installed. Native fill budgets differ from their 6 MB learned plans. |
| Engram WKV projection sharding | Native `EngramConfig.projection_tp` exists. Exposed as an experimental switch, off in both supplied presets pending rank-level precision and serving checks. |
| Two MoE barrier fills combined | New independent implementation for the pinned native arena. It covers the dynamic and micro launch sites, verifies same int32 storage and the exact next 16-byte alignment boundary, and otherwise retains two fills. It only clears barriers and unused alignment padding. |
| Block rejection and confidence-limited verification | Native block rejection and adaptive verification exist. The launcher exposes both plus the native cost scale. Native verification actually shortens rows; their router-anchor reuse inside a fixed-width verifier should not be stacked on it. |
| FP8 WO-A twins, dropping duplicate BF16 weights, fused mHC prefill | Native V4.1 already uses prepared b12x projection/mHC paths and packed source-weight release. Their alternate SGLang operators are not drop-in replacements. |
| Attention Q/KV and draft main-projection splits / compact column gather | Potential further bandwidth work, but native merged-projection loading and gather ordering differ. A class swap would interleave Q/KV shards incorrectly. No unvalidated SGLang import hook is installed. |
| Deterministic MoE planner admission, forced small-row/64-row plans | Native prepared-plan selection has its own contracts. Their renamed `b12x_next` planner changes are not applied wholesale; bounded tuning is available to measure native candidates. |
| Draft temperature 0.7 | Their sampler and rejection share a separate draft-temperature tensor. Native rejection consumes raw cached logits with request temperature. A sampler-only edit would use the wrong proposal probability; no such edit is included. |
| Removing broadcasts / folded sampling / eager glue / D2H fence | Native DSpark already samples with position/seed-based Gumbel keys and captures draft sampling in its graph. The named hooks operate on SGLang scheduler objects and broadcasts absent from this path. |
| 4 GiB set-associative Engram cache / fast sliced loader | Different row-store and loader bindings; a fixed extra cache changes shared-memory/KV headroom. Preserve native io_uring and the original checkpoint format. |
| Ring-specific RoCE, patched NCCL, display-reserve memory | Different transport or host setup. Preserve our existing qualified topology and independent RoCE switches. |

No SGLang adapter source is copied into this image. Attribution and the
SparkRing Apache license are included under `/opt/ds41/licenses`.

## Prefill findings and measurement limits

SparkRing reports substantially better cold prefill when Engram overlap is
disabled; the native flag was missing from our launcher. The candidate selects
`engram_overlap: false`. This needs both decode and varied-text prefill checks:
a prefill improvement alone does not establish the best decode setting.

The knapcio README explicitly distinguishes repeated filler from varied-text
prefill. A unique prefix prevents KV prefix hits but does not prevent repeated
Engram lookups. Its filler results are not used to justify batch size, capacity
or a speedup here. Its sequence-parallel/indexer patches target different
SGLang metadata and kernels. Native V4.1 already chunks its indexer work; no
large-logit-buffer backport is needed for that path.

The native SparkRing selection-cache fix addresses its custom sharded-hook
cache, whereas the pinned b12x lookup already retains the correct key. Its
draft-warmup shape shim targets tile specialization that our fixed-256-block
prepare-input kernel does not have. Both are superseded/inapplicable here.

The existing optional intermediate-prefill draft skip, 128-row draft-context
graph, bounded prefix-hash copies and 8192-token target graph are included in
this image. Each has its own switch. The larger batch is never silently chosen;
read [the measured capacity tradeoff](AUTOTUNE-SPARK.md) before selecting it.

## Switches and presets

Image: `spark-vllm-ds41:kk-502d6cb-b12x-d44247b-performance-v1`.
`cluster-karmic-c16.json` advances the pins while keeping the previous serving
choices. The two `cluster-performance-*-c16.json` files demonstrate a same-image
comparison. For the actual deployment, migrate its real operator JSON so paths,
hosts, batch size, utilization, context limit and resident-scale choice survive.

| JSON setting | Candidate | Same-image control / rollback |
| --- | --- | --- |
| `draft_sample_method` | `"probabilistic"` | `"greedy"` |
| `rejection_sample_method` | `"block"` | `"standard"` |
| `enable_adaptive_verification` | `true` | `true`; set `false` for a fixed verifier |
| `adaptive_verification_cost_scale` | Omitted: native `1.0` | Optional positive finite value; only with adaptive DSpark |
| `dspark_markov_nvfp4` | `true` | `false` |
| `dspark_draft_nvfp4_head` | `true` | `false` |
| `decode_graph_policy` | `"exact"`: each request count × every width 1..K+1 | `"upstream"` |
| `engram_overlap` | `false` | `true` |
| `l2_prefetch` | `true`, native SM121 default | `false` to isolate prefetch; supplied control keeps native `true` |
| `engram_projection_tp` | `false` | Experimental `true`, otherwise `false` |
| `moe_coalesce_barriers` | `true` | `false` |
| `h2d_staging` | `true` | `false` |
| `b12x_defer_gc` | `true` | `false`; affects tuning only |
| `bounded_prefix_hashes` | `true` | `false` |
| `dspark_skip_prefill_draft` | `true` | `false` |
| `dspark_compact_context_graph` | `true` | `false` |
| `prefill_8192_graph` | Only for an explicitly selected or preserved 8192 batch | `false` |
| `b12x_autotune` / `b12x_bounded_autotune` | Preserve existing selection; `--autotune` opts into bounded racing | `--no-autotune` disables racing |
| `shm_busy_loop_s` | Omitted: `1` second | Optional finite 0..1; lower values are experimental, not an assumed speedup |
| `engram_resident_scales` | Preserve operator choice | Optional exact-byte resident scales, about 1.43 GiB/rank |

Existing `roce_optimizations` switches are preserved in both presets. Runtime
choices are startup choices; restart all ranks together after editing them.
More graphs and probabilistic draft logits can consume additional memory even
at unchanged utilization. Confirm KV capacity and physical headroom. No display
reserve is added to this image.

`h2d_staging` gives each queued host-to-device copy its own pinned snapshot,
preventing the async scheduler from rewriting data still in flight.
`b12x_defer_gc` defers automatic cyclic collection only while a tuning stream
is held, with nested/concurrent restoration. Explicit `gc.collect()` and
reference-counted destruction are not intercepted. The gate releases queued
work before restoring GC, including on enqueue errors.

## Manual build and rollout

These commands are for the operator on an idle Spark after the active load
finishes. Build GPU checks and qualification consume GPU resources. They have
not been run from this task.

```bash
cd ds41-vllm
# Use the JSON that launched the real service, not an inferred template.
python3 performance-settings.py --from-config fleet.actual.json --output fleet.performance.json
python3 performance-settings.py --from-config fleet.actual.json --output fleet.control.json --control
# Optional larger batch and bounded tuning, in another recipe:
python3 performance-settings.py --from-config fleet.actual.json --output fleet.performance-8192.json --batch-tokens 8192 --autotune

bash build-image.sh
source versions.env
docker run --rm --entrypoint python3 "$IMAGE" /opt/ds41/performance-check.py
timeout 180 docker run --rm --gpus all --entrypoint python3 "$IMAGE" \
  -m pytest -q /opt/ds41/dspark-prefill-tests/test_performance_safety.py
timeout 900 docker run --rm --gpus all --entrypoint python3 "$IMAGE" \
  -m pytest -q /opt/ds41/dspark-prefill-tests/test_native_ced.py \
  /opt/ds41/dspark-prefill-tests/test_native_context.py

# Distribution and fleet actions contact all nodes over SSH.
# Perform them manually when the fleet is idle and restart is authorized.
python3 fleet.py --config fleet.performance.json share
python3 fleet.py --config fleet.performance.json plan
python3 fleet.py --config fleet.performance.json preflight
python3 fleet.py --config fleet.performance.json start
```

Use the existing manual service procedure to stop the running deployment before
`start`. It does not replace a running container automatically. The launcher
requires matching image IDs, pins, bundle label, the operator checkout's
manifest digest and all 26 final source hashes
on every rank, including the control recipe. The composed manifest is necessary
because the 8192 overlay extends an already-patched model runner.

The older child-image builders are restricted to the saved 2026-09-25 pins.
They reject the latest bundle with instructions to use `performance-settings.py`.
No second child build is needed. An old-image rollback template is saved as
`cluster-karmic-20260925.json`; retain the actual old operator recipe and old
image tag for rollback. Changing to `fleet.control.json` keeps the new upstream
but disables the new local serving overlays.

## One qualification pass

First confirm the build, safety/context probes, four-rank fabric qualification,
finite-logprob smoke requests, and representative tool/structured-output
requests. Run control and candidate in the same idle window, using identical
real prose, code and JSON prompt files and temperatures 0 and 1. Check draft
acceptance, output correctness, memory, graph replay and aggregate decode at
concurrency 1, 8 and 16. Record actual image IDs from manual preflight.

The benchmark's HTTP-only mode never contacts a node over SSH:

```bash
python3 benchmark.py --config fleet.performance.json --http-only \
  --prompt-file real-prose.txt --input-tokens 1024 --max-tokens 512 \
  --temperature 1 --top-p 0.95 --concurrency 16 --requests 32 \
  --output results/candidate-prose-t1-c16.json
```

Repeat for the control recipe, temperatures and prompt types. For prefill use
long, varied real text at the desired context lengths; an undersized file is
rejected rather than repeated. The report retains generated text, finish
reasons, corpus hash, sampling parameters and HTTP metrics before/after. The
legacy no-file filler mode is explicitly labeled `repeated_filler` and is not
evidence of general prefill throughput. TTFT includes queueing/sampling, and
client decode rate assumes one token in the first text chunk. Neither metric
is a kernel step timing. Cold NVMe/page-cache qualification must be established
separately by the operator, with the cache state and competing load recorded.

If the combined candidate regresses, use the independent switches for one
focused follow-up. No repeated rebuilds are needed to isolate a setting.

## Local validation and remaining hardware work

CPU/source tests exercise actual patched methods, exact apply/reapply/revert
guards, complete source composition, async-copy ownership, prefix-cache events
and promotions, IPC wakeup transitions, GC restoration, real tensor aliasing
and one-fill/two-fill dispatch. The native GPU probes cover delayed DMA with
pinned-allocator reuse, finalizers after a real CUDA gate, and repeated graph
replay of native barrier arenas. Existing DSpark tests cover native CED/context
projection and shrinking-batch replay.

The [local validation record](performance-validation.json) reports 123 passed
tests and five skipped display-reserve observation checks. ARM64 Docker build,
CUDA/CUTLASS execution, four-node serving, quality and performance remain to be
run on the idle fleet. Karmic still pins CUTLASS DSL 4.7.1 while b12x declares
4.6.2; the build preserves Karmic's dependency and requires GPU qualification,
as documented in the original migration report.
