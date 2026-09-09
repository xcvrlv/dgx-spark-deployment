# Current work — sparkrun GLM-5.3 EXL3 mixed-K performance

Working notes as of 2026-09-08. Scope: `sparkrun-glm53-exl3/`, four DGX
Sparks (GB10/SM121, 48 SMs), TP4/DCP4, RoCEnante from the local-inference-lab
b12x commit `1a7e3ec` backport, model `davidsyoung/GLM-5.3-EXL3-TR3-3.42bpw`
(mixed K3/K4 routed experts, BF16 shared/dense experts online-encoded to
Trellis K6), native MTP3. Everything below is source-level unless marked as a
user-observed measurement; nothing here is GPU-qualified on the dev host.

## State of the candidate line

- **v20 (built, served; MTP3 serving user-observed similar; target-only now
  measured): keep v20 as the serving baseline.** The matched target-only
  comparison (entries 1-2 in `TARGET-ONLY-PERFORMANCE.md`, 2026-09-08) shows:
  v20 prefill trails safe by ~96 tok/s at 8k (645 vs 741; ~112 at 64k, ~83 at
  128k) — the long-standing prefill gap persists and is now quantified; v20
  decode at ctx 0 (30-second samples) wins at conc 4 (+19.6%) and 8 (+5.1%),
  trails at conc 1 (-4.3%) and 2 (-7.2%). The keep-v20 decision rests on the
  MTP3 serving parity plus the production-concurrency decode wins; the clean
  tail-split isolation is a `TAILSPLIT=0` vs `1` target-only A/B.
  See `sparkrun-glm53-exl3/docs/r22-performance-v20.md` for scope and rollback.
- v19 remains the rollback: M32 prefill, `VLLM_GB10_EXL3_FC2_GROUP=4`
  (user-observed slightly faster than group2), M16 now compilable, and the
  recipe no longer carries `--disable-custom-all-reduce`.
- User-observed performance context: the historical "safe" image (recipe
  `glm53-exl3-4x-safe.yaml`) holds the best prefill (741/757/711 t/s at
  8k/64k/128k, target-only); v20 sits at 645/645/628 target-only. The
  safe-vs-v10..20 scheduling-difference inventory and the EXL3 kernel-path
  streamlining study (longer-horizon) live in `EXL3-SCHEDULING-NOTES.md`.

## What was mapped this session

### 1. Image building (sparkrun portion)

Two image lineages:

- **Safe lineage**: `spark-vllm-glm52-exl3:sparkring-switch-v1` is a
  pre-existing SparkRing R7 image (pinned R7 vLLM at `/opt/venv`, switched
  NCCL at `/opt/sparkring/nccl`). `Dockerfile.prefill` adds exactly one
  hash-gated patch (sparse-indexer workspace right-sizing, MiaAI-Lab
  `2022ce5`) to produce `sparkring-switch-prefill-v2`. No smoke, triple
  preflight hashing instead. Recipe differences vs the R22 line: instanttensor
  load format, `B12X_MLA_SPARSE` attention backend name, gpu memory 0.91,
  compat libcuda preloaded, `use_local_argmax_reduction: false`, no RoCEnante
  env block at all.
- **R22 line**: `Dockerfile.r22-dflash2` rebuilds R22 vLLM (`70b3c1c`) + B12X
  (`1e59a1f`) + EXL3 R7 loader + exllamav3 ext + QUTLASS on a digest-pinned
  ARM64 vLLM nightly, sourcing only `/opt/sparkring/nccl` from the safe
  image, then applies the v10 Python performance overlay. v11-v20 are one
  thin Docker layer each: `FROM` previous tag, `COPY patch+smoke` to
  `/opt/compose`, a heredoc that applies `patch(root); patch(root,
  check=True)` to BOTH the retained source tree and installed site-packages,
  `ENV` defaults, a build-time CPU smoke, `LABEL`. Every patch file carries
  SHA-256 INPUT/OUTPUT manifests and refuses unknown bytes; each version's
  docker build runs `smoke_r22_vNN.py` (CPU) and the per-node GPU smoke runs
  the newest script, which chains every inherited GPU test (latest hash wins
  via `source_overrides`). Recipes change only name/tag/metadata/env-adds
  (test-enforced).

Anomaly note confirmed: the "safe" recipe predates the versioning and is a
different image layout entirely (R7-based); it is not v9 or v10. Its prefill
advantage has not been reproduced by any R22-line setting tried so far
(v12 documented failed attempts: 4000/8000-token batches, scratch capacity,
piecewise graphs).

### 2. Kernel opportunity map -> v20 implementation

FC1/FC2 job math (TP4, FC1 width 1024/N128 -> 8 tiles, FC2 width
6144/N512 -> 12 tiles; decode block M8): one token to 8 distinct experts =
8 packed blocks = 64 FC1 jobs (one full 48-SM wave + 16-tile ragged
remainder) and 96 FC2 jobs (two exact waves). Wave-slot utilization 64/96.

The existing `B12X_W4A16_SMALL_M_SPLITK` switch is NOT usable for this: it
forces `schedule_whole_tiles=False` and FC2 factor 1 globally (undoes
group4, rewrites prefill too), and the stock non-whole-tile tail heuristic
(`tail*3 <= grid`) would stripe ALL FC1 tiles at 64/48, not just the tail.

v20 therefore adds a narrowly scoped schedule inside the whole-tile arm:
- new env `VLLM_GB10_EXL3_FC1_TAILSPLIT` (SM121-only, 0/1, invalid values
  raise), read at compile, in the GEMM cache key;
- complete waves stay whole-K; the ragged remainder (always a multiple of
  n_tiles, 8..40 tiles for this geometry) is striped across all 48 CTAs
  along K through the existing tail partition, lock-ordered fp32
  `fc1_scratch` turns and final-slice bf16 store;
- runtime guards: exact fills and single-wave batches keep the stock
  schedule (no split-K finalize for tiny phases);
- wiring: fused kernel offers the flag only to FC1 of route-packed M8 plans
  (`not small_m_splitk and schedule_whole_tiles and moe_block_size == 8 and
  not direct_topk_routes`); GEMM constructor raises for any other geometry;
  FC2 and prefill untouched (smoke requires an M32 cache hit with the flag
  off and on).

Safety proofs recorded (details in the v20 doc): pair-rate/staging use
absolute k indices (`reduce_k_tile + tile_idx`) unconditionally, validated
upstream with the stripe switch; lock slots stay 0..tail-1 well inside the
192-slot lock region; the doubled M8 scratch (== `2048 * route_slots` f32)
covers the worst remainder need (`tail * 2048`, tail <= 40, route_slots
>= 56 whenever a remainder exists) with the min-branch cap 786,432 f32 also
safe; c.g. reduction turns reset locks to 0, so graph replay stays clean.
Idealized ceiling: the remainder is 25% of FC1 tile-time at 64 jobs, and
striping reclaims at most 2/3 of it if fully memory-bound; finalize
overhead will eat part of that. No end-to-end percentage is claimed.

### 3. Remaining candidates (mapped, not implemented)

- **FC2 decoded-weight reuse across group subtiles (prefill candidate)**:
  `_dispatch_tier_gemm` loops the grouping factor calling a complete
  `_run_tile` per M8 subtile; each repeats B staging + trellis LUT dequant
  of the same weight tile. A fused two-subtile variant (one B stream, two
  accumulator sets) targets prefill directly, where v19 still trails the
  safe image. First gate per the opportunity map (and the repo's
  no-guessed-registers rule): compile an isolated variant and inspect real
  ptxas register/spill counts; reject spilling variants. Current FC2 M8
  register entry is 118; extra live accumulators (~16 f32 + A bundle) could
  cross a residency boundary. This is the natural v21; it cannot be
  responsibly hash-pinned without a GPU compile inspection.
- **Pipeline depth (`_STAGES=4`)**: still no clear edge; FC1 register
  pressure, not shared memory, limits occupancy. Unchanged recommendation.
- **Expert parallelism**: stays shelved (v19 doc's findings: adapter rejects
  `expert_map`; RoCEnante has no dispatch/combine protocol; ~3.60/4 nodes
  visited per token under uniform ownership).

### 4. Operational facts worth keeping in view

- v13-v18 serving runs did NOT use RoCEnante: the inherited
  `--disable-custom-all-reduce` disabled R22's `_ENABLE_CUSTOM_ALL_REDUCE`
  gate for both TP and DCP selection. v19 removed the flag. Historical
  decode/prefill observations before v19 ran NCCL on the RoCEnante-eligible
  paths; keep that in mind when comparing numbers across versions.
- The v19 recipe ships `VLLM_GB10_EXL3_FC2_GROUP: "4"` (user observation);
  `M32/group2` remains the documented rollback baseline.
- Known pre-existing test failures (both present at HEAD `3bae6b6`, not caused
  by v20; git history confirms my working tree only adds v20 files):
  - `tests/test_r22_dflash2.py:218` expects `gpu_memory_utilization: 0.895`
    while the v10 dflash2 recipe pins `0.89` (already flagged in the v11 doc
    as a stale expectation).
  - `tests/test_r22_v19.py::test_recipe_preserves_baseline_and_builder_is_cumulative`
    fails because HEAD's "v19 recipe tweaks" commit collapsed
    OPENBLAS/MKL/NUMEXPR/RAYON into `OMP_NUM_THREADS: "8"` in the v19 recipe
    without updating the test's reconstruction. Blessing that collapse or
    reverting the recipe is the operator's call; the v20 recipe gate below
    compares v19-to-v20 and is unaffected.
- Memory-utilization 0.89 attempt (with v17 startup reclaim) is a separate,
  orthogonal experiment; do not mix it into the v20 decode A/B.
- **[user-observed] `instanttensor` model loading is much faster** than the
  R22 line's `safetensors` mmap load and should be enabled for the latest
  recipes too. It is already in the safe recipe command
  (`--load-format instanttensor`, `glm53-exl3-4x-safe.yaml`); every R22-line
  recipe (v10..v20) still pins `--load-format safetensors`. This is a
  startup/model-load observation, not a throughput claim. Enable it as a
  recipe correction: first verify the R22 image's vLLM accepts the
  `instanttensor` load format (it is a different vLLM release than the safe
  image's R7 lineage), then flip the recipe value and restart all workers.

## Next steps

1. Target-only decode on v20 is **measured** (entries 1-2 in
   `TARGET-ONLY-PERFORMANCE.md`). The 300-second safe arm is now recorded
   (entry 5): safe wins conc 1 (+18.3%) and 2 (+9.0%); v20 wins conc 4
   (+6.6%) and 8 (+6.4%) - the same v20-wins-at-production-concurrency shape
   as the 30-second comparison. **Still unresolved: the `TAILSPLIT=1` arm**
   (entry 4 is byte-identical to the `=0` arm) - verify engagement first
   (set `VLLM_GB10_EXL3_FC1_TAILSPLIT: "2"` briefly; the helper raises at
   GEMM compile if the env is read), then rerun the intended arm or replace
   the table. Do not compare entry 3 against entry 2 (30 s vs 300 s).
2. Capture the safe image's `exl3.py` / `kernel.py` / `mixed_trellis.py`
   SHA-256 on a Spark (command in `EXL3-SCHEDULING-NOTES.md` §1.5) to close
   the lineage-identity question.
3. Ranked streamlining candidates and their first gates are tracked in
   `EXL3-SCHEDULING-NOTES.md` §3-4 (decode blocks_per_sm=2 spike, route-pack
   fusion trace, topk_sum/FC2-epilogue and FC2-weight-reuse ptxas gates).
   Old v20-era steps below are done or superseded.
4. Optional parallel experiment: `gpu_memory_utilization` 0.88/0.89 with
   v17's reclamation enabled, observing the `GB10 startup memory` boundary
   logs first (per the v17 doc).
5. **Blocked pending crash investigation (2026-09-08):** user reports a
   segmentation fault when trying InstantTensor with v20. Do not promote
   `--load-format instanttensor` to the latest recipes based on safe-image
   success or format recognition alone. Keep `safetensors` as the working
   baseline until this exact stack passes loading and serving. See the
   priority queue entry below; this supersedes the earlier recipe-flip advice.

## Priority issue queue: v20 InstantTensor segmentation fault

- **User-observed, 2026-09-08:** attempting InstantTensor with v20 segfaults.
  Worker log now supplied: rank 0 / pid 206 initializes at 16:39:21 via
  `tcp://10.3.10.1:25000`; vLLM reports NCCL 2.30.7 from
  `/opt/sparkring/nccl/libnccl.so.2`. TP and DCP RoCEnante initialize
  successfully. Model loading starts at 16:39:44, then the InstantTensor
  progress bar shows `0.00/331G` followed by `!!!!!!! Segfault encountered
  !!!!!!!`. Failure is during startup loading, before any reported loading
  progress, not a serving FC1-tail-split failure. Progress reporting is
  throttled, so 0% does not prove no tensors were processed. No native stack
  is present; later WorkerProc/EngineCore traces report worker loss only.
  Attachment: `C:/Users/Juho/.codex/attachments/3f7d7184-0738-4e9c-824e-17bf09386394/pasted-text.txt`.
- **Interpretation limits:** 331G is the iterator's logical selected tensor
  total, not evidence that 331G was allocated on one Spark. No explicit OOM,
  earlyoom action or RoCEnante timeout is shown. SymmMem's unsupported-12.1
  warning is followed by successful B12X/PyNCCL selection; do not treat that
  warning or the generic torch.compile warning as the demonstrated cause.
  vLLM's PyNCCL version log does not identify all libraries loaded by native
  InstantTensor or Torch's ProcessGroupNCCL; inspect actual mappings.
- **Verified in repo:** the checked-in v20 recipe still uses `safetensors`.
  Its image inherits InstantTensor from the ARM64 base; the base Dockerfile
  checks only package metadata `instanttensor >= 0.1.9`. That check and the
  ordinary smoke do not qualify real InstantTensor model loading on the
  CUDA 13 / Torch 2.13 stack. v20's new code changes FC1 inference scheduling,
  not the loader. A crash confirmed inside loading should be investigated
  there before attributing it to tail splitting.
- **Candidates, not diagnoses:** native I/O/CUDA or binary-library
  compatibility; distributed NCCL loading; staging allocation pressure;
  interaction between EXL3 streaming tensor ownership and reusable buffers.
  The R22 iterator fixture accepts a world-group NCCL process group and
  supports `copy=False`, marking tensors `_vllm_instanttensor_borrowed`.
  The restored EXL3 adapter already clones borrowed tensors in multiple
  paths, so a blanket assertion that ownership protection is missing would
  be wrong. Audit actual call-site settings and remaining retention paths.
- **Next evidence/gates:** obtain the native crash
  information (`PYTHONFAULTHANDLER=1` for Python context, native core/backtrace
  if available); compare loaded InstantTensor/Torch/CUDA/NCCL libraries with
  safe; isolate one small local safetensors file before a four-rank load;
  test supported I/O backend choices only after checking the installed
  package API. Inspect buffer-size and distributed/copy controls in the
  actual installed default loader before recommending flags. Separate
  staging-memory pressure from SIGSEGV: earlyoom SIGTERM/OOM SIGKILL are
  different failure modes. Retain the working safetensors load for serving.
- **Scope:** investigation queued, no image/recipe/kernel fix implemented.
  Safe-image InstantTensor success does not establish R22 compatibility.
- Sources: `Dockerfile.r22-dflash2:152`, `Dockerfile.r22-dflash2-v20`,
  `tmp/exl3-v17/raw/model_executor/model_loader/weight_utils.py:1206`,
  `tmp/exl3-v16/base/vllm/model_executor/layers/quantization/exl3.py:1789,2495`.
  Upstream buffer-lifetime/distributed-loading contracts:
  https://github.com/scitix/InstantTensor#zero-copy-mode

## File inventory for v20

- `sparkrun-glm53-exl3/overlay/patch_r22_v20.py` - single kernel.py overlay,
  INPUT `88b864c6...` (= v16 output, the pre-v20 image state), OUTPUT
  `f37db321...`.
- `sparkrun-glm53-exl3/overlay/smoke_r22_v20.py` - cumulative CPU gates +
  GPU decode shape ladder (rows 1/5/17/33 x two/three tiers x flag 0/1),
  prefill cache-hit requirement, graph replay with emptied experts, timings.
- `sparkrun-glm53-exl3/Dockerfile.r22-dflash2-v20` - FROM v19, ENV default
  `VLLM_GB10_EXL3_FC1_TAILSPLIT=0`, LABEL `glm53-r22-v20-1`.
- `sparkrun-glm53-exl3/scripts/build-r22-v20-image.sh` - cumulative builder
  (11..19 + v20).
- `sparkrun-glm53-exl3/recipes/glm53-exl3-v20-4x.yaml` - v19 + v20 metadata
  + `TAILSPLIT: "1"`; everything else byte-identical (test-enforced).
- `sparkrun-glm53-exl3/tests/test_r22_v20.py` - 8 CPU regression gates.
- `sparkrun-glm53-exl3/docs/r22-performance-v20.md` - full scope, math and
  rollback documentation.


## v20 InstantTensor revision 1 prepared (2026-09-09)

Supersedes the earlier "no image/recipe/kernel fix implemented" investigation
scope: a separate loading-policy candidate is now built on head 10.3.10.1,
`spark-vllm-glm53-exl3:r22-dflash2-sm121-v20-instanttensor-r1`. It uses buffered
local loading, bounded staging and explicit owned copies, retaining v20
compute and MTP settings. Source gates and parsed recipe parity passed.
No confirmed segfault fix or inference improvement yet: a preliminary GPU
probe hit CUDA OOM before loading while the existing service was running.
The original service remains up; worker distribution and GPU/full-model A/B
qualification await an idle window. See
`sparkrun-glm53-exl3/docs/r22-instanttensor-v20-r1.md` for evidence, exact image
IDs, build/launch commands, test gates and rollback.

## v21: archived R7 paired FC2 M8 weight reuse as a prefill option (2026-09-09)

Implements the plan's Priority 1: the archived R7 pair kernel
(`_run_tile_m8_pair` + `_run_mma_pipeline_m8_pair` + `_read_moe_block_data_pair`
+ `_tile_common_prologue_pair` + `_load_next_fragment_bundle_m8_pair`) is
adapted into an R22 prefill option, layered on the working v20-instanttensor-r1
image. The paired kernel decodes one weight fragment per pair of adjacent M8
subtiles and applies it to both accumulator sets with independent A operands;
production group4 dispatches as two pair calls and group2 as one; M8 decode
keeps factor 1 and never reaches the pair path. Each M8 half keeps its own
padded 16-row shared-memory slab and output metadata rows (the doubled A slab
and route/rd-route/top-k regions grow the shared footprint; the valid-count
slot stays a single region). The switch is read during GEMM compilation and is
part of every affected cache key; the mixed-pair contract raises a mismatch
between the fused flag and the stock schedule. Source gates and parsed recipe
parity passed. No confirmed inference improvement yet: ptxas register/spill
inspection and paired-FC2 numerical qualification await an idle GPU window.
See `sparkrun-glm53-exl3/docs/r22-performance-v21.md` for the evidence, exact
image identity, build/launch commands, test gates and rollback.
