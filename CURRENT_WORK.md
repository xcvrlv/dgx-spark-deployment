# Current work — sparkrun GLM-5.3 EXL3 mixed-K performance

Working notes as of 2026-09-08. Scope: `sparkrun-glm53-exl3/`, four DGX
Sparks (GB10/SM121, 48 SMs), TP4/DCP4, RoCEnante from the local-inference-lab
b12x commit `1a7e3ec` backport, model `davidsyoung/GLM-5.3-EXL3-TR3-3.42bpw`
(mixed K3/K4 routed experts, BF16 shared/dense experts online-encoded to
Trellis K6), native MTP3. Everything below is source-level unless marked as a
user-observed measurement; nothing here is GPU-qualified on the dev host.

## State of the candidate line

- **v20 (new, this session): GB10 FC1 whole-tile tail split.** Implemented,
  CPU-tested, hash-pinned; needs Spark build + GPU smoke + serving A/B. See
  `sparkrun-glm53-exl3/docs/r22-performance-v20.md`. This is the FC1-tail
  candidate from `docs/sm121-kernel-opportunity-map.md`, scoped to
  route-packed M8 decode/MTP plans only.
- v19 remains the rollback: M32 prefill, `VLLM_GB10_EXL3_FC2_GROUP=4`
  (user-observed slightly faster than group2), M16 now compilable, and the
  recipe no longer carries `--disable-custom-all-reduce`.
- User-observed performance context: the historical "safe" image (recipe
  `glm53-exl3-4x-safe.yaml`) still holds the best prefill at over 700 t/s
  for the 8k uncached prefill sample; v19 sits around 650 t/s prefill with
  higher decode. v13 closed most of the prefill gap (to under 100 t/s of
  safe at the time); the remaining prefill difference is unexplained.

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

## Next steps

1. Build v20 on a Spark:
   `bash sparkrun-glm53-exl3/scripts/build-r22-v20-image.sh WORKER1 WORKER2
   WORKER3` (distributes, verifies image IDs, runs the cumulative GPU smoke
   on all four nodes - includes the new tail-split shape ladder with
   timings).
2. Serving A/B with the existing benchmark (8k uncached prefill sample,
   decode at concurrency 1 and 8, medians, MTP acceptance from
   instrumentation): recipe `glm53-exl3-v20-4x.yaml` first with
   `VLLM_GB10_EXL3_FC1_TAILSPLIT: "0"` then `"1"` (restart all workers; one
   env change at a time). Decode is the target metric; prefill should be
   unaffected (verify with the prefill sample anyway).
3. If v20 wins: consider the FC2 decoded-weight reuse experiment (v21) with
   the ptxas-register gate; if it loses, the timing JSON from the smoke's
   shape ladder localizes which remainder sizes regressed.
4. Optional parallel experiment: `gpu_memory_utilization` 0.88/0.89 with
   v17's reclamation enabled, observing the `GB10 startup memory` boundary
   logs first (per the v17 doc).

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
