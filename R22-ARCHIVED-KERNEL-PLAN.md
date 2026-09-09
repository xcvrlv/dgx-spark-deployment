# Recover archived R7 performance in R22

2026-09-08. Replaces the proposed safe-image backport: retain R22 v20 and its
MTP work; bring across only archived behavior that wins a controlled comparison.
Source findings and qualifications: [investigation](sparkrun-glm53-exl3/docs/r7-to-r22-performance-investigation.md).

## Priorities

1. **Prefill: restore paired FC2 M8 weight reuse as an R22 option.** Archived R7
   decodes one weight fragment and applies it to two M8 accumulator sets.
   R22's grouped schedule repeats staging/dequantization for each subtile.
   This is a concrete implementation to adapt for the already-proposed FC2
   reuse experiment, with M32 prefill first. It does not execute on M8 decode.
2. **C1: isolate the small route-packer change.** R7 uses comparison/reduction
   counting; R22 uses an atomic histogram plus barriers and binary search.
   Benchmark both inside R22 at identical live and padded shapes. Preserve
   R22's capacity handling, preallocation and larger-batch path. A tiny-shape
   dispatch is justified only if measured; do not replace the packer globally.
3. **Account for the remaining end-to-end gap.** Safe's matched C1 result is
   12.9 versus 10.9 tok/s (v20 tail split off): about **14.2 ms/token** to
   explain. Neither a faster isolated kernel nor source differences establish
   where that time went. Price collectives, attention, dense K6, graph padding
   and CPU gaps before selecting further changes.

## Execution

1. **Record exact baselines on the Sparks.** Verify installed safe/R22 source
   hashes and Torch/CUDA/Triton/CUTLASS versions against the archived findings.
   Freeze model revision, quantization, cache state, power settings, batching
   and graph policy. Record actual MTP-off commands, graph-selected/padded rows,
   kernel `active_m`, route counts and engaged transport per rank. Repeat the
   300-second C1/2/4/8 runs; keep contexts/token positions comparable, since a
   faster 300-second run traverses more decode positions.
2. **Profile before attributing.** Capture representative steady C1 and uncached
   8k/64k prefill traces for both images. Split the critical path into route
   packing, fused MoE, top-k output, dense/rotations, attention/indexer,
   collectives and host gaps. Use identical saved inputs/routes for kernel
   replay. Test R22 with eligible RoCEnante paths disabled and NCCL verified
   as a separate diagnostic; do not combine that flip with kernel changes.
3. **Run two independent R22 prototypes.** Add a prefill-only paired-FC2 switch
   and a separately gated archived small-prefix variant. Start paired FC2 with
   group2 on both arms to isolate reuse, then compare against production
   group4 and evaluate group4 as two pair calls. Measure actual registers,
   shared memory, spills and residency. Preserve R22 binding/storage ABI and
   current dtype/rotation fixes; bump affected compilation identities.
4. **Qualify and retain winners.** Reuse existing smoke infrastructure with
   identical inputs, changed-input graph replay, empty/duplicate routes,
   expert boundaries and mixed K3/K4/K5 coverage. Test live/padded rows
   1/2/4/8/16/17/32/33 and M32 prefill, including partial blocks. Require
   numerical agreement and an end-to-end gain exceeding run variation.
   Validate MTP3 throughput, accepted/drafted tokens and output correctness
   separately, preserving the likely MTP advantage rather than assuming it.

Ship a new cumulative R22 candidate only after each winning change is
independently attributable, with its own rollback switch. Retest at least three
interleaved warm repetitions and report medians/spread. If neither prototype
accounts for the measured gap, follow the trace's largest unexplained cost;
do not assume the whole advantage belongs to the MoE kernel.

No serving code or defaults changed in this investigation. GPU timings,
installed-image identity and the exact cause of C1's advantage remain pending.
