# Archived R7 versus R22 v20: performance investigation

2026-09-08. Objective: recover safe's prefill and target-only C1 advantages
inside R22 while preserving its newer MTP work. This is source analysis, not
GPU qualification or proof that the entire measured gap is kernel time.
Action plan: [repo-root plan](../../R22-ARCHIVED-KERNEL-PLAN.md).

## Evidence boundary

The [target-only log](../../TARGET-ONLY-PERFORMANCE.md) records safe prefill
741/757/711 tok/s versus v20 645/645/628 at 8k/64k/128k. The matched
300-second safe decode row is 12.9/21.8/35.6/54.1 versus
10.9/20.0/38.1/57.8 at concurrency 1/2/4/8. The latter uses tail split **off**;
the supplied on-arm remains unresolved. Safe's C1 advantage is 18.3% relative
to that R22 throughput, or 77.5 versus 91.7 ms/token. The approximately
14.2 ms/token gap is the attribution budget, not an established kernel saving.
If all of it came from 78 routed layers, each would need to save about 182 us.
Repeated trials and matched token positions are still needed.

Selected archived files were reconstructed from SparkRing commit
`510556275ed3b77fc56a14367d319417072eeb8c`, using its
[pins](https://github.com/FujitsuPolycom/sparkring/blob/510556275ed3b77fc56a14367d319417072eeb8c/runtime/exl3-r7/pins.json)
and
[Containerfile](https://github.com/FujitsuPolycom/sparkring/blob/510556275ed3b77fc56a14367d319417072eeb8c/runtime/exl3-r7/Containerfile).
Both integration-patch SHA-256 values and reconstructed patched Git blobs were
checked. EXL3 scratch preparation reproduces the build's `8e0051fa…` hash.
Archived kernel/mixed hashes are `ac732a7e…` / `dc03f8e7…`. Final v20 compute
was reconstructed through checked overlay transformations, yielding
`f37db321…` / `e2e645f5…`. The route-pack/intrinsics helpers were read at the
pinned B12X commits; the archived integration patch does not modify those files.

The installed safe image has not been inspected. Archive-to-image identity
and actual dispatch therefore remain the first runtime gate. Scratch sources,
hash checks and function comparisons are under `tmp/safe-v20-feasibility/`
(ignored; not a deployed patch).

## Strongest prefill lead: paired FC2

R7's `compile_mixed_trellis` and three-tier equivalent select paired M8 FC2
for M32/M64 route blocks. `_dispatch_tier_gemm` invokes `_run_tile_m8_pair`;
`_run_mma_pipeline_m8_pair` loads one B/scale bundle, decodes one `b_frag`,
then applies it to **both** `acc0` and `acc1` with independent A operands.
Each M8 half has its own padded 16-row shared-memory slab and output metadata.

R22 removed that implementation. Its dispatcher loops over grouped subtiles,
calling the complete `_run_tile` for each. v19's group4 changes job grouping;
it does not share the decoded weight fragment between those calls. Thus the
archived kernel supplies a concrete reference for the FC2 reuse opportunity
already described as a possible v21 in the working notes.

At equal work, pairing can halve FC2 weight staging/dequantization repetitions;
this does **not** imply half the DRAM traffic or twice the model throughput.
FC1, attention, collectives and other work remain. The two accumulator sets
also change registers/shared memory. R7 queries compiled resource attributes
and rejects nonzero local-memory use when introspection is available; those
results cannot be assumed to carry over to R22's compiler.

Implement an R22-compatible pair path behind a prefill guard, retaining its
current ABI, output fusion and rotations. Compare group2 paired/unpaired
first, then test production group4 and a two-pair group4 implementation. This
separates reuse from grouping. Check pair boundaries, odd final subtiles,
empty experts and cooperative-grid residency.

## Concrete C1 lead: different route packing

For small shapes, R7 counts routes through a two-dimensional expert/route
comparison and reduction. It resolves block ownership through comparisons
against prefix intervals. R22 instead zeros a global histogram, issues
`tl.atomic_add` for live routes, synchronizes, loads counts, computes a prefix,
and performs a binary search over stored offsets. Its small-prefix path has
three unconditional `tl.debug_barrier()` calls and another when counts alias
the packed arena. Both then use the separate atomic sort/scatter kernel.

R22's scalable approach is not automatically the lowest-latency choice for
eight live routes. R7's comparison tensors also have a cost, so there is no
source-only winner. The correct experiment swaps only the small-prefix
algorithm in R22 with identical inputs, capacities and compiler dependencies;
measure its time and the downstream fused kernel, not merely launch count.

The selection boundaries also differ. Evaluating the pinned host capacity
functions for top-8, 256 experts and M8 gives:

| Input rows | Route-capacity bucket | Packed block capacity | R7 path | R22 path |
| --- | ---: | ---: | --- | --- |
| 1 | 8 | 8 | small | small |
| 4 | 32 | 32 | small | small |
| 8 | 64 | 64 | small | small |
| 16 | 128 | 128 | small | small |
| 17–32 | 256 | 256 | large | small |
| 33 | 512 | 288 | large | small |

These are host-function results, not observed serving graph shapes. They show
why a global revert could damage larger MTP batches. Keep the R22 capacity
handling and graph-safe workspace preallocation; select the archived variant
only for shapes that independently win. Check routing multisets and numerical
outputs, allowing valid changes in atomic ordering within an expert.

## What does not currently explain C1

- Paired FC2 is off for M8. The R7 dense K6 small-M specialization is gated
  to capability `(12, 0)`; GB10 is `(12, 1)`.
- Both use `_STAGES=4`, one persistent CTA/SM for M8, the same dense launch
  geometry function, and the familiar mixed FC1/FC2 tile geometry. There is
  no source evidence for a secret R7 two-CTA decode schedule.
- v14 shared-input reuse requires **compiled capacity >64**, while the recipes
  plan decode capacity 64. Compact input depends on that same guard. These
  prefill optimizations are therefore not expected to alter C1's input rotation.
- The inspected native MCG FP16 dequant, stream alignment, async-copy and MMA
  helper functions are identical after AST normalization. The extracted R22
  ring-geometry helper implements the same arithmetic. Added SQG/pair-rate
  support is compile-time gated; do not count unused branches as runtime work.
  Compiler-generated instructions and registers still need comparison.
- The safe lead exists against v20 with tail split off, so disabling tail
  split alone cannot account for it.

## Remaining attribution work

Profile actual **critical-path** time per rank, separating route packing,
fused MoE, output reduction, dense K6/rotations, attention/indexer, collectives
and host gaps. Graph capture must record selected/padded row count and the
actual `active_m` passed to kernels. A nominal concurrency of one does not
prove every operation processes one row. Use representative saved model
inputs/routes to avoid attributing route differences to code changes.

R7 has a CuTe dense H128 implementation while v20 uses the v13 Triton fused
rotation path. Its latency/codegen is a secondary candidate if dense time
differs. The active v11 output-fusion and v15/v16 sigmoid switches are also
cheap within-R22 diagnostics; test individually and retain correctness gates.
NCCL versus RoCEnante, attention implementation, compiler stack, memory pressure
and graph/host overhead remain possible contributors. Price them before
porting unrelated source.

Two earlier scheduling assumptions are corrected by the archive: R7 already
budgets B12X indexer chunks by supertile, and its CKV minimum defaults to 16.
Therefore “R22 alone has supertile chunking” and “safe has no minimum-token
gate” are not supported by these pinned sources. Prefill also differs in
attention/query-splitting behavior, so paired FC2 must earn its share of the
measured gain rather than inheriting credit for the whole gap.

The initial implementation priorities are **paired FC2 for prefill** and
**tiny-shape route-packer isolation for C1**, with independent switches.
The exact C1 cause and the size of either gain remain unmeasured.
