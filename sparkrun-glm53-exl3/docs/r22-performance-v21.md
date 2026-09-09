# v21: archived R7 paired FC2 M8 weight reuse as an R22 prefill option

2026-09-09. Layered on the v20 InstantTensor revision 1 image. Retains R22 v20
compute, MTP work and the loader policy; adds only the archived R7 paired-FC2
behavior behind its own rollback switch. Source findings and qualifications:
[investigation](r7-to-r22-performance-investigation.md); action plan:
[repo-root plan](../../R22-ARCHIVED-KERNEL-PLAN.md).

```bash
bash sparkrun-glm53-exl3/scripts/build-r22-v21-image.sh WORKER1 WORKER2 WORKER3
```

Use `recipes/glm53-exl3-v21-4x.yaml`. This candidate needs the GB10 build,
cumulative GPU smoke and serving measurements. Local CPU validation cannot
establish CUDA compilation, numerical correctness or a throughput change.

## The opening

The archived R7 kernel decodes one B/scale bundle and applies the decoded
weight fragment to two independent M8 accumulator sets. R22's grouped schedule
instead loops the grouping factor calling a complete `_run_tile` per M8
subtile; each repeats B staging and trellis LUT dequantization of the same
weight tile. At TP4, FC2 has width 6144 and tile N512, so each packed M8
route block contributes twelve mn-tiles and always fills two exact 48-SM
waves. Prefill is where v20 still trails the safe image (645 versus 741 tok/s
at 8k target-only), and the long-standing prefill gap persists and is now
quantified. Pairing halves FC2 weight staging/dequantization repetitions at
equal work; this does not imply half the DRAM traffic or twice the model
throughput, and FC1, attention, collectives and other work remain.

## The change

`VLLM_GB10_EXL3_FC2_M8_PAIR=1` restores the archived R7 pair path for
grouped M8 prefill plans:

- `_run_tile_m8_pair` decodes one weight fragment per pair of adjacent M8
  subtiles and applies it to both accumulator sets with independent A
  operands; each M8 half keeps its own padded 16-row shared-memory slab and
  output metadata rows (the doubled A slab and route/rd-route/top-k regions
  grow the shared footprint; the valid-count slot stays a single region);
- production group4 (`VLLM_GB10_EXL3_FC2_GROUP=4`) dispatches as two pair
  calls; group2 dispatches as one. Lock slots stay consecutive in both arms,
  so every subtile keeps exactly one slot;
- M8 decode plans keep factor 1 and therefore never reach the pair path:
  the pair requires an even grouping factor of 2 or 4, which only exists for
  M16/M32/M64 (the smoke requires an unchanged M8 decode binary under the
  flag);
- the pair call keeps the current ABI, output fusion and rotations: one
  decoded fragment per two adjacent subtiles with a single B stream, and the
  store/drain machinery threads a metadata row base (archived R7 semantics)
  so a paired second half drains its own metadata rows.

The switch is read during GEMM compilation and is part of the compiled-kernel
cache key; restart all workers after changing it. It is mutually exclusive
with `B12X_W4A16_SMALL_M_SPLITK` and the v20 FC1 tail split, which rewrite
both phases and stay exclusive with this one. The fused kernel only offers
the flag to the FC2 GEMM of route-packed M8 plans with an even grouping
factor; the GEMM constructor rejects it for any other geometry, and the
mixed-pair contract raises a mismatch between the fused flag and the stock
schedule. Compile an isolated variant first and inspect real ptxas
register/spill counts; reject spilling variants.

## Validation and rollback

CPU tests reconstruct the exact pre-v21 kernel and mixed states (the v20
sigmoid output and the v19 grouped schedule are the last prior overlays),
gate hash preflight/idempotence and rejection, exercise the helper's
architecture/value gating, execute the GEMM constructor validation against
stubbed geometries, and run a Python model of the dispatcher that requires
every subtile to own exactly one consecutive lock slot for all block counts
and supported groupings. GPU gates compare the paired schedule against the
stock schedule for M32/M64 prefill at group2 and group4 across two and three
tiers, require a distinct compiled binary under the flag, a larger shared
footprint, an identical M8 decode binary, CUDA-graph replay with changed
activations, router weights and emptied experts, and print both timings per
shape.

The pair reuses one B stream for two accumulator sets; the arithmetic order
per expert is unchanged, so the agreement is exact up to bf16 store rounding
(the GPU gate compares paired against unpaired outputs directly).

Rollback is `VLLM_GB10_EXL3_FC2_M8_PAIR=0` plus a worker restart: that
recompiles the unmodified v20 schedule from the same image. The v20 image,
instanttensor recipe and recipe remain available as full rollback. The
recipe keeps the InstantTensor loader policy, MTP3, the v19 `FC2_GROUP=4`
setting and the RoCEnante-enabling command unchanged, so prefill
measurements isolate the pair alone.

## Measurement guidance

Prefill is the target, not decode. Compare with one env flip at a time and
restart all workers:

1. group2 paired versus group2 unpaired (isolates weight reuse from
   grouping);
2. group2 paired versus production group4 unpaired;
3. group4 paired (two pair calls) versus production group4 unpaired.

Use identical tokens, output lengths, cache contents, graph settings, KV
capacity and machine state; warm up and repeat runs in alternating order and
report medians/spread. Then repeat with MTP3 and record acceptance as well as
throughput. A surviving gain could motivate examining weight placement and
memory pressure; a kernel gain alone does not establish an end-to-end
benefit. If neither prototype accounts for the measured prefill gap, follow
the trace's largest unexplained cost; do not assume the whole advantage
belongs to the MoE kernel.
