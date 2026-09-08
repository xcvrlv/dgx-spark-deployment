# v20: GB10 FC1 whole-tile tail split for M8 decode/MTP plans

```bash
bash sparkrun-glm53-exl3/scripts/build-r22-v20-image.sh WORKER1 WORKER2 WORKER3
```

Use `recipes/glm53-exl3-v20-4x.yaml`. This candidate needs the GB10 build,
cumulative GPU smoke and serving measurements. Local CPU validation cannot
establish CUDA compilation, numerical correctness or a throughput change.

## The opening

`docs/sm121-kernel-opportunity-map.md` mapped the strongest small decode/MTP
candidate: the whole-tile FC1 schedule fills complete waves, then runs a
ragged remainder on only part of the grid. At TP4, FC1 has width
2*512 = 1024 and tile N128, so each packed M8 route block contributes eight
mn-tiles. One decode token routed to eight distinct experts therefore makes
64 FC1 mn-tiles: exactly one full 48-SM wave plus a 16-tile remainder that
leaves 32 SMs idle while 16 finish their whole-K tiles. MTP3 verification
batches produce the same remainder pattern at larger wave counts. FC2 makes
96 mn-tiles (N512/tile) and always fills two exact waves, so only FC1 has
this underfill.

## The change

`VLLM_GB10_EXL3_FC1_TAILSPLIT=1` keeps every completed whole-K wave and
stripes ONLY the ragged remainder across all CTAs along K, reusing the
existing split-K machinery end to end:

- the remainder's `(tile, k)` cells are dealt to CTAs by the stock tail
  partition (`iters = ceil(k_tiles*tail/grid_x)` cells per CTA);
- slices combine through the existing lock-ordered fp32 `fc1_scratch` turns
  (`_wait_for_reduction_turn`/`_combine_splitk_accumulators`), and the final
  slice stores the bf16 tile as before;
- the pair-rate and staging paths already take absolute K-tile indices under
  sliced jobs (`reduce_k_tile + tile_idx`), which upstream validated with the
  stripe switch on and off;
- at 64 mn-tiles the remainder runs as three slices per tile across all 48
  CTAs instead of 16 whole-K tiles on 16 CTAs.

The remainder for this geometry is always a multiple of `n_tiles` between 8
and 40 tiles: lock slots 0..tail-1 stay far inside the `sms*4 = 192` lock
region, and the doubled M8 fp32 scratch (`2 * fc1_cols * route_slots`)
covers every decode shape with margin (it equals `2048 * route_slots` f32
against a worst need of `tail * 2048` with `tail <= 40` and
`route_slots >= 56` whenever a remainder exists). The workspace barrier
cells at `[sms*4]`, `[sms*4+1]` are untouched.

Two runtime guards keep the stock behavior where splitting cannot pay:

- exact grid fills (`global_mn_tiles % grid_x == 0`) and single-wave batches
  (`global_mn_tiles < grid_x`) keep the stock schedule — the idle-CTA handoff
  is cheaper than a split-K finalize for phases that small;
- the whole decision is device-side, so varying packed counts across decode
  steps and graph replays need no recompilation.

The switch is read during GEMM compilation and is part of the compiled-kernel
cache key; restart all workers after changing it. It is mutually exclusive
with `B12X_W4A16_SMALL_M_SPLITK`, which rewrites both phases and forces FC2
grouping to one. This schedule preserves FC2 grouping (v19's group4) and
every prefill plan unchanged: the fused kernel only offers the flag to the
FC1 GEMM of route-packed M8 plans, the GEMM constructor rejects it for any
other geometry, and M16/M32/M64 plans compile to bit-identical binaries (the
smoke requires a cache hit for M32 with the flag off and on).

Scope qualification is deliberately narrow, mirroring v19: mixed-K two-tier
and three-tier decode plans on SM121 with the pinned `(128, 128, 32, 512)`
tile geometry. No activation dtype, dequantization, rotation or FC2 path
changes.

## Validation and rollback

CPU tests reconstruct the exact pre-v20 kernel state (the v16 sigmoid output
is the last prior kernel.py overlay), gate hash preflight/idempotence and
rejection, exercise the helper's architecture/value gating, execute the
GEMM constructor validation against stubbed geometries, and run a Python
model of the runtime partition that requires every remainder cell to be
 covered exactly once and every remainder tile to own one lock slot for all
block counts. GPU gates compare the striped schedule against the stock
schedule for decode rows 1/5/17/33 across two and three tiers, require a
distinct compiled binary under the flag, an unchanged `blocks_per_sm`, an
identical prefill binary, CUDA-graph replay with changed activations, router
weights and emptied experts, and print both timings per shape.

The tail split reorders FP32 partial accumulation between slices, so the
numerical gate is tolerance-bounded (relative RMS below 0.5% and a per-
element check, like the v19 M16 gate) rather than bitwise.

Rollback is `VLLM_GB10_EXL3_FC1_TAILSPLIT=0` plus a worker restart: that
recompiles the unmodified v19 schedule from the same image. The v19 image
and recipe remain available as full rollback. The recipe keeps utilization,
KV/graph budgeting, batching, prefix policy, dense quantization, MTP, the
v19 `FC2_GROUP=4` setting and the RoCEnante-enabling command unchanged, so
decode measurements isolate this schedule alone.

## Measurement guidance

Decode is the target, not prefill. Compare with one env flip at a time and
restart all workers:

1. `TAILSPLIT=0` baseline (v19 schedule, everything else identical);
2. `TAILSPLIT=1` candidate.

Report median generic and structured decode at concurrency 1 and 8 with MTP
acceptance from the instrumentation counters; a decode gain is not sufficient
if acceptance falls. The relevant shapes are the captured decode graphs
(4-32 verification rows). Watch for the known failure mode: cheap MCG
finalize overhead overtaking the reclaimed idle time on small batches — the
runtime guards exclude single-wave batches, but mid-size remainders (8-40
tiles) can still lose; timing JSON from the smoke's shape ladder shows the
trend before serving is measured. The smoke prints `fc1_mn_tiles` per shape
so the observed packed counts can be matched against the remainder math.

If the striped schedule wins, the natural follow-up remains the mapped
prefill candidate: reusing decoded FC2 weights across adjacent M8 subtiles
(group4 currently repeats staging and dequantization per subtile). That
requires a dual-accumulator `_run_tile_m8` variant and a measured register
table entry, so its first gate is an isolated ptxas register/spill
inspection; reject spilling variants before integration. Expert parallelism
stays deferred per the v19 findings.
