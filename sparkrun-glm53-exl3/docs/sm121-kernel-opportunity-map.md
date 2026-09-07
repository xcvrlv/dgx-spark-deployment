# Limited SM121 kernel opportunity map

Baseline: user reports M32/FC2 group4 slightly faster than group2; M16 and
M64 are slower. EP is deferred. This is source-level mapping, not an image
patch or a measured speedup claim.

B12X explicitly targets SM120 and SM121, including Spark. Our pinned W4A16
source already contains an SM121 register table and uses the runtime SM count.
The useful question is whether each phase's geometry fits this workload on
48 SMs, rather than assuming the library has no GB10 support.

## 1. FC1 tail scheduling: strongest small decode/MTP candidate

In the current mixed-Trellis compilers, `schedule_whole_tiles=True` selects
whole-K jobs. At TP4, FC1 has width 2*512=1024 and tile N128: eight output
tiles per packed route block. FC2 has width 6144 and N512: twelve tiles.
Decode uses M8, independently of the M32 prefill setting.

For a single token routed to eight distinct experts, there are eight packed
route blocks: 64 FC1 jobs and 96 FC2 jobs. With the current one-block/SM
persistent grid, FC1 has 48 jobs followed by 16; FC2 has two full waves.
The FC1 wave-slot utilization is 64/96, or 66.7%. This is a scheduling count,
NOT measured GPU utilization or memory-bandwidth utilization. Sixteen SMs
might already provide enough outstanding reads to saturate GB10 memory.

For 10/12/14/16 packed blocks, FC1 wave-slot utilization is respectively
83.3%/100%/77.8%/88.9%. Actual MTP routing determines packed-block counts,
so a policy must be conditional on work count, not just token count.

Candidate: keep the grid and FC2 schedule intact, partition only an
underfilled FC1 tail along K, and combine its partial results. The existing
persistent scheduler has a split-K tail path and mixed-tier emission hooks,
so this need not start as a new GEMM implementation. Retain K128 and the
current expert/bitrate mapping. Extra reduction, locking and scratch traffic
may erase the benefit on cheap MCG decoding.

Do not simply enable `B12X_W4A16_SMALL_M_SPLITK`: the pinned constructor
overrides whole-tile scheduling and forces FC2 grouping to one for all affected
plans. It does not isolate the hypothesized FC1 win and would undo group4.
The source also has mixed historical correctness commentary for that switch;
it is not a qualification result for our mixed-K model.

First gate: compare whole-K versus FC1-tail-only scheduling for real 1/4/8/16/32
row routing, including K3/K4/K5 imbalance within a node. Measure complete MoE
time as well as FC1. Check numerical results and graph replay, and verify that
FC2 group4 remains unchanged for prefill. Stop if memory throughput is already
saturated and tail scheduling only adds overhead.

Sources: `kernel.py::_run_persistent_gemm`,
`W4A16FusedMoeKernel.__init__`, `mixed_trellis.py::compile_mixed_trellis`
and `compile_mixed_trellis3` in the retained pinned fixtures.

## 2. Reuse decoded weights between FC2 subtiles: prefill candidate

`mixed_trellis.py::_dispatch_tier_gemm` loops over the grouping factor and
calls the full `gemm._run_tile` for each M8 subtile. Group4 co-locates four
jobs, but does not itself share decoded weight registers across them. Each
subtile executes its own pipeline; cache hits may avoid external-memory
reads, but staging and dequantization work are still repeated.

Candidate: decode a K tile once and use it for two adjacent M8 accumulators.
Start with two, not four. This targets repeated staging/dequantization while
preserving M32 route packing. User-observed group4 improvement makes locality
worth investigating, but does not prove dequantization is the bottleneck.

This is more invasive than schedule tuning. Wide N512 accumulators already
consume substantial registers. Reusing B while keeping two output tiles live
can spill registers and become slower. Shrinking N to compensate adds more
output jobs and may require different prepared weight layouts.

First gate: compile an isolated FC2 variant and inspect actual register and
spill use before integrating it. Reject spilling variants. Compare against
M32/group4, including complete fused-MoE time and peak scratch usage.

## 3. Pipeline depth: lower priority

The pinned W4A16 code uses four staging slots (`_STAGES=4`). A phase-specific
three-stage variant could be benchmarked, but lowering shared memory does not
automatically increase occupancy: the current FC1 register estimates already
limit M16/M32 to one block/SM. Fewer stages can also expose memory latency.
This is not a clear edge without evidence of pipeline/resource stalls.
It would need consistent shared-memory layout, wait-group and cache-key
changes, not a runtime environment edit of a single constant.

## Scope and decision

Investigate FC1 tail scheduling first; decoded-weight reuse second. Do not
increase the cooperative grid beyond residency limits, force two blocks/SM,
or assume datacenter Blackwell instructions apply to GB10. Existing compact
input rotations, sigmoid and fused output patches already address several
obvious activation/temporary-buffer costs.

No candidate has a defensible 3-5% end-to-end gain estimate yet. For scale,
if one affected phase accounts for 25% of runtime and becomes 20% faster,
the total speedup is 1/(0.75+0.25/1.20) = 1.0435, about 4.3%. Both inputs
need measurement, particularly with RoCEnante confirmed active.

References:
- B12X supported targets: https://github.com/local-inference-lab/b12x
- GB10 shared LPDDR memory architecture:
  https://docs.nvidia.com/dgx/dgx-spark-porting-guide/overview.html

No serving defaults, kernels, image tags, memory settings or transport
settings were changed during this preliminary pass.
