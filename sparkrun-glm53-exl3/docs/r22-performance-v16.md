# v16: DCP prefill, workspace reuse and RoCEnante CPU overhead

Build on the head Spark, from the repository root:

```bash
bash sparkrun-glm53-exl3/scripts/build-r22-v16-image.sh WORKER1 WORKER2 WORKER3
```

Use `recipes/glm53-exl3-v16-4x.yaml`. This overlays v15 and preserves its
compute changes. GPU memory utilization remains **0.87**, prefill block M32,
batch/scratch capacity 4096, and native MTP3. No throughput measurements were
possible on the Windows development machine; v16 is ready for cluster testing.

Revision `glm53-r22-v16-2` also fixes the inherited sigmoid's CuTe
`UNSUP_EARLY_EXIT` compilation failure. Both runtime branches assign a result
and the method returns once afterward. The reciprocal and division fallback
calculations are preserved. Re-run the same v16 build command; the fix is a
late v16 patch over the existing v15 image and does not require changing v15.

## What changed

| Change | Why it can help | Disable for comparison |
| --- | --- | --- |
| CKV gather dispatch to the existing RoCEnante `out=` API | The full-CKV helper previously called PyNCCL directly, bypassing the DCP RoCE adapter. Eligible shards now use direct switched-fabric fanout into the attention workspace, with no temporary gathered tensor. | `VLLM_GB10_DCP_GATHER_INTO=0` |
| In-place CKV gather | The local packed cache occupies this rank's slice of the final gathered cache. Removes the separate rank-local CKV staging reservation. | `VLLM_GB10_CKV_INPLACE=0` |
| Bounded indexer merge workspace | Reuses the existing workspace manager for packed candidates and gathered output; merges up to 256 independent query rows at a time. Covers prefill, target decode and MTP indexer invocations. | `VLLM_GB10_DCP_MERGE_ROWS=0` |
| Initialize only RoCE protocol state | Skips clearing unused send/receive payload slots at startup. Flags, control words and device counters still start at zero. | `B12X_ROCE_LAZY_PAYLOAD_INIT=0` |
| Skip polls on an empty send completion queue | Tracks pending signaled completions per HCA. The proxy avoids repeated provider calls between collectives when no completion can arrive. | `B12X_ROCE_SKIP_EMPTY_CQ=0` |

All five are enabled in the image and recipe. Set the same values on every
rank and restart all workers when comparing. Disabling all five restores v15
execution paths. The merge-row setting accepts 0..1024; 256 is the initial
candidate, not a measured optimum. The gather flag controls both CKV and the
indexer's direct output path. Indexer workspace reuse can operate without it,
using the existing group collective and copying its result into the arena.

## Concrete findings and memory accounting

The full-CKV helper in the pinned R22 backend explicitly preferred PyNCCL,
then the Torch distributed group, and only used `group.all_gather` as a final
fallback. Consequently, the earlier DCP adapter changes did not route normal
full-CKV prefill through RoCEnante. This is a concrete dispatch difference;
it does not prove why the older safe image was faster.

v16 checks the active, collectively initialized RoCEnante adapter and its
rank-invariant eligibility limits before using its existing runtime. Errors
after dispatch propagate; they never trigger an unsafe local NCCL fallback.
The runtime retains its lock, stream ordering, health checks, staging protocol,
and dual-HCA payload/flag ordering. The configured 16 MiB per-rank gather cap
remains in force. Larger CKV shards still use the original NCCL path. The
switch permits direct peer fanout, but total traffic still shares each NIC's
bandwidth; no ring-to-mesh protocol rewrite was necessary.

For top-k 2048, four ranks and 4096 query rows, an unbounded indexer merge
allocates 64 MiB of packed candidates plus 256 MiB of gathered candidates.
v16 reserves **20 MiB** in the shared operator arena for 256 rows (4 + 16 MiB).
Small decode/MTP merges use smaller views; they do not create these two
transient allocations during graph capture. The exact stable top-k reducer,
FP32 scores, global-index mapping and per-row selection are preserved.
No candidate pruning, quantization or approximate top-k is introduced.
More chunks mean more collective launches for large merges, so an A/B run
should check this tradeoff. Existing indexer prefill chunks may already be
smaller than 4096, reducing the old path's actual transient peak.

The paged indexer's scratch is dead before candidate merge. Scores and the
output index buffer remain separately owned. The merge reducer does not
borrow another workspace. The new reservation is made in **every existing
microbatch/model lane before KV profiling**; target and draft lanes remain
separate. The arena never grows during graph replay. A larger attention or
indexer scratch requirement may dominate its allocation, so 300 MiB is the
removed worst-case transient allocation, not a guaranteed net-memory saving.

With 656-byte cache records and 131072 aggregate context tokens, a separate
CKV local staging region costs about **20.5 MiB plus alignment/request slack**
per workspace lane. v16 removes that requirement, preserving the gathered
capacity. The local slice offset uses `rank * padded_active_tokens`, not the
maximum capacity. This matches [NCCL's documented in-place all-gather layout](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/inplace.html).
For RoCE dimension-zero gather, local copies map to their original input
addresses and peer output ranges are disjoint. No global staging barrier is
assumed. Arbitrary overlapping input/output layouts are not newly enabled.

The pinned RoCE region remains about 160 MiB per TP4/DCP4 communicator at a
16 MiB slot size, plus small protocol records. Selective initialization skips
about 160 MiB of CPU zero stores per communicator; **it does not reduce pinned
allocation size or memory registration**. Every payload byte consumed by a
collective is written first. Slots outside the active message are never read.

## CPU and remaining paths

The CQ change counts one signaled completion per successful payload/flag
chain, regardless of whether that chain has one or two work requests. Polling
continues while anything is outstanding, including idle-period polling and
queue backpressure. Completion errors remain fatal. No CPU affinity, real-time
priority, sleep threshold, fences or RDMA queue-depth changes are imposed.
The current eight-thread CPU math settings are retained; there is no cluster
trace here establishing that reducing them or pinning proxies to unknown
GB10 CPU cores would help.

Large NCCL operations remain in use. [NCCL buffer sizing](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
can trade memory against communication behavior, but shrinking those buffers
without a measurement could offset the prefill gains. v16 leaves them alone.
It also retains the small DCP reduce-scatter-via-all-reduce path: eliminating
its redundant network bytes requires a peer-specific payload protocol, beyond
an output-slice optimization. No unsupported claim of zero-copy RDMA is made;
RoCEnante still stages payloads in its registered pinned region.

## Validation

Seven v16 CPU tests pass. They execute the transformed CKV and indexer methods
with instrumented tensors, exercise partial chunks and arena bounds, check
rank offsets and empty shards, verify untouched padding, enforce fail-stop
dispatch, and verify selective protocol initialization. The overlay rejects
source drift before writing any file and accepts exact reapplication.
The sigmoid regression check requires a single final return and exercises
finite, infinite and NaN inputs with the optimization enabled and disabled.
This CPU check does not substitute for CuTe compilation on the Spark.

The Docker build compiles the actual changed proxy and runs a native C harness
covering pending, empty, completed and failed CQ states without needing an
HCA. Its normal source-hashed loader is also checked. Per-node GPU smoke keeps
the inherited v12-v15 numerical checks and adds actual `cp_gather_cache`
in-place tests, real Triton candidate packing and CuTe stable merge comparisons,
locked workspace reuse, and graph replay with changed scores. Old verifier
entry points are not run against superseded source hashes.

The build script distributes the exact image and runs per-node GPU tests.
It does **not** run cross-node RDMA qualification automatically. With the model
service stopped, use the [four-node command in v10](r22-performance-v10.md#four-node-communication-check),
changing the image to `spark-vllm-glm53-exl3:r22-dflash2-sm121-v16` and script to
`/opt/compose/smoke_r22_v16.py --distributed`. Also pass
`-e VLLM_ROCE_ALLGATHER_MAX_SIZE=16MB` so the larger CKV cases fit the same limit
as the recipe. It checks actual in-place RoCE gathers across changing message
sizes, bounded DCP indexer merge/replay, and the inherited TP4 MTP winner path.

GPU execution, ARM64 compilation, cross-node transport, end-to-end throughput
and actual peak memory still require the Sparks. Compare the same prompts,
prefix-cache state and MTP settings; retain v15 for rollback. If prefill
regresses, compare the gather switch and merge-row switch separately first.

Pinned sources: [CKV backend](https://github.com/local-inference-lab/vllm/blob/70b3c1c7f1c76fcf0847fcbb4a0b8b5583b78d19/vllm/v1/attention/backends/mla/b12x_mla_sparse.py),
[DCP indexer](https://github.com/local-inference-lab/vllm/blob/70b3c1c7f1c76fcf0847fcbb4a0b8b5583b78d19/vllm/v1/attention/backends/mla/b12x_indexer.py),
[workspace lanes](https://github.com/local-inference-lab/vllm/blob/70b3c1c7f1c76fcf0847fcbb4a0b8b5583b78d19/vllm/v1/worker/workspace.py),
[RoCE runtime](https://github.com/local-inference-lab/b12x/blob/b58f34eaf978277621efced6678e6713fd7122e4/b12x/comm/roce/roce_oneshot.py).
