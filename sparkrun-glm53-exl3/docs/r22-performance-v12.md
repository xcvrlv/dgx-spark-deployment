# v12: CKV preparation and query-copy candidate

Ready for a Spark build, not throughput-qualified. The user has confirmed v11
runs and improves decode in their observations, while prefill remains below
the older safe image. No hardware timing or CUDA execution was available on
the Windows development host for v12.

## Investigation

The user's 4000/8000-token batch and EXL3 scratch-capacity comparisons did not
improve prefill, and piecewise graphs did not help. v12 therefore retains 4096
batch/prefill capacity, block-M 32 and decode-only graphs. Its recipe sets
**GPU memory utilization to 0.87**, as required. TP4/DCP4, the checkpoint,
online K6 cache, mixed-K weight kernels and v11's fused output remain intact.

The safe image is `sparkring-switch-prefill-v2`, built from the older SparkRing
image with the indexer workspace patch. The checked-in older GLM runtime and
R22 use different attention implementations; copying legacy env names cannot
make them equivalent. Both have a full-CKV route with local query heads and an
ordinary DCP route with gathered query heads and an output combine. Both older
and newer source paths include multi-operation causal-length preparation.
Thus the changes below remove observed overhead, but **do not establish that
this overhead alone caused the safe-to-v11 regression**. The actual old image's
complete source/profiles and matched benchmark measurements are still needed
to attribute the entire gap.

The R22 full-CKV path previously performed output initialization, count
initialization, tiled atomic index compaction, a sequence of PyTorch operations
for causal lengths, a count clamp, and a tail mask for every attention layer.
Those small operations impose GPU launch and Python dispatch costs even when
the attention query batch has thousands of rows. Separately, ordinary DCP
can copy a contiguous gathered query into another buffer before consuming it.

Correction to the earlier CKV-limit explanation: the max-tokens setting is
checked against batch tokens **and** determines the gather workspace capacity.
At `131072`, roughly 128K aggregate context (plus alignment allowance) fits.
Longer contexts or multiple requests whose contexts exceed that capacity fall
back even if the current batch has only 4096 query tokens. v12 logs
`CKV context-capacity fallback` with per-rank required/available token counts.
It retains the 131072 limit; increasing it would increase memory and transport
costs and has not been qualified at memory utilization 0.87.

## Changes

- `VLLM_GLM53_FUSED_CKV_METADATA=1`: one Triton CTA per query row computes the
  rank/slot mapping, stable compaction, global causal length, selected count
  and masked tail. A builder-owned int32 vector holds causal lengths, avoiding
  per-layer temporary tensors. It is used only for GLM-DSA full-CKV batches.
  Other paths and GLM5Next retain their implementation. One-time engagement
  message: `v12: fused GLM-DSA CKV metadata active`.
- `VLLM_GLM53_BORROW_MLA_QUERY=1`: consume a BF16 query directly when its exact
  shape, contiguity and 16-byte alignment match, and its byte range does not
  overlap the attention scratch. Otherwise retain the copy. The backend reads
  this query without modifying it. This is principally a decode/MTP opportunity
  on the ordinary DCP route; tuple queries still use the existing concat.
  One-time engagement message: `v12: borrowed contiguous MLA query active`.

Flags are frozen at backend initialization. Set either to `0` in the recipe
and restart all workers for an independent A/B test. No weights, quantization
bit widths, attention precision or MTP sampling policy are changed.

The fused compaction uses input order instead of the old cross-CTA atomic
completion order. It preserves selected token sets for valid causal indexer
outputs, but the attention accumulation order may change floating-point
rounding. GPU smoke therefore checks exact selected sets/counts/causal lengths
and toleranced full attention, rather than promising bitwise model outputs.
End-to-end acceptance and quality still need observation.

## Build

```bash
bash sparkrun-glm53-exl3/scripts/build-r22-v12-image.sh WORKER1 WORKER2 WORKER3
```

The wrapper builds/reuses v10 and v11 locally, builds the late v12 layer, and
distributes only the v12 image with image-identity checks. Use
`recipes/glm53-exl3-v12-4x.yaml`. The older recipes remain rollback options.
The supplied recipe keeps MTP3; retain your intended MTP-step and graph-size
overrides consistently in both benchmark runs. Changing steps also changes
verification row counts and potentially which graph/kernel plan is used.

The build checks every GPU with the existing image checks, v11's output-fusion
checks, and the v12 smoke. The v12 smoke tests:

- Multiple requests and uneven four-rank shards, interleave 1 and 16, unsorted
  selections with holes, and 2–4096 query rows.
- Exact selected sets/counts/causal lengths versus the existing implementation.
- Graph replay after replacing valid input by entirely invalid selections,
  detecting stale output tails/counts.
- Real packed FP8 cache attention using the new index mapper, compared with
  global-cache attention and simulated four-rank DCP, including shuffled pages.
- Query borrowing eligibility and rejection of overlapping scratch.

The output includes `metadata_4096_rows` timing for legacy and fused preparation.
This isolates a changed stage; it is not a prediction of whole-model tokens/s.
No timing threshold fails the build, since GPU conditions vary. Numerical
failures do stop it. The optional four-node transport check is still available
via `/opt/compose/smoke_r22_v12.py --distributed` under the serving fabric's
four-node torchrun environment.

## Measure

Compare v11 and v12 at the same 0.87 memory setting, prompt, concurrency, MTP
steps, graph sizes, and cold-prefix-cache state. Also compare v12 with both
new flags off to separate recipe effects from code changes. Use repeated
alternating runs and keep the selected attention-path log with each result.
Measure prefill, decode, TTFT and accepted draft tokens per cycle. A matched
safe-image run is still needed to identify the remaining old/new difference.

CPU tests execute the actual new mapping function through a NumPy-backed
Triton-operation shim against an independent reference, verify alias/alignment
rejection, check pinned UTF-8 input/output hashes and idempotence, and check
build gates plus the required 0.87 recipe value. These do not substitute for
Triton compilation on SM121 or end-to-end model qualification.

Local result: 17 targeted tests pass (4 v12, 6 v11, 7 v10 performance), and
Python compilation plus whitespace checks pass. The separate legacy recipe
suite still stops on its stale 0.895 expectation for the older recipe, which
currently specifies 0.89. The new v12 test explicitly requires 0.87.
