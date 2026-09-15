# Exact c1-c8 graph coverage with a c16 scheduler — RETIRED 2026-09-15

This experiment is retired: the DS41 request-bucket patch, its child image
builder, its launcher activation variable and its config key were removed
after the state-stage coverage OOM (see UPSTREAM-R38.md). The machinery
described below no longer exists in the recipe; this document is retained as
the decision record. The capture ladder is the sparse spread in fleet.py.

This candidate fixed graph shape coverage, not a proven throughput result.
Keep the operator's existing k5, OMP, memory utilization0.85 and prefill batch
settings for the comparison. No full JJ/b12x compilation is required: build a
small child image on top of the installed pinned image.

## Change

The current DSpark adaptive-verification graph path provides dense request
specializations only at c1/c2, while most other token totals have
num_reqs=min(num_tokens,max_num_seqs). At max_num_seqs16 this can replay an
8-request workload with 16-request metadata and use different verification
cost estimates than an otherwise identical max_num_seqs8 server.

The optional patch adds exact request capacities 1,2,3,4,5,6,7,8 and the
configured maximum. For each capacity R, it enumerates every integer token
count from R to R*(maximum drafts+1), bounded by the graph token cap. It covers
heterogeneous verification lengths, not only multiples of the maximum width.
The original compatibility checks and priority ordering select the smallest
compatible request shape at the nearest token count. Mixed prefill must still
pass the query-length checks; it cannot enter a decode-only graph.

With max16/k5 this raises target FULL graph count from106 to276; k7 goes from
142 to380 in the pinned planner. Counts are for normal capture, not the two
sample graphs in memory profiling. The profiler sees the added descriptors and
includes them in its graph-memory estimate. Adaptive verification subsequently
profiles the actual shapes and obtains exact request-count cost curves for
c1-c8. Buffers remain bounded by existing token96/k5 or128/k7 capacity and
request16 capacity. More graph instances can consume more memory even with
shared buffers; 0.85 memory fit must be verified on the fleet.

Runtime switch: graph_request_buckets=true in the JSON passes
DS41_GRAPH_REQUEST_BUCKETS=1. Missing/false restores the original policy.
The patch is limited to DSpark variable-length decode; fixed verification and
target-only graph policies are untouched. It retains c9-c16 fallback coverage.

## Upstream and additional optimizations considered

Heads checked 2026-09-13 22:13 UTC: JJ fa6ae921 and b12x135c9715. Compared with
pinned JJ9342b1a/b12xfd3c638; the newer JJ does not modify cudagraph_utils.py.
The patch guards the complete source SHA256 and rejects unknown versions.

New upstream adds Engram resident scales, bounded cross-table prefetch,
projection TP, decode-capacity forwarding, and corrected adaptive-verification
metrics. Those are separate candidates, not bundled here. Resident scales
consume more host memory; projection TP adds communication; prefetch changes
I/O concurrency. Their wins on this four-Spark workload need matched runs.
The decode-capacity change addresses reservations beyond256 rows; current
c16/k5 reserves176 and c16/k7 reserves240, so it does not directly explain this
observed c8 regression. New LM-head defaults also change between revisions,
which is another reason to keep the baseline image fixed for this experiment.

## Build and launch

First copy the updated ds41-vllm files from Windows to the Spark. From the Spark
repo directory, choose SOURCE as the actual measured k5 config. This generator
preserves image base, model path, memory/OMP/batch/depth settings and adaptation
options. It changes only maximum sequences, graph switch and child image tag.

```bash
(
set -euo pipefail
SOURCE=.build/cluster-c16-latest.json
BASE_IMAGE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["image"])' "$SOURCE")" bash build-graph-image.sh
python3 configure-graphs.py --from-config "$SOURCE" --output .build/cluster-graphs.json
python3 fleet.py --config .build/cluster-graphs.json share
python3 fleet.py --config .build/cluster-graphs.json stop
python3 fleet.py --config .build/cluster-graphs.json start
)
```

Expected worker log: `DS41 DSpark graph request buckets=[1, 2, 3, 4, 5, 6, 7, 8, 16]`.
The launcher rejects enabling this flag on an image lacking the patch label.
The start command performs fabric qualification and c16 serving smoke tests.
If the base image was changed locally, the source hash guard still checks the
actual installed vLLM file before applying the child layer.

## Compare and roll back

Run the same benchmark, prompt set and generation settings as the original
comparison, without profiling, at every concurrency c1..c8 and at c16. Compare
c8/max16 patched against both c8/max16 original and c8/max8 original. Preserve
image IDs, actual container arguments, acceptance counts and TTFT as well as
aggregate decode throughput. Profile short windows separately if needed.
No tuning change should be made between these runs.

For a same-image control, generate with --control, then restart:

```bash
python3 configure-graphs.py --from-config .build/cluster-graphs.json --output .build/cluster-graphs-control.json --control
python3 fleet.py --config .build/cluster-graphs-control.json stop
python3 fleet.py --config .build/cluster-graphs-control.json start
```

Alternatively select the original image/config. The image patch also supports
--revert for a separate image build. Do not modify a live server's Python files.

## Validation status

CPU tests execute the actual upstream graph enumeration and compatibility
functions with lightweight configuration objects. They exhaust all legal token
totals for c1..c16 at k1/k3/k5/k7, require exact token AND request counts for
c1..c8, compare c8 shapes with the max8 baseline, reject overlong prefill as
FULL decode, check graph-cap limits, deduplication, fixed-mode/no-switch parity,
source drift rejection, apply/reapply/revert, configuration preservation and
image-label preflight. These are graph-policy checks, not kernel correctness
or memory measurements. The ARM64 Docker build, actual captures, numerical
serving checks and throughput qualification have not run in this Windows
workspace. The fleet experiment is required before calling throughput fixed.
