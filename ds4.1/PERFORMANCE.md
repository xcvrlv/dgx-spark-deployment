# V4.1 performance candidate

The working c16 full-decode-only `cluster.json` remains unchanged. These are
unmeasured candidates, not a claim of a 3x prefill or 50% decode improvement.

## Profiles

| File | Graphs | SSD read unit |
|---|---|---|
| cluster-perf-c8-control.json | Full decode only | 4096 |
| cluster-perf-c8.json | Full decode + breakable piecewise | 4096 |
| cluster-perf-c8-sector512.json | Full decode + breakable piecewise | 512 |

All three use c8, fixed DSpark5, 10 GiB KV per node, 1024-token scheduler chunks,
and the existing 1,048,576 context limit. Explicit KV allocation supersedes
automatic utilization-based sizing; it does not cap total host or graph memory.
Keep earlyoom enabled. Adaptive verification stays off to avoid its implicit
mode override and variable-length verification graphs.

Piecewise buckets include 128/256/512/1024 and all multiples of 5 and 6 needed
for eight draft/target batches. The native full graph manager bounds decode
captures by the request count. Starting with 1024 limits the larger prefill
graph allocations. Once it fits, test 2048 by adding that bucket AND increasing
max_num_batched_tokens to 2048. Treat 4096 similarly; graph capture retains
communication buffers, so the previous eager chunk size is not automatically
a safe capture size.

## Build and run on the head

Copy the updated ds4.1 directory to the head. This is a small overlay of the
working v2 image; it does not rebuild vLLM, require the GLM fabric image again,
or download/convert weights.

```bash
python3 ds4.1/cluster.py build --config ds4.1/cluster-perf-c8.json &&
python3 ds4.1/cluster.py copy-image --config ds4.1/cluster-perf-c8.json

# Benchmark with serving stopped to avoid competing for NVMe/GPU resources.
python3 ds4.1/cluster.py stop &&
python3 ds4.1/cluster.py disk-bench --config ds4.1/cluster-perf-c8.json > ds41-disk-bench.log 2>&1

python3 ds4.1/cluster.py start --config ds4.1/cluster-perf-c8.json
```

The real-checkpoint benchmark tests both tables on every host at
1/6/48/512/4096 token-equivalent batches, at both read sizes. It applies TP row
ownership and verifies sampled weight AND scale bytes against file reads. It
does not load the model. These are warmed-device random O_DIRECT measurements,
not end-to-end inference or guaranteed cold-flash measurements.

If 512-byte O_DIRECT is unsupported, the benchmark fails with earlier output
retained; the 4096 profile still works. Serving preflight tests only its selected
read size against real shards, plus the existing synthetic FP4 graph oracle.
If 512 has lower latency at relevant batch sizes on all hosts, try it:

```bash
python3 ds4.1/cluster.py stop --config ds4.1/cluster-perf-c8.json &&
python3 ds4.1/cluster.py start --config ds4.1/cluster-perf-c8-sector512.json
```

For the controlled graph comparison, use cluster-perf-c8-control.json. Rerun
the same c1/c2/c4/c8 tests and 8K/64K prompts at identical thinking mode,
temperature, output length and prefix-cache state, with warmup and at least
three repetitions. The original c16, 4096-chunk run is not an isolated graph
comparison. Collect complete logs using:

```bash
python3 ds4.1/collect-logs.py --config ds4.1/cluster-perf-c8.json
```

Reported baseline: prefill 496 tok/s at 8K and 452 at 64K; decode aggregate
c1/c2/c4/c8/c16 = 30/45/58/79.8/105.6 tok/s. Targets, not predictions: prefill
about 1500/1350 tok/s and decode c1 >=45, c8 >=119.7 tok/s.

## SSD changes and interpretation

The existing native reader uses O_DIRECT: the OS page cache does not satisfy
its table reads. It coalesces repeated/adjacent blocks within each batch but
keeps no cross-step data cache. Weight and scale rows occupy separate regions.
One isolated FP4 lookup can therefore read 8192 bytes for 128+8 useful bytes,
before boundary crossings/coalescing. FP4 storage does not automatically mean
half as much physical I/O as FP8.

The optional patch makes request alignment/length 512 or 4096 bytes per reader.
Buffer alignment remains 4096. It preserves existing sorting, coalescing,
ownership, decoding and error handling. Smaller reads reduce amplification,
not the two-plane read count; an 8x speedup is not implied. Unsupported direct
reads fail, without silent buffered fallback. Exact source hashes gate patches.

DS41_DISK logs contain cumulative per-table native time, boundary wall time,
useful/read bytes and read count. Difference consecutive samples within one
prefill or decode phase. Sum both tables on a node, not times across hosts.
Boundary wall time includes waiting for prior GPU work and ID transfer; its
excess over native reader time must not be blamed entirely on the SSD.
No extra GPU synchronization is added. Set disk_log_every to 0 after diagnosis.

## Architecture and reference review

Reviewed [Joe's DSV4.1 patches](https://github.com/josephdrose/joe-spark-patches/tree/9f51d214e4a9e2874f6e06fcbf4008a5edee8006/dsv41)
at 9f51d214e4a9e2874f6e06fcbf4008a5edee8006. They use older vLLM, FlashInfer
sparse MLA and an interleaved FP8 reader. Their measurements are comparison
points, not results for our native B12x backend.

| Reference change | Decision |
|---|---|
| Prestage Engram outside forward | Already in our pinned model_state; retain it |
| Full/piecewise and draft/target buckets | Adapted to c8 and bounded prefill |
| Force page64 | FlashInfer-specific; keep native B12x block256/SWA32 ABI |
| Replace persistent_topk | Not our MXFP4 indexer path; do not patch unused code |
| Missing-op shim | No missing-op evidence in our working B12x runtime |
| Interleaved asynchronous reader | Different FP8 representation; not a drop-in FP4 replacement |

The native MXFP4 indexer computes local head scores, all-reduces them over TP4,
then selects globally. Some score widths come from configured context capacity,
rather than actual prompt length. This is another plausible short-context cost.
Do not select local top-k before reduction: that changes the global selection.
Several GB10 V4.1 policies are untuned; RTX PRO 6000 profiles do not establish
valid or faster Spark settings.

For a short GPU trace, copy a profile JSON and set profile_trace to true. Start
that config and run the existing benchmark; while it is busy, trigger:

```bash
curl -fsS -X POST http://127.0.0.1:8000/start_profile
```

The trace is limited to four iterations, below each node's
cache_path/ds41-traces. Inspect score collectives, MoE/GEMM, graph/eager dispatch
and DSpark acceptance before choosing an architecture rewrite. Use an unprofiled
launch for throughput measurements.

DCP4 is explicitly rejected by the pinned attention constructor: V4.1 shards
heads, not context. It requires cache ownership, distributed index selection
and attention reduction changes. GLM DCP patches do not supply that contract.
DCP1 remains enforced until there is correct support and a throughput comparison.

## Validation

CPU contract tests, Python compilation, pinned-source patch application,
idempotence and deliberate drift rejection pass. Native compilation, actual
512-byte reads, graph fit, model correctness and performance still require the
Sparks. Keep the working v2 image and cluster.json for rollback via stop/start.
