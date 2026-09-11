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

Piecewise buckets are 128/256/512 plus all multiples of 5 and 6 needed for
eight draft/target batches. The 1024 bucket is dropped: the capture order is
descending (largest first, cudagraph_utils.py:476-477), each captured graph
retains `capture` + `resources` sized by its width with strong refs that the
next width's capture cannot reuse, and with explicit `kv_cache_memory_bytes`
memory profiling is skipped entirely (gpu_worker.py:519-541), so nothing
reserves the graphs' retained memory — the 19-width set accumulated past the
pool at capture ~15-16 (width 12-15, not 1024). The published DSv4.1 recipe
captures no big prefill buckets for the same reason. With 1024 dropped,
`max_cudagraph_capture_size` becomes 512; prefill chunks above 512 run eager
(the 512 bucket still matches partial chunks and small mixed batches). Once
capture completes, the prefill bucket question re-opens with the memory
picture from `VLLM_DEBUG_GRAPH_MEMORY_ACCOUNTING=1`, which is now set: it
logs per-capture `[CG MEM]` pool growth and attributes active blocks to
file:line sites.

## Earlyoom and page cache

The September 11 piecewise crash (worker died at capture 12/19, exit code
None) matches the documented GLM-5.3 failure on these hosts: available memory
fell during `FULL_AND_PIECEWISE` capture and earlyoom sent SIGTERM to the
highest-oom-score worker. SparkRun's earlyoom configuration prefers
`vllm|python` processes, so the worker dies the moment MemAvailable collapses.
Lazy safetensors loading brings the ~81 GiB of per-rank weight files into page
cache page by page over the whole load window, and capture plus retained
communication buffers consume more on top; NVIDIA forum reports confirm
page-cache saturation on DGX Spark requires reclamation before every
large-model run.

The GLM-5.3 fix is ported. `cache-flusher.sh` runs on each host (not in the
container): `sync` plus `echo 3 | sudo -n tee /proc/sys/vm/drop_caches` every
60 s for a bounded 90-minute window, single-instance via flock, matching the
SparkRun scoped clear-cache sudo rule. `cluster.py start` installs and starts
the flushers on all four nodes and refuses to launch without live flushers
(fail closed), then stops them once health is ready — health is ready only
after capture, so the window covers the whole load and capture phase.
`cluster.py flush-cache` runs them stand-alone. The DS41 Engram reader uses
O_DIRECT, so the flusher cannot interfere with the SSD path's data; only
file-backed pages are dropped. The qualified GLM measurement on these hosts
was 101.49 GiB free before reclamation and ~112.9 GiB after.

If earlyoom still intervenes with the flusher active, retain its message, the
flusher log and the last startup log; the host-policy fallback is raising the
swap threshold (`EARLYOOM_ARGS` `-s 80` → `-s 20` in `/etc/default/earlyoom`)
or excluding the serving processes, then restarting earlyoom. Keep earlyoom
enabled.

## Runtime JIT wedge

The published 4x Spark DSv4.1 recipe exhausted host memory on all four nodes
from a FlashInfer GEMM compiled at runtime with 22 parallel jobs, and its fix
is prebuilt kernels plus MAX_JOBS=2. The image ships `flashinfer-jit-cache`
precompiled, but DeepGEMM/CuteDSL shapes can still compile at capture time.
The env now bounds runtime JIT parallelism with `MAX_JOBS=2` and
`FLASHINFER_NVCC_THREADS=1`, and carries `CUDA_DEVICE_MAX_CONNECTIONS=32`
from the qualified GLM env — the classic NCCL/CUDA-graph scheduling
mitigation. =1 is a pre-Blackwell setting that must not be used on GB10.

## P2P and fabric

`VLLM_ENABLE_PCIE_ALLREDUCE=0` is env-asserted and NVLS is off (0 nvls
channels logged). The `P2P Chunksize`, `isAllDirectP2p` and `p2p channels`
lines in the NCCL log are NCCL's intra-node P2P tuning output — moot with one
GPU per host — and the ring/tree connections go via NET/IB over the two RoCE
HCAs as intended. Neither the qualified GLM env nor the published DSv4.1
recipe sets `NCCL_P2P_DISABLE`, so none is added here.

## Published reference

[tonyd2wild's DSv4.1 recipe](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark)
(4x DGX Spark TP4, DSpark k=5, FULL_AND_PIECEWISE, Engram-on-disk) reports
one-stream decode 73.8 tok/s, six streams 131.9 aggregate and prefill
902-1539 tok/s cold. That confirms the eager 496/30 baseline is not the
hardware's ceiling; their documented four-node host-memory crash fix was
prebuilt kernels plus MAX_JOBS=2, ported above. Their one-stream number is a
comparison point, not a result for our native B12x backend.

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

`start` now brings up the page-cache flushers on all four hosts first and
refuses to launch without them; it stops them once health is ready. A failed
launch leaves them running for the bounded window; `flush-cache` restarts
them to refresh the full window on the next attempt.

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

User-supplied disk benchmark (September 11): all 75 pasted records passed byte
checks. Rank 0's layer-1 4096-byte records were omitted from the paste. For the
three complete ranks, the median of each rank's summed two-layer median wall
times is:

| Token-equivalent batch | 4096-byte reads | 512-byte reads |
|---|---:|---:|
| 1 | 0.495 ms | 0.335 ms |
| 6 (c1 target verification) | 1.021 ms | 0.763 ms |
| 48 (c8 target verification) | 4.808 ms | 3.386 ms |
| 512 | 34.044 ms | 31.713 ms |
| 4096 | 221.006 ms | 238.839 ms |

These are sums of separate medians, not measured joint end-to-end percentiles.
Smaller reads help the decode-size gathers but lose about 8% at 4096 despite
roughly seven times fewer bytes. At 4096 they issue about 48K reads per table
versus 44K, consistent with less coalescing and a request-rate/processing limit.
1024 was not measured, so its winner is unknown. Keep both profiles for A/B.
At the previous 4096-chunk setting, ~0.22 seconds for both tables compares with
~8.3 seconds to prefill 4096 tokens at 496 tok/s. Standalone SSD time is thus
only roughly 3% of that budget; it cannot by itself explain a 3x prefill gap.
Live DS41_DISK timing and GPU traces remain necessary to locate in-serve stalls.

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
