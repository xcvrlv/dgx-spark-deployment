# Spark prefill audit — 2026-09-27

**Follow-up:** the subsequent [DSpark prefill implementation](DSPARK-PREFILL.md)
adds two optional source changes after the improved benchmark result. This
document records the preceding investigation and its original scope.

**Recommendation:** retain the Karmic 4096-token chunk and existing decode
configuration. First verify that its existing prefill graph is replaying, then
A/B upstream resident Engram scales (about **1.431 GiB per Spark**). A persistent
hot-row cache is plausible but is not yet justified by measured locality. The
only new source change prepared here is an optional, bounded prefix-hash copy;
it is CPU bookkeeping with tested equivalence, not a measured fleet speedup.

The active service could not be inspected: SSH to the inventoried head node
`10.3.10.1` timed out. No Spark was restarted, no image was built here, and no
GPU throughput, capture success, or memory fit is claimed. The screenshot is
measurement evidence, not instructions. “Gen4 ssh” is interpreted as Gen4 SSD.

## Evidence and upstream comparison

The screenshot reports one request per point:

| Actual prompt tokens | TTFT | Reported prefill tok/s |
| ---: | ---: | ---: |
| 8,188 | 1.95 s | 4,191 |
| 64,352 | 17.48 s | 3,682 |
| 128,519 | 37.34 s | 3,441 |

The last rate is 17.9% below the first. At the first rate the last prompt would
take approximately 30.7 s, leaving about 6.7 s of additional elapsed time.
This does **not** isolate disk latency: TTFT includes scheduling, CPU work,
communication, model execution and first-token work. One run gives no estimate
of variability, and prompt content changes Engram locality. Decode scaling
does not establish whether large prefill chunks use graphs.

Live GitHub API checks completed at **2026-09-27 16:25:59 UTC**:

| Source | Image pin | Checked head | Relevant difference |
| --- | --- | --- | --- |
| [Karmic Kraken](https://github.com/local-inference-lab/vllm/compare/1794dcf18454900263e0c66711af8ea4a1283ac1...953a636d3ae1fde86a02874fc919a95091c95275) | `1794dcf18454900263e0c66711af8ea4a1283ac1` | `953a636d3ae1fde86a02874fc919a95091c95275` | One EXL3 preparation commit; no changes to the audited graph, Engram, memory-accounting or block-pool files. |
| [Jovian Judgement](https://github.com/local-inference-lab/vllm/commit/8e1f1e587f8d24faf606f334a1c4bdaaa6bd4368) | Legacy comparison, not current base | `8e1f1e587f8d24faf606f334a1c4bdaaa6bd4368` | Unchanged from the preceding audit. |
| [b12x](https://github.com/local-inference-lab/b12x/compare/a7d7d29b2ef8869086e0ceaa787321f17544e3c9...d44247b6171f7c2f9787341ae884b537887d7df9) | `a7d7d29b2ef8869086e0ceaa787321f17544e3c9` | `d44247b6171f7c2f9787341ae884b537887d7df9` | Two commits for trellis/EXL3 and PCIe collectives; no fix to disk staging or the preparation race budget. |

Pins remain unchanged. Raw responses are in the local ignored build directory:
`.build/upstream-check-prefill.json` and `.build/prefill-audit/*-compare.json`.
The source audit uses the exact image snapshots, not general upstream vLLM.

## CUDA graphs and b12x preparation are different stages

Karmic auto-enables breakable graphs for `DeepseekV41ForCausalLM` on CUDA.
Its V4.1 state explicitly enables an **exact 4096-token, one-request piecewise
graph** with DCP1/PP1/DP1 and no LoRA. The checkpoint's compressed encoder / full
resolution decoder geometry activates this path. Prompt-logprob requests are
excluded. The current TP4/c16/K5/4096 profile meets these configuration gates.
The graph manager adds this graph separately from the ordinary decode capture
ladder. Therefore the smaller default decode limit is **not** proof that 4096
prefill lacks capture.
[Configuration](https://github.com/local-inference-lab/vllm/blob/1794dcf18454900263e0c66711af8ea4a1283ac1/vllm/config/vllm.py#L131),
[V4.1 eligibility](https://github.com/local-inference-lab/vllm/blob/1794dcf18454900263e0c66711af8ea4a1283ac1/vllm/models/deepseek_v4_1/nvidia/model_state.py#L80),
[extra graph descriptor](https://github.com/local-inference-lab/vllm/blob/1794dcf18454900263e0c66711af8ea4a1283ac1/vllm/v1/worker/gpu/cudagraph_utils.py#L596).

Consequences:

- Keep 4096 initially. Changing to 8192 disables that special graph at this
  pin and also grows preparation/activation storage. It is not a simple upgrade.
- Tail chunks and mixed requests need separate coverage assessment. Do not
  expand the full decode graph ladder to thousands of tokens to cover them.
- `b12x_autotune=false` disables candidate racing. Preparation and CUDA graph
  capture still run. A stalled `selecting norm.vision` with one fixed candidate
  is not evidence of a graph-capture or candidate-race failure.
- `forward_mqa` is an eager breakpoint within the prefill graph. It preserves
  active-prefix score width and the short-prefix indexer regime. Forcing all
  attention into one full graph can remove those dynamic bounds and increase
  work. Disk I/O also correctly stays outside capture; graphs consume stable
  prepared outputs. Keep the event/epoch synchronization.
  [Attention path](https://github.com/local-inference-lab/vllm/blob/1794dcf18454900263e0c66711af8ea4a1283ac1/vllm/models/deepseek_v4_1/attention.py#L1376).

The preparation race currently takes half of CUDA-reported free memory. More
critically, it checks resident bytes **after** materializing a candidate, and
keeps a champion and sometimes a carried candidate. That is a soft batching
budget, not a peak allocation cap. A 4 GiB race setting alone cannot guarantee
2 GiB physical headroom. Compiler processes also share the same RAM. Keep
autotuning disabled pending a specific failure trace; no speculative “capture
fix” or old reduced-tuning overlay has been enabled.
[Race budget](https://github.com/local-inference-lab/b12x/blob/a7d7d29b2ef8869086e0ceaa787321f17544e3c9/b12x/preparation/session.py#L328),
[candidate allocation order](https://github.com/local-inference-lab/b12x/blob/a7d7d29b2ef8869086e0ceaa787321f17544e3c9/b12x/preparation/session.py#L1211).

## Engram: cache scales first, establish locality before caching rows

`DiskRowCache` holds only batch-sized staging. The native reader deduplicates
and coalesces 4 KiB blocks **within** a transaction, but does not retain table
payload between transactions. Its O_DIRECT reads bypass the filesystem page
cache. Increasing `max_lookups`, queue depth, or spare OS page cache is not a
persistent hot-row cache. The default queue depth is already 128 per Engram
table, with 8 MiB native read buffers per table.
[Staging contract](https://github.com/local-inference-lab/b12x/blob/a7d7d29b2ef8869086e0ceaa787321f17544e3c9/b12x/sequence/_shared/disk_table.py#L128),
[native planner](https://github.com/local-inference-lab/b12x/blob/a7d7d29b2ef8869086e0ceaa787321f17544e3c9/b12x/loader/_row_plan.h).

Upstream already supports `disk_resident_scales=true`: retain exact eight-byte
E8M0 rows for both Engram layers, while the 256-byte FP8 weight rows remain on
SSD. No requantization or model-behavior change is intended. Because tiny scale
reads otherwise participate in aligned I/O, their time cost is not simply
8/264 of disk traffic. The actual reduction depends on block sharing and
coalescing; do not assume a 50% speedup.

For table sizes 384,006,168 and 384,016,682 and TP4, each rank retains
**1,536,045,704 bytes = 1.430554 GiB** of scales, plus staging/metadata. The full
local weight-plus-scale shard is **47.208 GiB**, which cannot fit alongside the
reported 82 GiB footprint and 2 GiB reserve in 121 GiB. These calculations use
ceil-row TP padding and the original checkpoint.
[Upstream scale implementation](https://github.com/local-inference-lab/b12x/blob/a7d7d29b2ef8869086e0ceaa787321f17544e3c9/b12x/sequence/engram/_disk.py#L24),
[checkpoint geometry and benchmark](https://github.com/local-inference-lab/b12x/blob/a7d7d29b2ef8869086e0ceaa787321f17544e3c9/docs/disk-embedding-backends.md).

A 2–4 GiB persistent cache would hold only roughly 4–8% of the packed shard
before metadata. It can still help skewed natural-language n-grams, but neither
uniform hashing nor a high intra-batch duplicate count establishes useful
cross-batch hits. Before implementing it, collect real n-gram/block access
traces and simulate hit rates for several budgets. A sound native design would
cache immutable file blocks by source identity/offset, filter hits before I/O
submission, preserve compact output order and existing stream ownership, and
charge payload plus tags/eviction metadata against the same physical budget.
It needs changing-query, eviction, invalid/nonlocal-ID, TP-shard, concurrent
table, and graph-replay correctness tests. None of that is replaced by raising
a staging limit. No hot-row cache has been added.

The current model already overlaps Engram work on a side stream with its first
layer (`VLLM_DS41_ENGRAM_OVERLAP` defaults on). No duplicate prefetch pipeline is
needed. GDS is also not an automatic Spark win: qualify the actual transport
and reject comparisons that silently use compatibility mode. Keep io_uring
for this experiment.
[Existing overlap](https://github.com/local-inference-lab/vllm/blob/1794dcf18454900263e0c66711af8ea4a1283ac1/vllm/models/deepseek_v4_1/nvidia/model.py#L456).

## Memory envelope per Spark

Treating the supplied 121/82/2 numbers as **GiB** (verify units on the hosts):

| Item | GiB per Spark |
| --- | ---: |
| Physical usable capacity, supplied | 121.000 |
| Post-weight-load footprint, supplied | 82.000 |
| Minimum headroom, reserved | 2.000 |
| Remaining for **all** KV, graphs, workspaces, compilation and further host growth | **37.000** |
| Optional resident scales | 1.431 |
| Remaining after those scales | **35.569** |

This is an envelope, **not 37 GiB available for graph capture alone**. Count any
RAM already included in the observed 82 only once. With utilization 0.88 and
121 GiB total, the nominal vLLM requested pool is 106.48 GiB: approximately
24.48 GiB above an 82 GiB footprint, before its actual profiling deductions.
The remaining 14.52 GiB is not evidence that all of it is idle physical RAM.
Do not set utilization to 119/121 merely to leave an apparent 2 GiB reserve.

Karmic already measures graph memory, subtracts late persistent allocations,
and uses host `MemAvailable` through psutil for integrated GPUs. Preserve auto
KV sizing and graph-memory estimation. KV, resident scales, model allocations,
compiler RSS, staging and graph pools compete for the same physical memory.
The native graph estimate samples full graphs, so compare estimated versus
actual capture memory and sample physical availability throughout startup.
[KV accounting](https://github.com/local-inference-lab/vllm/blob/1794dcf18454900263e0c66711af8ea4a1283ac1/vllm/v1/worker/gpu_worker.py#L660),
[NVIDIA shared-memory description](https://docs.nvidia.com/dgx/dgx-spark/system-overview.html).

`memory-watch.py` samples all four hosts at 0.5 s by default without allocating
a CUDA context. It refuses to start its foreground command if a node fails
the initial check, records samples and a per-rank minimum, and returns failure
for a sampled value below 2 GiB, swap I/O, or an incomplete sampler. It can wrap
startup or benchmarks. **It does not reserve RAM, catch every transient peak,
or stop serving containers on a breach.** It is a qualification gate; a passing
run is not an absolute OOM guarantee. Use a larger operational margin until
startup and sustained mixed traffic have been measured on every node.

## Other prefill candidates

| Candidate | Decision and reason |
| --- | --- |
| Bound prefix-hash copies | Prepared as a separate child image. Current Karmic still slices all remaining prompt hashes for each chunk. Only newly full blocks are used. |
| Share indexer page-table rows | Still a plausible kernel candidate; `_pages` repeats request rows per query. Mixed requests, padded/invalid pages, slot reuse and replay need GPU tests. No kernel change without a trace showing material cost. |
| Full-scan indexer | Its scoring grows with the visible compressed prefix, despite selecting only 512 outputs. Natural longer-context falloff remains possible. Later reindex/reuse layers already limit work. Do not change top-k/model semantics. |
| 8192-token chunks | Defer: loses the special 4096 graph, raises memory demand and can affect decode latency. |
| Smaller context cap | Can reduce cap-sized metadata/workspace, but changes the 1M serving contract. Only a separately approved capacity/performance comparison. |
| Engram projection TP | Upstream opt-in exists; trades per-rank compute for a BF16 gather over TP. Needs matched four-node profiling, not a default change. |
| DBO / extra prefill parallelism | Engram explicitly rejects DBO/microbatching at this pin. Keep the supported TP4/DCP1 topology. |

The hash patch now accepts the exact Karmic block-pool SHA256
`720215b0508dbab462063b2a21d9f7ae1fdbf0bc3041f66ec9e19c6e2dd018d3`, retaining
its historical guarded input as well. It bounds the slice at `num_full_blocks`.
Tests execute the actual pinned method before/after, including scaled hash
views, masked/null blocks, promotion and event calls. A 384K/4096 synthetic
case copies 1,536 instead of 74,496 hash references with identical recorded
cache operations. This is **48.5× fewer copied references, not 48.5× faster
prefill**. Apply/reapply/check/revert and drift rejection are tested. The base
Dockerfile and default image remain unchanged.

## Run the isolated comparisons on Spark 1

Copy the updated directory first. Use the actual operator config below;
`fleet.karmic.json` is an example filename. These launcher controls need no
image rebuild:

```json
{
  "engram_resident_scales": true,
  "graph_memory_debug": true,
  "b12x_preparation_trace": true,
  "b12x_hang_dump": true
}
```

Merge those keys into a copy of the actual config; this fragment is not a
complete fleet config. Start with `engram_resident_scales=false` for the
control, then true for the candidate. Keep image, K5, c16, OMP, utilization,
4096 chunk size, checkpoint and prompts identical. Each option rolls back
independently by setting it false or removing it. `torch_profile=true` is now
accepted on Karmic using its native profiler; enable it only for short trace
runs, not the throughput comparison.

With the previous service explicitly stopped, qualify startup:

```bash
python3 memory-watch.py --config fleet.karmic.json --output results/prefill-start-control -- \
  python3 fleet.py --config fleet.karmic.json start
```

Inspect all rank logs for automatic breakable graphs, the 4096-token PIECEWISE
capture (`graph_memory_debug` adds `[CG MEM]` records), and estimated/actual
graph pool sizes. Use a short worker trace to establish **replay during the
request**, not merely startup capture. The preparation JSONL identifies a
preparation stall; a capture traceback identifies a different problem. Preserve
both if startup fails. Do not infer failure from a long silent compile step.

Run matching prompts at roughly 8K/64K/128K, warm once, then at least five
measured repetitions per point, first c1 and then mixed traffic with decode.
Use a unique leading nonce to defeat prefix reuse, but keep the prompt body
matched between arms so Engram locality is comparable. Avoid prompt logprobs
in timed runs. The supplied benchmark generates synthetic repeated text; it
is useful for a controlled smoke comparison but cannot by itself qualify a
hot cache for production text. A monitored synthetic example is:

```bash
python3 memory-watch.py --config fleet.karmic.json --output results/prefill-128k-control-memory -- \
  python3 benchmark.py --config fleet.karmic.json --input-tokens 128519 \
  --max-tokens 128 --requests 5 --concurrency 1 --output results/prefill-128k-control.json
```

For Engram-only diagnosis, the pinned image already contains the upstream
benchmark at `/opt/b12x/benchmarks/benchmark_ngram_ssd.py`. Run in a separate
GPU-enabled container with the local checkpoint/cache mounts and io_uring
permissions, with serving stopped so the benchmark does not contend for memory.
Use `--models engram --backends io_uring --capacities 4096 --tokens 4096 --seqs 1
--max-seqs 16 --tp-size 4 --tp-rank <rank> --engram-token-bound`, the actual
`--engram-checkpoint` and representative `--text-file`. Compare with/without
`--engram-resident-scales`, saving distinct `--output` files. Repeat on all
four SSDs. Compare `read_bytes`, `read_calls`, native reader time, ID/sync time
and total transaction time. These counters describe individual transactions;
do not treat them as lifetime cache-hit counters. Run the upstream disk/Engram
tests described in its storage document, then fleet numerical smoke checks
and changing-request graph replay before promoting the option.

Test the bounded-hash image separately from resident scales. On an ARM64 Spark,
set `BASE_IMAGE` to the exact control image and `HASH_IMAGE` to a new tag:

```bash
python3 check-upstream.py --output .build/upstream-check-prefill-build.json
docker build -f Dockerfile.prefill-hashes --build-arg BASE_IMAGE="$BASE_IMAGE" -t "$HASH_IMAGE" .
docker run --rm --gpus all --entrypoint python3 "$HASH_IMAGE" /opt/ds41/image-check.py --gpu
```

Select that new image in a copy of the control config, distribute it with
`fleet.py share`, then use the same preflight/start/smoke, memory and benchmark
gates. Retain the original image for rollback. Do not modify source in a live
container. The hash patch's `--revert` is also available for a separate rollback
image. Promote only if numerical checks pass, each node retains the required
headroom with no swap activity, repeated TTFT improves, and decode does not
regress. Otherwise retain the original profile.

## Validation here

The eight new control/memory tests, 31 existing deployment tests, four
historical/Karmic hash tests and existing build-tag test pass. Python compilation
and whitespace checks pass. The broad suite also exercised unrelated historical
patches; two tests require PyTorch, which is absent on this Windows Python.
The initial Git subprocess permission failure was resolved by running that
temporary-repository test with the necessary permission. Five display-KV fleet
tests require recorded GPU experiments and remain skipped. Linux ARM64 image,
native Engram, real SSH sampler, capture/replay and fleet performance checks
remain outstanding. Existing user edits to the K5 profile, startup diagnostics
and log-following behavior were preserved.
