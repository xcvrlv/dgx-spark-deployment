# Bounded Karmic autotuning on Spark

The [latest performance bundle](SPARKRING-PERFORMANCE.md) includes bounded tuning
and the optional 8192 graph in one image. For that image use
`performance-settings.py --autotune --batch-tokens 8192`; the separate child-image
builders below apply to the saved 2026-09-25 pins. The measured capacity tradeoff
and operator investigation below retain their original context.

## Batch size and graph coverage

The launcher passes `max_num_batched_tokens` to vLLM. The live container inspected
on 2026-09-27 had **8192**, a **1,048,576** model limit and utilization **0.86**.
Those are not the earlier 262k-context benchmark settings. The token budget is
a scheduling ceiling, not a promise that every iteration contains that many
tokens.

At pinned vLLM `1794dcf`, `DeepseekV41ModelState` enables the single-request
long-prefill PIECEWISE graph only when `self.max_num_tokens == 4096` (and the
other parallelism/LoRA conditions hold). An 8192 setting removes that special
capture, even if some individual chunks are smaller. Short-batch/decode graphs
and the separate DSpark 128-row context graph can still be present. Consequently,
doubling the scheduling budget can exchange fewer chunks for more eager work.
It need not improve measured throughput. Extending this condition to 8192
requires its own source, memory and serving checks, described below.

The initial tuning experiment explicitly used **4096** to retain the proven prefill
graph. It preserves the source recipe's context limit, memory utilization, K,
resident Engram scales, paths and network. Its base image does not enable an
8192 graph; the optional follow-up child does.

## Source audit

Live heads were checked **2026-09-27 20:04:57 UTC** before editing:

| Source | Image pin | Checked head |
| --- | --- | --- |
| vLLM Karmic | `1794dcf18454900263e0c66711af8ea4a1283ac1` | `953a636d3ae1fde86a02874fc919a95091c95275` |
| vLLM Jovian | Not the serving base | `8e1f1e587f8d24faf606f334a1c4bdaaa6bd4368` |
| b12x | `a7d7d29b2ef8869086e0ceaa787321f17544e3c9` | `d44247b6171f7c2f9787341ae884b537887d7df9` |

The existing comparisons show one Karmic EXL3-preparation commit and two b12x
trellis/EXL3 and PCIe commits. They do not replace these preparation limits or
extend this prefill graph. Pins are unchanged. Evidence is in
`.build/upstream-check-autotune.json` and `.build/prefill-audit/*-compare.json`.
The heads were rechecked at **20:44:41 UTC** before the memory-counter fix and
were identical (`.build/upstream-check-autotune-counter.json`).
Jovian and b12x were rechecked at **22:04:07 UTC** before adjusting the host
memory watchdog, with the same heads (`.build/upstream-check-memory-guard.json`).
Karmic, Jovian and b12x were rechecked at **22:29:03 UTC** before the 8192
graph patch, again with the same heads (`.build/upstream-check-8192-graph.json`).
They were checked again at **23:27:11 UTC** before adding the serving replay
marker, with unchanged heads (`.build/upstream-check-8192-replay.json`).
The audited Karmic model state still has only the 4096-token gate; these
upstream heads do not supersede the candidate without a separate rebase.

## What changes

`Dockerfile.autotune` is a child of the operator's actual image. It preserves
the existing RoCE, bounded-hash and DSpark patches. The new patch validates the
exact vLLM preparation and b12x session files before any write, supports check
and revert, and installs a small preparation-only helper. It is independently
disabled by `DS41_B12X_BOUNDED_AUTOTUNE=0`.

The v2 image also hash-checks `b12x/preparation/_memory.py` and builds its native
allocator counter against the image's Torch/CUDA libraries during Docker build.
Bounded mode imports that binary directly. On this fleet, captured worker stacks
were blocked in `torch.utils.file_baton.FileBaton.wait` while loading this counter;
all four persistent extension directories had stale `lock` files dating back
days. The image-built import bypasses those locks without clearing user caches.
Before weight loading, each rank must confirm that the helper reports the same
allocated bytes as Torch for an actual GPU allocation, under a 90-second timeout.

With the bounded recipe:

- Autotuning is explicitly enabled in both vLLM kernel config and `B12X_AUTOTUNE`.
- Each Spark uses **one compiler worker** in weights, state and bind stages.
- A race batch admits one new candidate alongside the previous champion.
- The nominal candidate-residency budget is **1 GiB**, further reduced using
  current CUDA free memory and host MemAvailable above a **4 GiB reserve**.
- Preparation and candidate admission check that physical reserve. Candidate
  memory is measured after allocation upstream; this is not a hard allocator
  quota. A single candidate can exceed the race budget.
- The candidate set and upstream sampling/round defaults are retained. The
  changed race batching can change selection results; a separate selection-cache
  namespace keeps this experiment distinct while reusing compiled artifacts.
- Per-worker progress records include request, phase, compiled/measured counts
  and completion totals. Clock-only heartbeats do not reset the stall timer.

The earlier `selecting norm.vision` observation occurred before candidate racing:
that family has a fixed configuration. A race-memory cap alone cannot explain
or fix that stall. The external runner detects lack of meaningful progress,
requests worker stack dumps when a worker PID is available, stops candidate
containers and retains their logs. It does not quietly finish with untuned
heuristics and report success.

## Build and qualification

Run from the deployment directory on Spark 1. Use the actual current source
recipe filename. Existing recipe and output files are preserved.

```bash
python3 build-autotune.py --from-config fleet.dspark-prefill.json \
  --output fleet.autotune-v2.json --batch-tokens 4096 --share
```

The new recipe is published only after a successful build. The builder checks
parent pin labels; the Docker build checks source hashes. Fleet preflight
requires the new overlay label and identical images on every rank.

Stop the working service only after the image has been built and distributed:

```bash
python3 fleet.py --config fleet.dspark-prefill.json stop &&
python3 run-autotune.py --config fleet.autotune-v2.json \
  --output .build/autotune-new-run
```

The runner refuses to replace existing containers. It samples all four hosts
every 0.5 seconds throughout startup, the existing c16 smoke test, and three
128,519-token requests with unique prefixes. It aborts on observed MemAvailable
below **3 GiB**, failed/stale monitoring, worker failures,
15 minutes without meaningful fleet progress, or two hours total. Progress on
any rank counts, since other ranks may legitimately await its tuning shard.
The in-process 4 GiB reserve adds a margin above the operator's 2 GiB requirement;
sampling and stop latency cannot guarantee that every short-lived peak is caught.

Failure stops only containers matching the candidate image, preserves those
containers and logs, and returns nonzero. Success requires completed startup,
smoke and long requests, measured/cached preparation evidence, and all sampled
hosts meeting the **2 GiB minimum**. Host-wide swap reads and writes are reported
separately for startup and preparation; they cannot identify which process was
paged or prove physical-memory exhaustion. An initial 16 MiB write guard stopped
the first 0.87 restart after about 43 MiB of writes despite over 31 GiB available
on every host. That false-positive guard was removed; physical-headroom and
monitoring guards remain. Paging may still affect performance and is retained
in the raw samples and report. This qualifies the
tested workload, not every possible 1M/concurrent workload or a throughput gain.

Results include `summary.json`, `latest-progress.json`, per-rank memory samples,
startup/benchmark output and container logs. Native preparation traces remain
under each node's cache in `b12x-preparation-trace`. Repeated launches with the
same tuning profile can reuse completed selections; do not delete working
caches to force a cold run.

To roll back after a failed experiment, collect the retained logs, remove the
stopped candidate containers with the fleet stop action, then launch the saved
working recipe. Its image and configuration are not overwritten by the build.

## Validation

Twelve new local checks cover hash-guarded apply/check/revert, drift rejection
before writes, memory budget/reserve arithmetic, disabled-mode behavior,
explicit flag propagation, failed-build publication, meaningful progress and
transient memory detection and separate paging accounting. The 31 deployment
and eight existing memory checks also pass (51 checks total).

## Completed cold qualification

The four-node run completed at **2026-09-27 21:41 UTC**, using
`fleet.autotune-v2.json`, image
`sha256:c0219a450da0822aaf2cb53894ebc5e3e1137be9e969cee781445a8da7ed3b5d`,
4096 batched tokens, a 1M model limit, utilization **0.86**, K5 and OMP8.
It passed native allocator checks, fabric qualification, startup, c16 finite-
logprob/chat smoke, and three 128k requests. The total observed interval was
about 48 minutes, including the full cold search.

| Rank | Measured candidates | Completed preparation jobs | Lowest MemAvailable | Swap writes after preparation observed |
| --- | ---: | ---: | ---: | ---: |
| 0 | 62,066 | 5 | 7.72 GiB | 4.77 MiB |
| 1 | 61,872 | 5 | 10.93 GiB | 6.86 MiB |
| 2 | 60,865 | 5 | 10.06 GiB | 2.25 MiB |
| 3 | 60,727 | 5 | 10.82 GiB | 0.27 MiB |

Every rank exceeded the required 2 GiB sampled headroom. The graph reports were
3.37 GiB during profiling and 2.84 GiB at actual capture; serving KV capacity was
13,019,698 tokens. Both DSpark prefill execution markers were observed.
Evidence is in `.build/autotune-run-4` on Spark 1 and the local
`.build/qualified-autotune-20260927/summary.json`.

The first user prefill comparison overlapped the qualification benchmark, which
continued until **00:41:06 Helsinki time on September 28**. The engine logged
concurrent/waiting requests alongside the 64k completion. Those timings do not
isolate prefill throughput. An idle-server repeat with the repository's synthetic
prompt produced 8k TTFTs of 1.725/1.697 s, 64k TTFTs of 14.374/14.049 s, and
128k TTFTs of 35.419/30.643 s. This removes the large reported falloff on that
workload; different prompt content still needs a matched comparison. These
repeats are not an OMP A/B or a direct speedup claim against the user's corpus.

## Requested 0.87 validation and matched benchmark

At utilization **0.87**, 4096 batched tokens, K5 and OMP8, the guarded restart
passed on all four nodes at **22:17:42 UTC** on September 27. Minimum sampled
host memory was **5.07 / 8.16 / 7.95 / 8.24 GiB** by rank, above the required
2 GiB. vLLM reported **20,251,925 KV tokens**. PIECEWISE 4096-token and DSpark
context graphs were captured, and both DSpark prefill runtime markers appeared.
Evidence is in `.build/autotune-087-omp8-retry` on Spark 1.

The operator's actual `llm_decode_bench.py` v0.7.3 was then run twice on an idle
server with the original `--contexts 0` and its default integrated prefill scouts.
Only decode concurrency was limited to 1 and 16, at 30 seconds each. Its 8k
JIT warm-up ran each time. Results from `.build/operator-omp8`:

| Context | Prefill run 1 | Prefill run 2 |
| --- | ---: | ---: |
| 8k | 4,391 tok/s | 4,356 tok/s |
| 64k | 4,218 tok/s | 4,212 tok/s |
| 128k | 3,996 tok/s | 3,988 tok/s |

Decode C1 was 55.25/53.83 aggregate tok/s and C16 was 266.73/263.63. Both
benchmark windows passed physical memory checks with no swap writes. The earlier
1,398/2,089 long-prefill screenshot coincided with concurrent qualification
requests; the repeated same-tool measurements no longer show that falloff.

With the same image, 0.87 utilization and 4096-token batch setting, OMP1 also
passed the guarded startup and two benchmark repeats. Its 8k prefill was
4,367/4,353, 64k 4,207/4,199, and 128k 3,955/3,975 tok/s. C1 decode was
55.80/57.80 and C16 decode 266.68/268.79 tok/s. Compared with OMP8 above,
the prefill differences were under about 1%, C16 varied around 1%, and C1
favored OMP1 in these short runs. That is too little evidence to claim a stable
throughput gain. OMP8 is retained for the prefill-focused 8192 trial; testing
OMP2 or OMP4 would add lengthy fleet restarts without a clear trend from the
endpoints. OMP1 evidence is in `.build/autotune-087-omp1` and
`.build/operator-omp1` on Spark 1.

## Opt-in 8192-token graph candidate

The pinned model state currently allows an exact single-request PIECEWISE
prefill graph only at a 4096-token scheduler ceiling. The child image in
`Dockerfile.prefill-8192` changes that gate for 8192 only when
`DS41_PREFILL_8192_GRAPH=1`; the source hash is checked at build and the switch
is independent of the bounded autotune and DSpark switches. With the switch off,
the original 4096 behavior remains. `build-prefill-8192.py` derives the
candidate from the qualified 4096 recipe, retains its utilization and OMP
setting, changes the batch ceiling to 8192, and publishes a new recipe only
after a successful build.

The batch-budget audit follows the recipe through `fleet.serve_args`, the
scheduler config, runner/input/block-table capacities, b12x preparation,
attention/compressor/Engram state and DSpark context capacity. These read
`scheduler_config.max_num_batched_tokens`; the only literal 4096 gate found
in these model and DSpark paths was the exact prefill-graph condition. Live
preparation on every rank reported `m8192` workloads, including DSpark context
KV and MLA extend. The separate compact DSpark graph remains 128 rows because
it processes the compacted context; decode graphs also retain their short
shapes. Neither means the target's 8192-token scheduling ceiling was ignored.

For this candidate, `run-autotune.py` checks the actual container command and
environment on all four ranks, requires an observed `PIECEWISE tokens=8192
reqs=1` capture on rank 0 (the only rank emitting per-graph debug accounting)
and model-state eligibility at 8192 on every rank. The v2 child also records
`DS41 serving 8192-token PIECEWISE graph replay active` from the actual runner
PIECEWISE branch, only when `dummy_run` is false. Qualification requires that
marker on every rank after the long-context requests. This separates real
serving from dummy warm-up. The public iteration-token histogram is retained
as auxiliary observations only: prompt statistics can accumulate across
prefill chunks before the engine reports an output, so those buckets do not
establish the size of a GPU step. It keeps
the same RAM, startup, c16 and long-context request checks. Missing route
evidence fails qualification. Compare its prefill and decode with the same
operator benchmark before adopting it.

The first 8192 trial completed startup and three 128k requests, with lowest
sampled RAM **5.19 / 7.86 / 7.51 / 8.30 GiB** by rank. Its proof check incorrectly
required per-graph log messages on secondary ranks and rejected the trial,
triggering rollback. Secondary ranks emit runtime eligibility but not per-graph
debug accounting. The check was corrected and covered by a regression test;
the original failed report remains in `.build/autotune-b8192-1` rather than being
rewritten as a pass. The image is unchanged for the corrected retry.

Four additional local tests check the exact patch hash/rollback, the opt-in
condition at 4096/8192/other limits, all-rank launch propagation, and rejection
of missing model-state eligibility, rank-0 capture, or real serving replay.
Both patched source files are validated before either is written; rollback
restores the existing DSpark overlay. The replay condition excludes dummy
runs and other token counts. Together with the earlier checks, 55 relevant
local tests pass.

The corrected v1 qualification passed (`.build/autotune-b8192-2`), with
minimum sampled available RAM **4.36 / 7.44 / 7.14 / 7.37 GiB** by rank.
Two original-tool benchmark repeats in `.build/operator-b8192-2` measured:

| Context | 4096 mean tok/s | 8192 run 1 | 8192 run 2 | Gain over 4096 |
| --- | ---: | ---: | ---: | ---: |
| 8k | 4,373.5 | 4,678 | 4,677 | 7.0% |
| 64k | 4,215 | 4,481 | 4,457 | 6.0% |
| 128k | 3,992 | 4,233 | 4,227 | 6.0% |

C1 decode was **58.04 / 56.32 tok/s**, compared with **55.25 / 53.83**
at 4096; C16 was **268.84 / 266.01**, compared with **266.73 / 263.63**.
These are two 30-second measurements per cell, sufficient to detect a large
regression but not to establish a small lasting decode gain. The benchmark
window had no swap writes and at least **4.92 GiB** available RAM.

KV capacity in that v1 run was **15,535,987 tokens**, compared with **20,251,925**
at 4096. That is about **23% less KV capacity** at the same 0.87 utilization.
The throughput measurements used zero-context decode, so they do not establish
equal decode behavior when that smaller cache is full. Sixteen requests each
near the configured one-million-token limit would exceed the 8192 recipe's
reported capacity. Keep `fleet.autotune-087-omp8.json` as the qualified
4096 fallback for workloads that prioritize KV capacity.

## Final v2 serving qualification

The final image adds the hash-guarded non-dummy replay marker to the existing
DSpark-patched model runner. Its image ID is
`sha256:6e922330e5f75c7bfe4d68d928fcaffadfb1b347e65c560344f39fcb9a953adf`,
identical on all four Sparks. The recipe is `fleet.autotune-b8192-v2.json`:
8192 batch ceiling, 0.87 utilization, OMP8, K5, 16 sequences, 1M model limit,
bounded autotuning, resident Engram scales and both DSpark prefill switches.

Qualification completed at **2026-09-27 23:53:12 UTC** and passed the native
GPU/allocator and fabric checks, c16 finite-logprob/chat smoke, three 128k
requests, and real serving graph replay on every rank. Every container had
the 8192 command limit and enabled graph flag. All four model runners emitted
the non-dummy replay marker. Evidence is in `.build/autotune-b8192-v2`.

Minimum sampled available RAM was **4.65 / 7.48 / 7.45 / 7.78 GiB** by rank,
above the required 2 GiB. Host-wide swap writes after the preparation monitor
was armed were about **37.8 / 9.3 / 31.8 / 4.1 MiB**. These are reported
separately from physical headroom and do not identify which process was paged.
The final serving KV capacity is **15,352,352 tokens** (14.64 times the 1M
request limit), about **24% below** the qualified 4096 recipe's 20,251,925.
Profiling and final graph capture reported 1.48 and 2.29 GiB respectively;
the 8192 graph's individual debug pool delta was 1948 MiB.

Two matched repeats on this exact final image then passed in
`.build/operator-b8192-v2`:

| Context | 4096 mean tok/s | Final 8192 run 1 | Final 8192 run 2 | Mean gain |
| --- | ---: | ---: | ---: | ---: |
| 8k | 4,373.5 | 4,665 | 4,665 | 6.7% |
| 64k | 4,215 | 4,474 | 4,468 | 6.1% |
| 128k | 3,992 | 4,229 | 4,216 | 5.8% |

Zero-context C1 decode was **57.56 / 59.12 tok/s**; C16 was
**263.02 / 268.55**. The two-run C16 mean was **265.78**, compared with
**265.18** at 4096. There is no meaningful decode regression in these short
measurements, and no basis to claim a small stable gain. Longer runs and
decode near KV capacity remain outside this qualification.

Benchmark minimum available RAM was **5.16 / 7.74 / 7.55 / 7.80 GiB** by
rank, with **no swap writes** during the two-run window. Both runs began and
ended with zero running/waiting requests. These results confirm the earlier
v1 prefill gains while adding direct serving graph proof on every rank.

The fleet is left running this final image. On Spark 1,
`fleet.autotune-v2.json` now contains the same qualified recipe as
`fleet.autotune-b8192-v2.json`. Its earlier 4096/0.86 contents were saved as
`.build/fleet.autotune-v2-before-8192-20260928.json`. The qualified 4096/0.87
fallback remains `fleet.autotune-087-omp8.json`.

For a later restart from `/home/juho/Documents/juhon-deploy/ds41-vllm`:

```bash
python3 fleet.py --config fleet.autotune-v2.json stop &&
python3 fleet.py --config fleet.autotune-v2.json start
```

To choose the larger-KV 4096 fallback instead:

```bash
python3 fleet.py --config fleet.autotune-v2.json stop &&
python3 fleet.py --config fleet.autotune-087-omp8.json start
```

The final child can be rebuilt from the saved qualified 4096 recipe, using
new image/output names to preserve the installed image and existing recipes:

```bash
python3 build-prefill-8192.py --from-config fleet.autotune-087-omp8.json \
  --image spark-vllm-ds41:prefill-8192-rebuild \
  --output fleet.prefill-8192-rebuild.json --share
```

Local evidence is archived under `.build/qualified-prefill-20260928` with
the final qualification, both original-tool benchmark repeats, RAM samples,
and exact candidate/fallback recipes. This is sampled workload qualification,
not a guarantee of 2 GiB headroom under every possible serving load.
