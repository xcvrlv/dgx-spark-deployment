Current build/defaults: see [latest JJ update](UPSTREAM-R37.md). Older profiles and audit snapshots below are historical.

# c16 communication candidate — 2026-09-13

The new default is **16 sequences, 393216 tokens (384 × 1024), GPU memory
utilization 0.88**, TP4/DCP1 and 4096-token prefill batches. The original
FP8-Engram checkpoint is read directly from each node's local SSD. The existing
directory name containing `FP4-Engram` is accepted; contents are still checked.
These are requested test limits, not measured capacity. Sixteen simultaneous
384K requests are not guaranteed to fit. `cluster-c8.json` remains available
with the older image as a rollback profile.

## Upstream check and pins

Checked before this patch:

| Repository | Previous pin | New pin | Relevant change |
| --- | --- | --- | --- |
| [JJ](https://github.com/local-inference-lab/vllm/compare/35601be19df6be8be33f05aa482139e4b0a12bff...b40673cd006bf3496fdd70361dad2ad29eff54e7) | `35601be` | `b40673cd006bf3496fdd70361dad2ad29eff54e7` | Fused full-vocabulary greedy DSpark Markov reduction, with incompatible-buffer rejection. Applies when testing greedy DSpark, not the initial target-only run. |
| [b12x](https://github.com/local-inference-lab/b12x/compare/323107ff948ca532f1f7c793b4b550c30ba5212b...9043b448622764a598969518d413b3fd8b3c0c07) | `323107f` | `9043b448622764a598969518d413b3fd8b3c0c07` | V4.1 TP3 prefill head support; not an expected TP4 speedup. |

Neither update contains the old local RoCEnante proxy changes. The latest
`_roce_proxy.c` and `roce_oneshot.py` have **exactly the original GLM v15/v16
input hashes**. Porting the four communication transformations also reproduces
the original output hashes. No broad GLM model overlay is imported.

`AGENTS.md` records the requirement to check upstream before future patches.
`check-upstream.py` runs before every image build and writes
`.build/upstream-check.json`. A different upstream head is reported, not
silently substituted for the audited pin. A failed network check fails the
build script; it must not be represented as an up-to-date audit. Every source
patch verifies exact input/output hashes and refuses unknown source drift.

New image: `spark-vllm-ds41:jj-b40673c-b12x-9043b44-roce-v1-sm121`.
Advancing JJ requires a source rebuild for this version. The ARM64 builder
digest from the working build is retained.

## Communication changes included

| Port | Why retain it | Independent recipe switch |
| --- | --- | --- |
| v15 inline tiny payloads | Requests 64 bytes of inline capacity, retries at 16, respects the provider-returned limit. Avoids a NIC fetch for eligible tiny stripes; ordinary activation messages are usually larger. | `inline_payload` |
| v15 rotated peer posting | Starts each rank's asynchronous posts at a different destination on the switch. Preserves reduction order and peer coverage. | `balanced_fanout` |
| v16 skip empty send-CQ polling | Tracks pending signaled completions per HCA; avoids provider calls when nothing can complete. Polls while pending, including backpressure. Errors remain fatal. | `skip_empty_cq` |
| v16 initialize protocol state only | Clears flags/control and retains separately zeroed GPU counters; skips payload initialization that later writes replace. Saves startup CPU writes, not allocation or steady-state bytes. | `lazy_payload_init` |

These booleans live in `roce_optimizations` in the fleet JSON. All four are
enabled in the new candidate. Missing entries are false, preserving old
profiles; set one false and restart every rank to isolate it. Profiles with
any switch enabled require the `ds41-roce-v1` image label in preflight.

This is the strongest source-supported carryover, but the previous repository
does not contain isolated fleet speedups proving these four optimizations.
The payload-before-flag work-request chain, signaled completion accounting,
sequence/slot protocol, GPU fences, health checks and fixed reduction order
are preserved. Dual-rail striping, small-gather handling and Gloo setup already
exist upstream and are not patched again. NCCL remains the large-message path.

Build-time C tests include the **actual patched C file** with mocked verbs
calls. They check empty/pending/error CQs, both posting orders, provider inline
limits, pending counts, stripe bytes and payload-before-signaled-flag chains.
They do not require an HCA. The existing four-node GPU qualification still
checks both rail counters, NCCL numerical agreement and graph replay. It must
pass for each A/B configuration; do not bypass it based on an import test.

## Prefill and SM121 decode findings

* **GLM full-CKV/DCP patches are not direct ports.** This recipe runs DCP1.
  V4.1 uses compressed caches, its own metadata and a candidate-reindex path;
  importing GLM all-gather workspaces would change unrelated execution paths.
* **The old indexer batching idea partly exists already.** JJ's V4.1 indexer
  uses a bounded 256-row prefill chunk and ranks replicated heads locally;
  decode uses 64-row chunks. It already narrows the score width to active
  context on eligible non-captured prefill. Blindly raising the indexer chunk
  increases context-scaled score workspace and is a poor first change at 0.88.
* **Larger scheduler batches are a useful isolated experiment.** The helper's
  `--prefill8192` profile doubles the token budget from 4096 to 8192. It may
  amortize dense/MoE execution and collectives, but increases planned buffers,
  graph/prefill memory and potentially decode latency. This does not double the
  indexer chunk. It is an optional profile, not a claimed kernel speedup.
* **Disk Engram can dominate prefill.** Current JJ reads native FP8 table rows
  from the checkpoint via b12x/io_uring. Keep files local, and compare SSD I/O,
  TTFT and model throughput together. The old row-pack/FP4 readers do not apply.
* **EXL3 scheduling/rotation/FC1-tail/FC2-pair patches do not transfer to the
  native MXFP4 path.** Their archived benchmarks concern mixed-K EXL3 kernels.
  No evidence justifies changing native expert tile schedules from those results.
* **Use the new upstream greedy DSpark kernel before inventing a similar
  patch.** Target-only is still the base. Test draft depth 3 separately at c16;
  graph capture then spans 1–64 verification tokens. Acceptance, scratch and KV
  memory costs need observation. This upstream fusion is not an assertion of
  reduced inter-node vocabulary traffic or an SM121-specific measured speedup.

No larger custom prefill/decode kernel patch is promoted without a trace or
GPU correctness comparison. The communication port plus isolated prefill
batching profile makes the next measurements actionable without attributing
an upstream update or memory-capacity change to a local kernel optimization.

## Build, migrate and launch

From `ds41-vllm` on Spark 1, after copying the updated directory:

```bash
bash build-image.sh
python3 configure-c16.py --from-config .build/cluster-roce.json --output .build/cluster-c16.json
python3 fleet.py --config .build/cluster-c16.json share
python3 fleet.py --config .build/cluster-roce.json stop
python3 fleet.py --config .build/cluster-c16.json start
```

The helper preserves model directory, flat-layout setting, revision, cache,
SSH identity/user and draft depth from the source configuration. It updates
the new image, c16/384K/0.88 limits, RoCEnante switches and 4096 batch budget,
and uses CX0 addresses for SSH. It refuses to overwrite its source file.
For a fresh install, omit `--from-config` to use the checked-in c16 defaults.

## Same-image A/B plan

Create controls that retain the same image and memory limits:

```bash
python3 configure-c16.py --from-config .build/cluster-c16.json --control --output .build/cluster-c16-control.json
python3 configure-c16.py --from-config .build/cluster-c16.json --prefill8192 --output .build/cluster-c16-prefill8192.json
```

Stop/start between profiles. First compare communication off/on at batch 4096;
then compare 4096/8192 with communication on. Do not change draft depth, model
context, memory utilization or host cache state during an individual comparison.
To test 0.88 itself, clone a configuration and change **only** that value back
to 0.80. A startup allocation failure at 0.88 is a failed capacity trial, not
a reason to weaken correctness checks.

For each running profile, use matching `--config` and distinct output names:

```bash
python3 benchmark.py --config .build/cluster-c16.json --input-tokens 8192 --concurrency 1 --requests 3 --output results/comm-on-prefill8k.json
python3 benchmark.py --config .build/cluster-c16.json --input-tokens 65536 --concurrency 1 --requests 3 --output results/comm-on-prefill64k.json
python3 benchmark.py --config .build/cluster-c16.json --input-tokens 8192 --concurrency 16 --requests 16 --output results/comm-on-c16.json
```

The benchmark puts a unique nonce at the start of every prompt to prevent
prefix-cache reuse, tokenizes before timing, records streaming TTFT, request
latency and output rate, and saves the complete configuration and image ID.
TTFT includes scheduling and first-token generation; it is not pure kernel
prefill time. Full token prompts fit the configured context cap. Compare warm
repeats and collect GPU/host memory and SSD activity separately; the benchmark
does not flush OS caches or modify host settings. Ramp sustained concurrency
and long-context workloads after short checks pass.

## Validation status

CPU tests cover real pinned-source application/reapplication and drift
rejection, poisoned-payload versus zeroed-protocol initialization, independent
switches, c16 graph limits, flat checkpoint paths and the previous failure
regressions. Native proxy tests run during the image build; RDMA and graph
tests run on `start`. This Windows workspace cannot execute those ARM64/GPU
checks. No speedup, 0.88 stability, or c16 long-context capacity is claimed yet.
