# DSpark prefill overlay — 2026-09-27

The [latest performance bundle](SPARKRING-PERFORMANCE.md) includes both switches
in one image on the 2026-09-28 pins. The child-image commands below apply to the
saved 2026-09-25 source pair; use `performance-settings.py` for the new image.

This implements two opt-in changes for the existing Karmic TP4/c16/K5 recipe:

- **Skip unused drafts on intermediate prompt chunks.** Context KV insertion
  still runs. Only the subsequent draft backbone, Markov sampling and confidence
  publication are skipped. The final prompt chunk and any batch containing a
  decoding request keep the normal path. The predicate uses pre-step CPU prompt
  counters; it does not synchronize with the GPU. Structured output, profiling,
  restored context and unsupported parallel configurations take the old path.
- **Capture compact context preparation.** A single-request CED prefill packs
  128 retained rows and replays projection, normalization, rotary and KV insertion.
  Original-coordinate query anchors and rejection masks are prepared first;
  packing touches separate context buffers. Invalid slots remain PAD. Mixed or
  larger context batches retain eager preparation. All original decode graph
  capacities are preserved; at most one 128-row capacity is added.

The switches are independent: `dspark_skip_prefill_draft` and
`dspark_compact_context_graph`. Both default to false in the launcher and image.
The configuration helper enables both in a **new** recipe and enables graph
memory diagnostics. It preserves the operator's model paths, network, Engram
resident-scale setting, K, batch size, utilization and context limit.

The new child image also includes the previous bounded-hash fix. Both patches
accept the exact pinned source hashes, validate every source before writing,
support `--check` and `--revert`, and reject source drift. Fleet preflight requires
the new image label when either optimization is enabled. Image pins are unchanged.

## Upstream audit

A live check completed **2026-09-27 17:35:25 UTC**, before these edits:

| Repository | Image pin | Checked head |
| --- | --- | --- |
| vLLM Karmic | `1794dcf18454900263e0c66711af8ea4a1283ac1` | `953a636d3ae1fde86a02874fc919a95091c95275` |
| vLLM Jovian | Not the serving base | `8e1f1e587f8d24faf606f334a1c4bdaaa6bd4368` |
| b12x | `a7d7d29b2ef8869086e0ceaa787321f17544e3c9` | `d44247b6171f7c2f9787341ae884b537887d7df9` |

The intervening Karmic EXL3-preparation commit and the b12x trellis/EXL3 and PCIe
changes do not supersede these DSpark changes. Raw evidence is in the ignored
`.build/upstream-check-dspark-prefill.json` and `.build/prefill-audit/*-compare.json`.
See the [previous audit](PREFILL-SPARK-AUDIT.md) for linked upstream comparisons
and the separate Engram-scale investigation.

The build-helper correction was preceded by another live check at
**2026-09-27 18:08:03 UTC**. All three heads above were unchanged, so the previous
pin comparison still applies. This correction handles local recipe discovery
and build sequencing; upstream does not replace that deployment tooling.
Evidence: `.build/upstream-check-dspark-build.json`.

The read-only diagnostic addition was preceded by a live check at
**2026-09-27 19:03:17 UTC**. All three heads above were unchanged; the comparison
still applies and upstream does not supersede this deployment diagnostic.
Evidence: `.build/upstream-check-prefill-diagnostics.json`. Serving pins and
runtime patches were not changed during this follow-up.

## Memory and validation status

For c16/K5, the existing context limit is 96 rows. Extending its three auxiliary
inputs at width 5120 in BF16 to 128 rows adds **983,040 bytes (0.9375 MiB)** to that
buffer. Hidden states, positions and context-slot buffers are reused. This is
**not** the total extra graph cost: native scratch, captured outputs and CUDA
bookkeeping must also fit.

The extra graph is created in the existing speculator capture hook, which is
included in startup graph-memory profiling before KV sizing. Context graphs are
explicitly closed during profiling teardown. No graph is captured lazily on the
first live request, and no workspace lock is relaxed. This does not enlarge the
4096-token target graph or the decode capture ladder.

At the reported 121 GiB total and 82 GiB after weights, the upper envelope for
all remaining allocations is about **37 GiB** after reserving 2 GiB. If those
82 GiB exclude resident Engram scales, subtract their additional 1.43 GiB too.
Retain the current utilization and automatic KV sizing. Qualification must show
at least **2 GiB host MemAvailable on every Spark**, including preparation,
capture and long/mixed requests, with no new swap activity. The watcher fails
qualification on a breach; it does not reserve RAM or stop containers. A passing
sampled run is evidence for that workload, not a hard guarantee for all loads.

Local validation executes the actual patched `propose` method with recording
CPU dependencies: intermediate/final boundaries, cached prefixes, mixed batches,
PAD and per-layer slots, context-before-skip ordering, query-buffer preservation,
restored context, flag combinations, hash round trips and drift rejection.
All **56 targeted local checks** pass: 10 DSpark overlay checks, 8 prefill
controls/headroom checks, 31 deployment checks, 4 bounded-hash checks and
3 Karmic RoCE checks. Python syntax and patch round-trip validation pass too.
The build-helper fix adds four passing checks for recipe discovery, missing or
invalid inputs, failed image inspection/builds, and successful build/distribution
ordering. These tests mock Docker; they do not establish image-build success.

The child image carries **native GPU checks** derived from the pinned upstream
suite: FP8 projections at widths 128 and 5120, rotary and cache writes, exact KV
comparison, multi-group slots, 128-to-decode replay transitions, stale padding,
restored-cache no-op and real CED/prepare-inputs kernels under all four switch
combinations. These tests require SM12x and fail qualification on a wrong host.
They have not been run here. SSH to the inventoried head `10.3.10.1` timed out;
no fleet restart, successful image build, GPU capture, memory fit or speedup is
claimed by this implementation.

## Build and run on Spark 1

Run from your `ds41-vllm` directory on Spark 1. Use the existing recipe that
produced the 4,226 / 4,065 / 3,843 tok/s result. The earlier instructions assumed
`fleet.prefill-scales.json`; that name was an example and may not exist on your
machine. Do not substitute a checked-in template unless it is actually the
recipe you used, since doing so could change paths, network or Engram settings.

The build helper lists existing Karmic recipes in this directory and `.build`,
including image, K, resident-scales setting and model path. Select your working
recipe by number:

```bash
python3 build-dspark-prefill.py --share
```

If you know its filename, use `--from-config` with that actual path instead.
`python3 build-dspark-prefill.py --list-configs` only lists candidates.
The helper does not guess in noninteractive sessions. It validates the source
and candidate recipe and checks that the base image exists locally, then builds
and distributes the child image. Failed steps stop the sequence. The candidate
`fleet.dspark-prefill.json` is published only after a successful build. An
identical candidate may be reused on retry; a different existing file is refused.

If the earlier commands failed with a missing source config, blank Docker base
and missing output config, those are one failure chain. No successful new image
or candidate config was produced by that sequence. Rerun with the actual recipe.

This is a small source-overlay build, not a vLLM or b12x rebuild. The old image
and recipe are retained. Run the GPU tests with the existing model stopped so
the tests' compilation and scratch allocations do not contend with serving:

```bash
IMAGE=$(python3 -c 'import json; print(json.load(open("fleet.dspark-prefill.json"))["image"])') &&
python3 fleet.py --config fleet.dspark-prefill.json stop &&
docker run --rm --gpus all --ipc host --entrypoint python3 "$IMAGE" \
  -m pytest -q /opt/ds41/dspark-prefill-tests
```

Continue only if all eight GPU cases pass; investigate failures before launch.
Then qualify startup, including the launcher's existing fabric and smoke checks:

```bash
python3 memory-watch.py --config fleet.dspark-prefill.json \
  --output .build/memory-dspark-start-1 --minimum-available-gib 2 -- \
  python3 fleet.py --config fleet.dspark-prefill.json start
```

Use a fresh memory-output directory on each attempt. Keep startup logs and the
per-rank memory summary. After a long request, verify the one-time log messages
`DS41 compact-prefill context graph replay active` and
`DS41 skipping unused intermediate-prefill drafts`. These confirm that each
optimization actually ran. Startup's small-prompt smoke alone does not cover the
new path: it must also complete long prompts and mixed workloads.

```bash
python3 memory-watch.py --config fleet.dspark-prefill.json \
  --output .build/memory-dspark-128k-1 --minimum-available-gib 2 -- \
  python3 benchmark.py --config fleet.dspark-prefill.json \
  --input-tokens 128519 --max-tokens 128 --concurrency 1 --requests 5 \
  --output .build/dspark-128k-1.json
```

Repeat the user's original benchmark at 8k, 64k and 128k, using the same prompt
corpus and cache conditions as the baseline. Also exercise prompt lengths
4095/4096/4097 and 8191/8192/8193, then long prefills mixed with continuing short
requests. Check finite logprobs, sensible output, prompt/completion counts and
decode throughput. Use at least five repetitions after warmup. Synthetic repeated
text in `benchmark.py` is useful for a regression check but its Engram locality
need not match the user's benchmark. Do not compare rates across different
prompt generators as an isolated optimization effect.

## A/B and rollback

Generate separate recipes from the same original configuration, with the same
new child image, to compare each change independently. Set `SOURCE_CONFIG` to
the actual working recipe selected earlier; it is not exported by the helper.

```bash
IMAGE=$(python3 -c 'import json; print(json.load(open("fleet.dspark-prefill.json"))["image"])') &&
read -r -p "Original working recipe path: " SOURCE_CONFIG &&
test -r "$SOURCE_CONFIG" &&
python3 configure-dspark-prefill.py --from-config "$SOURCE_CONFIG" \
  --output fleet.dspark-control.json --image "$IMAGE" \
  --skip-prefill-draft off --compact-context-graph off &&
python3 configure-dspark-prefill.py --from-config "$SOURCE_CONFIG" \
  --output fleet.dspark-skip.json --image "$IMAGE" \
  --skip-prefill-draft on --compact-context-graph off &&
python3 configure-dspark-prefill.py --from-config "$SOURCE_CONFIG" \
  --output fleet.dspark-graph.json --image "$IMAGE" \
  --skip-prefill-draft off --compact-context-graph on
```

Stop the current recipe before starting another. The helper refuses to overwrite
existing recipes. Setting either JSON switch false disables that optimization
on the next restart. To fully roll back to the previously measured deployment:

```bash
read -r -p "Original working recipe path: " SOURCE_CONFIG
test -r "$SOURCE_CONFIG" &&
python3 fleet.py --config fleet.dspark-prefill.json stop &&
python3 fleet.py --config "$SOURCE_CONFIG" start
```

The observed remaining 8k difference from target-only is around 4%. DSpark still
requires target auxiliary outputs, context insertion and final-chunk drafting;
these changes do not promise to eliminate all of that overhead.

## Reported 128k regression and running-image verification

The operator reported roughly **3,300 tok/s at 128k**, versus the earlier
3,843 tok/s, with 8k/64k approximately unchanged. This is about 14% lower
throughput; it is a regression report, not evidence of an optimization gain.
Rank 0 logs from the new run contain both activation markers, graph capture
reports of 0.22 and 0.52 GiB, and 21,552,184 KV-cache tokens. Both DSpark paths
therefore executed on that rank. The capture figures come from separate
startup phases; they do not measure the patch's incremental allocation or
prove 2 GiB physical headroom. The supplied excerpt contains no preemption/OOM
message. It does not establish the cause of the slowdown or verify other ranks.

Run this from the updated deployment directory on Spark 1 **before restarting**:

```bash
python3 diagnose-prefill.py --config fleet.dspark-prefill.json \
  --output .build/prefill-diagnostic-1.json
```

The diagnostic checks every existing serving container: running image ID versus
the current recipe tag, image equality across ranks, vLLM/b12x pin labels,
effective environment and arguments (including resident scales), and exact
source validation for both the bounded-hash fix and DSpark overlay. It checks
the installed validation scripts against this checkout before executing their
read-only checks. It also collects activation/capture/KV/Engram/preemption log
lines and a host RAM snapshot. It does not start containers, load the model,
create CUDA contexts or restart anything. Six local diagnostic tests pass;
live fleet execution remains pending because SSH from this workstation failed.

A passing static check and an activation marker answer different questions:
installed/configured correctly versus observed executing. A missing marker in
the bounded log tail is inconclusive. Neither proves a speedup. A memory snapshot
cannot qualify minimum headroom during prefill; use `memory-watch.py` around the
same original benchmark, preserving corpus and warmup/cache conditions.

To isolate the regression without rebuilding or needing the old recipe filename,
derive these controls from the actual new recipe. This preserves its image,
resident scales, batch size, utilization, thread count and graph diagnostics:

```bash
python3 - <<'PY'
import json
from pathlib import Path
source = json.loads(Path('fleet.dspark-prefill.json').read_text())
for name, skip in [('fleet.dspark-skip-only.json', True),
                   ('fleet.dspark-both-off.json', False)]:
    candidate = dict(source, dspark_skip_prefill_draft=skip,
                     dspark_compact_context_graph=False)
    with Path(name).open('x') as output:
        json.dump(candidate, output, indent=2)
        output.write('\n')
PY
```

Existing files are deliberately preserved. First test only the compact graph
disabled, with intermediate draft skipping retained:

```bash
python3 fleet.py --config fleet.dspark-prefill.json stop &&
python3 fleet.py --config fleet.dspark-skip-only.json start
```

Repeat the same 128k benchmark several times, with the same warmup and fresh
prompt-prefix policy. If performance recovers reproducibly, the compact graph
path or its allocation/capture effects are implicated. If it remains low, test
both switches off in the **same image**, retaining the bounded-hash fix:

```bash
python3 fleet.py --config fleet.dspark-skip-only.json stop &&
python3 fleet.py --config fleet.dspark-both-off.json start
```

Recovery only in that second control implicates draft skipping or its interaction
with the execution path. If neither recovers, compare the previous working recipe
and image under the same conditions, and inspect memory/storage pressure and
target prefill capture. A single recovery after restart is not enough to assign
causality: repeat the original both-enabled case to check reproducibility.
No further runtime optimization is justified by the current evidence alone.
