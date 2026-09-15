# Requested R38 update: latest JJ, capped at c8

Upstream checked 2026-09-14 13:39 UTC. The reproducible source identities are:

- JJ dev/jovian-judgement: c9dc4e543cbc48f3fc4d28818d6942967ce0fd40
- b12x: 3a8b879adaace8688de1e0b8112ec9183dd733a4

R38 is the requested release name; builds resolve these commits, not a floating
release alias. The native ARM64 CUDA13 builder digest remains pinned in
versions.env. The Docker build retains JJ's full build process and SM121 target.

## Current defaults

| Setting | Value |
| --- | --- |
| Cluster | Four Sparks, TP4, DCP1 |
| Maximum sequences | 8 |
| Maximum context | 393216 |
| Draft depth | DSpark5 |
| GPU utilization | 0.85 |
| Prefill batch | 4096 |
| OMP threads | 2; explicit operator override preserved |
| Main / SWA pages | 256 / 128 tokens |
| Graph token capture sizes | Minimal base plus the cap: 1,2,8,48 |
| Extra exact request coverage | Every concurrency1..8 |
| Adaptive verification | On; latest upstream cost/metric fixes |
| Acceptance-history depth adaptation | Off in the new baseline |

The checkpoint remains original MXFP4 with FP8 Engram contents. Its historical
FP4-Engram directory name is preserved. No model tensors or index_topk changes.
Disk Engram stays enabled; resident-scale/projection experiments are not enabled.

## KV-cache and upstream changes

JJ #747 and #749 make SWA physical pages configurable and default the model to
256-token main /128-token SWA geometry. The older deployment used256-token main
blocks and32-token SWA pages, which packed the shared allocator less efficiently.
The recipe explicitly passes --block-size256 and --swa-block-size128. This is
physical layout, not a change in the model's logical attention window. Compare
reported KV token capacity at the same utilization to measure the fleet gain;
there is no claimed capacity multiplier or guaranteed eight full384K requests.

The update also includes #748: adaptive-verification costs follow compatible
padded graphs, rather than extending a lone exact shape as flat/free work.
Metrics now track drafts actually verified. It includes DSpark draft-only shard
loading, coordinated native operation preparation, and current b12x plan support.
The newer upstream removes the short-lived disk-prefetch feature; the recipe
does not carry obsolete flags. Upstream LM-head defaults also changed since
9342b1a, so whole-image comparisons contain more than graph-policy changes.

## Local patches and compatibility

The prefill hash-copy input remains byte-identical; its one-line port is kept.
The RoCEnante C proxy is unchanged. The runtime has new required prepared-plan
arguments: its hash was rebased after inspection, preserving all upstream
execution changes and retaining only our initialization/proxy optimizations.
roce-check.py now obtains the actual adapter's preparation units, prepares them
in a retained session, and passes its plan to direct collective/replay probes.
Preparation is serial in this short diagnostic process; serving uses JJ's own
startup coordinator. The plan remains alive until the probes finish.

Graph-policy enumeration is unchanged upstream. The local exact c1-c8 patch
was retired on 2026-09-15 (see the state-stage OOM section below): it is no
longer applied, the graph child image builder and the launcher gate are gone,
and baked images keep the patch dormant because the launcher never sets its
activation variable. The capture ladder below is the sparse spread, not the
full ladder. Old configs without swa_block_size can also compare geometry,
but use explicit32 for a true old-layout comparison on the new upstream.

## Build and launch on the Spark

Copy the updated ds41-vllm directory from Windows first. Set SOURCE to the
existing operator config containing the correct checkpoint path and host setup.
The migration writes a separate file and resets recipe limits to the table above,
while preserving paths, revision, SSH settings and explicit OMP choice.

```bash
(
set -euo pipefail
SOURCE=.build/cluster-c16-latest.json
bash build-image.sh
python3 configure-r38.py --from-config "$SOURCE" --output .build/cluster-r38-c8.json
python3 fleet.py --config .build/cluster-r38-c8.json share
python3 fleet.py --config .build/cluster-r38-c8.json stop
python3 fleet.py --config .build/cluster-r38-c8.json start
)
```

Use --draft-tokens0 for target-only or7 for a separate experiment (graph token
cap becomes64 at c8/k7). Old c16 and c8 JSON profiles are retained, but are not
the current launch default; they now carry the R38 image tag so every profile
runs the same upstream revision.

## Validation

32 CPU tests pass against the new source trees, including exact c1-c8 shapes,
mixed lengths and c16 fallback in the reusable patch; prefix hash behavior;
RoCEnante source guards/reapplication; launch geometry, migration, and label
guards. Native C proxy probes run during the ARM64 build. GPU ABI checks run
after the build. Fleet start runs four-node NCCL/RoCEnante correctness and CUDA
replay checks, then serving smoke. These GPU/fleet checks have not run from this
Windows workspace, so actual memory-capacity and throughput improvements remain
unmeasured. Capture memory can partially offset the allocator-capacity gain.

## Rust setuptools-scm artifact-tag build fix

Checked upstream again 2026-09-14 13:55 UTC: JJ ab03e871 / b12x9e90d60f exist,
but neither addresses the implicit Rust setuptools version-discovery failure.
Pins remain unchanged for this build-only fix.

Rust installation succeeds. tools/build_rust.py explicitly excludes wheel tags
when deriving VLLM_RS_BUILD_VERSION, but its later setup() call triggers a second
setuptools-scm discovery from pyproject.toml without that exclusion. A tag such
as vllm-jovian-cu134-beta-<sha> is selected and cannot parse as a package version.

build-image.sh now runs prepare-build-tags.py on its disposable .build clone.
It verifies HEAD, backs up refs matching vllm-jovian-cu134-* and removes only
those refs locally. Semantic-version tags, commit, source files and remote refs
are unchanged. Repeated runs are idempotent. The helper refuses repositories
outside this recipe's .build directory. Backups live outside the Docker context.

To restore refs for debugging, run:

```bash
source versions.env
python3 prepare-build-tags.py ".build/vllm-$VLLM_COMMIT" --commit "$VLLM_COMMIT" --restore
```

FILTER_ARTIFACT_TAGS=0 skips filtering on a later build. Restore first if the
original tag-discovery behavior is desired. No CUDA/Rust code was patched.
33 CPU tests pass, including a real temporary Git repository reproducing tag
selection and checking filtering, normal-tag preservation, idempotence, rollback
and unchanged HEAD/worktree. The ARM64 Docker build still must run on the Spark.


## RoCEnante preparation dtype fix

Checked upstream again 2026-09-14 15:18 UTC: JJ ab03e87100efa9536ec87e01994828b459c956ff
and b12x 9e90d60f0cc8f204aa2fd219ed9b6abee32de7d8 are newer than our pins.
The latest b12x RoCE preparation file is byte-identical to our pinned version;
it still passes torch.dtype objects to a launcher requiring dtype-name strings.
Pins remain unchanged for this targeted repair.

patches/roce_dtype.py converts only the launcher argument to float16, bfloat16,
or float32. The prepared-program dictionary retains torch.dtype keys because
runtime lookup uses inp.dtype. Both retained b12x source and installed package
are patched. The exact source SHA is guarded; --check verifies application and
--revert restores the original. This patch is independent of the four transport
optimizations. The image tag gains -dtype-v1 and label local-inference.roce-dtype=v1.
The later contextlib destructor errors in the failed fabric log occur during
shutdown after the preparation failure.

For an already-built R38 image, copy the updated recipe and use the small child
image below. This preserves the existing launch settings and avoids recompiling
vLLM. BASE_IMAGE can override the existing image tag if it was customized.

```bash
(
set -euo pipefail
bash repair-roce-image.sh
source versions.env
python3 - "$IMAGE" <<'PY'
import json, sys
from pathlib import Path
p = Path(".build/cluster-r38-c8.json")
c = json.loads(p.read_text())
c["image"] = sys.argv[1]
p.write_text(json.dumps(c, indent=2) + "\n")
PY
python3 fleet.py --config .build/cluster-r38-c8.json share
python3 fleet.py --config .build/cluster-r38-c8.json stop
python3 fleet.py --config .build/cluster-r38-c8.json start
)
```

35 CPU tests pass, including reproduction using the upstream compile_roce
function, validation of all three supported dtypes and runtime dictionary keys,
and source guard/idempotence/rollback checks. Actual GPU compilation, four-node
collective correctness and CUDA graph replay still require fleet qualification.


## RoCEnante compiled-program metadata fix

Checked again 2026-09-14 15:40 UTC: latest JJ is
ab03e87100efa9536ec87e01994828b459c956ff, latest b12x is
9e90d60f0cc8f204aa2fd219ed9b6abee32de7d8. Both RoCE launcher files remain
byte-identical to our pinned b12x version, so upstream does not supersede this
repair. Pins stay unchanged.

The dtype repair exposes a second failure: compile_roce returns launcher
closures without __b12x_programs__, and describe_compilation rejects them with
"compile factory returned an unannotated function". This is the same preparation
path used by serving. Skipping fabric qualification would not fix that path.

patches/roce_programs.py applies the existing PCIe launcher pattern:
return attach_programs(run, raw). It retains both exact program keys and the
compiled dependency for all-reduce and all-gather. It does not disable program
validation or change kernels. Both source hashes are guarded before either file
is changed; --check and independent --revert are supported.

The repair script now defaults to the previously built -dtype-v1 image and
produces -dtype-v1-programs-v1. Use the same repair/share/stop/start command above,
which preserves existing recipe settings. BASE_IMAGE may select the original
R38 image instead; the script applies both repairs idempotently. Full builds
also include both repairs. Roll back the image through the config if needed;
reverting this metadata fix restores the known preparation failure.

37 CPU tests pass. New tests execute the actual upstream launcher factories
with a fake GPU compiler and the real metadata functions, checking exact keys
and retained dependencies for both launchers, plus hash guards and rollback.
No GPU execution was performed here; fleet qualification remains required to
establish collective correctness and CUDA replay on the four Sparks.

## RoCE fabric compile-pool fix

The metadata fix alone leaves one follow-on failure. The fabric check created
its session with compile_workers=0, which compiles in-process: the RoCE
launcher factories are functools.cache memoized per argument key, and planning
evicts only b12x program caches and Triton JIT caches, never functools.cache
wrappers. On a cold shared compile cache (/cache/b12x/compile, new with this
b12x fingerprint) the second factory run returns the memoized launcher without
lowering, no artifact is written, and _wait_programs fails closed with
"required compiler artifacts are unavailable". Serving never reaches this
because get_b12x_session uses 16 workers: the spawned offline workers have
empty memos, lower the launchers for real, and publish the disk objects the
availability check polls.

roce-check.py now creates its session with compile_workers=16, the same pool
path serving uses. The pool is created only when a required artifact is
missing, so a warm cache keeps the check fast; a cold cache compiles once and
also warms the shared cache for the serving containers. The check also guards
its body with __main__: spawned compiler workers re-import the main module
and CUDA is hidden from compiler children, so their re-import must not run
the comparison. repair-roce-image.sh now copies the fixed roce-check.py into
the image and verifies the baked copy against its sha, since the -dtype-v1
base still carries the old in-process check. 38 CPU tests pass; the added
test walks the full compile_roce carrier dict (dtype keys plus gather) and
the check's full AST, which is the exact reported failure path. GPU
qualification on the four Sparks is still required for correctness and replay.

## Serving autotuning reduction

Motivated by CUDA OOM during candidate racing: the race memory budget defaults
to half of free GPU memory, race_batch defaults to 32 resident candidates per
batch, and the per-trial residency snapshot excludes measurement buffers and
fragmentation, so races can overdraw the device in both lifecycle stages.

Checked upstream 2026-09-14 21:54 UTC: both heads moved since our pins (JJ
26c055577be301bb35039d5c83c0fe29a4b9aaf8, b12x
b0a0381335ebab1ae7aa867a0c8c83e9d68052e0). The head's get_b12x_session still
passes neither rounds nor samples to PreparationSession, and still carries the
exact anchor this patch replaces, so upstream does not supersede the proposed
work. Rechecked 2026-09-14 23:05 UTC: the JJ head is unchanged and the b12x
head advanced again to 40bcdf82a03b23c7ac45efc30d13f6b9516e35e5, whose
PreparationSession still declares race_batch=32 and race_budget=None, so
upstream still does not supersede. Pins stay unchanged; the build's
check-upstream.py re-reports the heads and the patch hash guard fails closed
on drift.

Startup autotuning volume has three levers. First, fleet.py captured a
cudagraph size for every decode batch 1..48 at r38-c8, and
b12x_preparation_token_counts turns that into 50 exact serving
specializations; norm.mhc's query includes max_tokens, so each shape is a
separate declaration with its own candidate race. The capture list is now a
minimal base plus the cap: 1,2,8,48 gives 8 exact specializations. For
comparison, dropping the override entirely (upstream default,
performance_mode balanced) yields 15 captured sizes up to 96 and roughly 22
specializations, so the upstream default is the wrong direction for tuning
volume. Decode batches pad to the nearest captured size, which coarsens
mid-batch padding: 3-7 request decode batches pad to the 8-request cap; the
smoke concurrency of 8 still hits the cap exactly.

Second, patches/b12x_tuning.py reduces the serving tuning budget, bounds its
resident memory, and halves the compile pool's anonymous host RAM peak:
get_b12x_session passed only autotune and compile_workers, while
PreparationSession defaults to SURVIVOR_ROUNDS=3 rounds with 8 timed samples
per candidate per round, race_batch=32 resident candidates per batch, and
race_budget = half of free GPU memory. The patch passes compile_workers=8,
rounds=1, samples=4, race_batch=8, race_budget=4GiB, cutting timed GPU work
about 6x per candidate, capping resident candidate memory per race batch, and
halving the host-side compile pool (64 to 32 processes fleet-wide, 16 to 8 per
node) whose anonymous compilation buffers are the main MemAvailable consumer
during compilation. b12x's own recorded race
evidence (preparation/_measurement.py): round-to-round spread stays under 4%
and the eventual winner never trailed the first round's leader by more than
0.2%, so the extra rounds almost never change the outcome, and the
champion-carrying design keeps winner choice head-to-head at any batch size.
Winners are provisional until benchmarked on the target path. A single
candidate whose residency exceeds race_budget is still prepared (mandatory
preparation always runs), so a legal huge-shape config can still OOM if the
device is truly exhausted.

Both changes are reversible. The patch is hash-guarded against the pinned
vllm source 2213eb87da148aba3508547dcb25585b69b2d6d28da211befa0bc9c5487eecaf;
--check verifies application, --revert restores it, and re-applying over an
image baked with the previous patch form repairs it in place. The image tag
gains -tuning-v2 and label local-inference.b12x-tuning=v1; preflight guards
the label while reduced_tuning is not false in the fleet config, so a stale
bake fails fast, and a rollback rebuild passes after setting
reduced_tuning=false. Separately, the observed SystemExit preparation failures
are shutdown signals (SIGTERM/SIGINT via the WorkerProc handler, the only
bare-SystemExit source) that the startup coordinator reports as preparation
errors, not b12x failures; identifying the trigger requires the full rank-0
container logs.

The rapid OOM persisted even with race_batch=8 and race_budget=4GiB, so the
race budget is not the binding constraint: the baseline (model and draft
weights, KV pool at 0.85, DSpark draft-lane declarations roughly doubling
plans and winners, graph-bucket pools) sits near the ceiling and a single
candidate materialize can overdraw. For comparison, the earlier c8 profile
(jj-35601be/b12x-323107f, draft_tokens=0, utilization 0.80, no draft lane, no
graph buckets) ran stock tuning with the same free//2 budget and fit, but
slowly - stock rounds=3 x samples=8, race_batch=32, and compile_workers=8 are
exactly what the reduction above addresses.

fleet.py now gains a b12x_autotune flag (default true); when false, serve_args
passes --kernel-config {"enable_b12x_autotune": false}. enable_b12x_autotune
is in KernelConfig.ignored_factors, so toggling does not invalidate compile or
selection caches; cached winners are still used where available. With
autotune disabled, b12x_batches puts every request in the default-only batch
and nothing is timed: no candidate races, so the racing OOM disappears;
mandatory preparation (compile plus one default-config materialize per plan)
always runs. The dataclass-typed --kernel-config arg parses JSON via
TypeAdapter, and the per-field --moe-backend/--linear-backend flags are
applied on top via deepcopy in create_engine_config, so the b12x backends
survive. This is a launcher-only change: no image rebuild; stop, update the
flag, start. Tradeoff: uncovered choices run heuristic defaults instead of
measured winners.

## Disk Engram shard metadata

The pinned vllm c9dc4e5 _ensure_disk_table
(models/deepseek_v4_1/common/engram.py) reads table.shard_start,
table.shard_end, and table.shard_rows from the b12x engram DiskTable; the
pinned b12x 3a8b879 keeps that window on self._cache (its own add_shard and
require_complete read the same), so disk-backed Engram lookup fails with
AttributeError at preparation. Checked upstream 2026-09-15: the b12x head
40bcdf82a03b23c7ac45efc30d13f6b9516e35e5 is identical in this region, so
upstream does not fix it; JJ's own default table_memory is device, so the disk
path is evidently less tested upstream. The failure only triggers with
table_memory=disk, which fleet.py has hardcoded since db73040; the older
b12x 323107f still exposed those attributes directly. patches/engram_disk.py
adds shard_start, shard_end, and shard_rows properties exposing the clamped
_cache values, matching the b12x internal pattern; both the lookup and embed
call sites are covered. Hash-guarded against the pinned b12x source
cd01e2b1d69b3d59731f8cb847e09d577a8dda29144ab90752756d4b93bd5183; --check
verifies application and --revert restores it. The image tag gains -engram-v1
and label local-inference.engram-disk=v1; preflight guards the label, so a
stale bake fails fast. The alternative launcher-only escape, table_memory=ram,
moves the tables to mapped host RAM and avoids the DiskTable entirely at the
cost of host RAM and PCIe reads per lookup.

42 CPU tests pass, including the patch's guard/idempotence/rollback checks
against the pinned tree, the prior-patch-variant repair check, the pinned
anchor assertion, the new capture-size lists for every profile, and the
preflight label guard with rollback. GPU qualification on the four Sparks is
still required to establish collective correctness, replay, and the actual
fleet speedup.


## Conservative c8 startup (2026-09-15)

Checked latest JJ 5bca5a58d970216bd46be82575e824c6e424c465 and b12x
213fc1b204b306bdbaa7d40d2a27529658128bf7 at 11:05 UTC. These are newer
than the R38 pins. No source patch or pin update is needed: pinned R38 already
supports kernel_config.enable_b12x_autotune=false. This uses the existing native
switch, so there is no local source workaround for upstream to supersede.

The launcher now defaults missing b12x_autotune to false. configure-r38.py
explicitly copies that default even when its input enabled tuning. The c8 recipe
opts out of the reduced-tuning label gate; the extra request-bucket experiment
was retired on 2026-09-15 (see the section below), and the capture ladder is
the sparse spread (1,2,8 plus the cap: 1,2,8,48 at k5), not the full ladder.
R38 cache geometry, 384Ki context, c8/k5, 0.85 memory utilization and OMP2
remain. Cached/default kernel preparation and graph capture still run; this
removes candidate racing and the bucket coverage explosion, not all startup
allocations. It is not a guarantee against every possible OOM.

Existing repaired R38 images need no rebuild. Copy fleet.py, then update the
operator config to b12x_autotune=false, reduced_tuning=false, max_num_seqs=8,
draft_tokens=5, and remove the dead graph_request_buckets key. Keep its
existing image, checkpoint path and cache path; stop and start the fleet.

## State-stage coverage OOM root cause and graph patch retirement (2026-09-15)

The pasted startup log ("b12x priming gemm.block_fp8_linear: 28121/31454
ready, 0 measured, 0 cached, 12 compilations") died with exit code None (the
unified-memory OOM killer; a CUDA OutOfMemoryError would exit 1 with a
traceback) at ~89% of a 31454-request prepare() call. Source-level audit
against the pinned trees (JJ c9dc4e5, b12x 3a8b879) establishes:

- b12x preparation runs at two lifecycle points: the weights stage (after
  model load, before memory profiling) and the state stage (after the KV and
  state pools exist, before graph capture). The compile pool is created
  lazily and closed at the end of every job, so each stage spawns fresh
  compiler workers whose torch imports produce the warnings seen ~9s before
  the first progress line.
- The DS41 request-bucket capture descriptors (num_tokens from N to N*6 for
  N=1..8, union {1..48}) feed CudaGraphManager.planned_token_counts(), which
  the state stage consumes through _planned_decode_counts. With the sparse
  ladder {1,2,8,48} the state stage therefore declared 50 exact token counts
  instead of the 8 the weights stage saw, breaking the documented invariant
  that both stages declare the same counts (b12x_prepare.py lines 74-79).
  Each added count re-declared and re-compiled every weight-only family per
  layer (norm.mhc three operations per count, moe.decode, gemm.bf16_gemv
  two dtypes per count, vocabulary projection) on top of the state-own
  attention/indexer declarations: an estimated 27k-31k requests, matching
  the observed 31454. The weights stage (8 counts, roughly 4k-6k requests)
  had already completed and fit before the paste window.
- Priming resources persist for the plan lifetime (the owners inside the
  prepared payload are cleared only on release), so volume accumulates
  across thousands of requests while the 16 compile workers per rank compete
  for the same 121.7 GiB unified pool as the GPU. Host RAM is GPU memory on
  a Spark.
- The full 1..48 ladder re-introduces the same explosion through the ordinary
  enumeration: both stages then declare 50 counts (roughly six times the
  sparse-spread volume), so the conservative c8 startup that restored the
  full ladder while disabling the bucket patch did not fix the OOM.

Fixes (launcher-side except the patch removal; no vLLM kernel recompile):

1. The DS41 request-bucket patch is retired: patches/graph_requests.py, the
   graph child image builder, configure-graphs.py, the launcher activation
   variable DS41_GRAPH_REQUEST_BUCKETS, the graph_request_buckets config key
   and its preflight gate are removed. Baked images keep the patch dormant
   (its own code defaults the variable to "0"), so removal is fail-closed.
   The image tag drops -graphs-v1; the rebuild reuses the cached compile
   layers, so bash build-image.sh after copying the recipe is fast. Existing
   repaired R38 images also work under their old tag because the launcher
   never sets the activation variable. The main Dockerfile now also applies
   the engram_disk and b12x_tuning patches with their labels, so a full
   build-image.sh run produces the complete image directly; the repair script
   remains for already-built bases.
2. serve_args restores the sparse capture spread: minimal base plus the cap,
   {1,2,8,48} at c8/k5. Both preparation stages then declare the same 8
   counts, the state-stage volume collapses back to the weights-stage scale
   that fits, and graph capture drops from 48+ graphs to 4. Decode batches
   pad up to the nearest captured size: 3-7 request batches pad to the
   8-request cap and the smoke concurrency of 8 still hits the cap exactly.
3. The tuning-v3 bake (compile_workers=8, rounds=1, samples=4, race_batch=8,
   race_budget=4GiB) is kept: compile_workers=8 halves the compile pool's
   anonymous host RAM peak, the main MemAvailable consumer during
   compilation.

The earlier c16 and c8 recipes now carry the R38 image tag: every profile
runs the same upstream revision, and with the sparse ladder their
preparation volumes stay small. The c16 profile keeps its historical 0.88
utilization, 16 sequences and draft 7; its KV-capacity fit at 0.88 on this
upstream is an unmeasured trial. A startup allocation failure there is a
failed capacity trial, not a reason to weaken checks.

42 CPU tests pass after the retirement, including the minimal capture lists
for every profile, the migration stripping the dead key, and the preflight
label gates with rollback. GPU qualification on the four Sparks is still
required to establish collective correctness, replay, and the actual fleet
gain.

## RoCE collective priming coordination (2026-09-15)

**User-observed, after the retirement rebuild:** the sparse ladder works — the
weights stage collapsed to 5720 requests (from 31454) and model priming
completed in 0:58 — but startup then failed with
`RuntimeError: b12x preparation failed on rank 1: RuntimeError:
distributed.roce.0-1-2-3.collectives failed to prepare with configuration
BackendConfig(backend='native') (fixed): RoCE collective on rank 1 timed out
waiting for rank 0, HCA 0, at sequence 1; the runtime is poisoned (its epoch
stopped at 0, later launches do nothing) and rank data is no longer
trustworthy`. The failure is the fail-stop timeout in
`b12x/comm/roce/roce_oneshot.py check_health` (the kernel's spin limit wrote
failed_seq=1, peer=0, hca=0 into the pinned ctrl region), not a transport or
setup problem.

**Root cause (source-verified):** the RoCE adapter declares its weights-stage
request (`vllm/distributed/device_communicators/b12x_roce_all_reduce.py
get_b12x_preparation_units`) with only `prepare_call=prepare` — no
`collective=CollectiveRequirement(...)`, unlike the PCIe adapter which
declares one. `PreparationJob._run` yields the world-coordination requirement
only for requests that declare a collective, so the RoCE priming — which
primes a real four-rank all-reduce and all-gather
(`b12x/comm/roce/_preparation.py prepared_call` -> `state.all_reduce`) — ran
uncoordinated whenever a rank reached the request. Rank skew near the end of
the weights batch (rank 1 about 14 requests ahead of rank 0, drifted by the
per-rank 0.05s `pool.wait_for_progress` waits) left rank 1's first launch
waiting for rank 0 until the kernel spin limit timed out and poisoned the
runtime.

**Fix:** `patches/roce_collective.py` (hash-guarded against the pinned vLLM
source, independent `--revert` switch) declares
`collective=CollectiveRequirement(key=self._request_name(),
ranks=tuple(sorted(self.global_ranks)))` on the RoCE request and sets the
unit's `autotune=False` (a real collective cannot be raced per rank, and
`PreparationJob._run` raises for collective declarations in tuned batches).
The coordinator then authorizes the collective only when every participant
rank reported ready (`_authorize_ready`), so all ranks prime in the same
advance round. With the current recipe (`b12x_autotune: false`) the request
already landed in the defaults batch, so the fix is a pure correctness change
there.

**Checked revisions (check-upstream.py, 2026-09-15):** vllm pin c9dc4e5,
upstream head 5bca5a5 — different, and the upstream head still carries the
missing declaration, so upstream does not supersede this fix. b12x pin
3a8b879, upstream head 92cd380 — different; the pinned b12x
`_preparation.py` is byte-identical to the audited copy
(`roce-preparation-latest.py`), so the b12x side is current as pinned.

**Deployment wiring:** the main Dockerfile applies the patch to the installed
vLLM wheel with `LABEL local-inference.roce-collective="v1"`; the preflight
greps that label unconditionally (RoCEnante is always enabled on this fleet);
the image tag gains -collective-v1 in versions.env and all three recipes; the
repair script also bakes the patch so a repaired image carries it too.

44 CPU tests pass, including the patch gates (reapplication, rollback, drift
fail-closed, the world-coordination declaration), the preflight label guard,
and the versions.env/recipe image-tag consistency gate. GPU qualification on
the four Sparks is still required: the fix must establish that the four ranks
prime the RoCE collective in the same round without the sequence-1 timeout.

### Follow-up: the fabric check needed the coordinator too (2026-09-15)

**User-observed, on the first start after the coordination rebuild:** the
preflight passed (the new label gate works) and RoCEnante initialized, but the
fabric qualification failed with `ValueError: collective preparation requires
a coordinator` from `roce-check.py:39` -> b12x `session.py:317`. The
declaration is doing its job: `session.prepare()` without a coordinator
correctly refuses collective-declaring requests, and the check script called
it uncoordinated — the same latent manifestation the fix removed from
`prepare_b12x_locally`, in the standalone fabric path instead.

**Fix:** `roce-check.py` now drives its preparation through the same
`B12xPreparationCoordinator` machinery serving uses (imported from
`vllm.v1.worker.b12x_startup`), wrapped in a small gloo exchange shim that
exposes `.ranks` and `.tcp_store_group.all_gather_obj` over the script's 180s
gloo group. This reuses the tested all-participant readiness gate and the
keep-every-rank-in-the-exchange completion rule — a done job keeps advancing
safely, so rank drift in the compile/drain steps cannot strand the exchange
and the comparison loop starts on all ranks together. `PreparationResult.
close()` releases only retained benchmark trials, so the published RoCE plan
stays installed for the comparison. The repair script's stale-bake guard
`check_sha` is updated together with the file.

**Second user-observed failure, first bake:** the coordinator constructor
expects `(requests, autotune)` pairs per batch; the first shim passed a bare
request list, failing at construction with `ValueError: not enough values to
unpack (expected 2, got 1)` (the workload declares one request, so the bare
tuple had one element). Fixed to `[(requests, False)]`; the AST gate now
evaluates the actual argument expression against the real one-request shape
and asserts it unpacks with `autotune=False`, so this shape mistake fails the
suite instead of the fleet.

46 CPU tests pass, including the AST gates asserting the check uses the world
coordinator (no uncoordinated `session.prepare(`), asserts on the
coordinator's error outcome, and that the batches argument is a
request/autotune pair. GPU qualification on the four Sparks is still
required.

## VLLM pin rebase to 5bca5a5 (2026-09-15)

**User-observed, on the first start after the coordination rebuild:** the
fabric qualification passed and the b12x weights-stage preparation completed
for the first time — the RoCE coordination fix works end to end — but startup
then failed at profile_run with `PreparationResourceUnavailableError: V4.1
attention metadata is not prepared` (deepseek_v4_1/attention.py _forward).
Source-level audit against the pinned trees (JJ c9dc4e5, b12x 3a8b879)
establishes the failure is pre-existing in the pin, not a regression of the
coordination work.

**Root cause:** the DeepSeek V4.1 helper unit declares `stage="state"`, so its
`_prepare(device)` only runs in the compile_or_warm_up_model stage — after
profile_run, which consumes the attention metadata. Upstream fixed this at
5bca5a5: the helpers unit moved to `stage="weights"`, profile_run accepts a
`prepare_profile_state` callback (gpu_worker passes
`self._prepare_b12x_profile_state`), and _dummy_run invokes it after
`_init_minimal_kv_cache_for_profiling(num_blocks=1)`, plus
`_wo_preparation_unit` and additional guards. A narrow one-line patch
(helpers stage only) is insufficient: at the weights stage the KV caches are
`torch.tensor([])` placeholders (numel == 0), so the provider returns no units
there regardless of stage — the full upstream restructure is what makes
profile_run see prepared attention metadata.

**User-approved pin update:** versions.env advances VLLM_COMMIT to
5bca5a58d970216bd46be82575e824c6e424c465, adopting all upstream execution
changes; the image tag becomes jj-5bca5a5-... in all three recipes. The b12x
pin stays at 3a8b879: the b12x head (92cd380) still declares API_VERSION = 1
(the adapter contract), the b12x-side patches are hash-guarded against the pin
and apply cleanly, and re-auditing every b12x patch against the head is
unnecessary for a failure that lives in the vLLM tree.

**Patch targets re-audited at 5bca5a5 (blob identities, raw-file sha256):**

- `vllm/distributed/device_communicators/b12x_roce_all_reduce.py` — unchanged
  (b8987d09...), so patches/roce_collective.py applies as is.
- `vllm/v1/core/block_pool.py` — unchanged (a0932da6...), so
  patches/prefill_hashes.py applies as is.
- `vllm/model_executor/warmup/b12x_prepare.py` — changed (2213eb87... →
  9521a4f8...), but the exact anchor `compile_workers=16,\n    )` survives
  once and the patched result compiles, so patches/b12x_tuning.py applies
  with its SOURCE_SHA rebased. get_b12x_session's autotune gate
  (`session.autotune and os.environ.get('B12X_AUTOTUNE', '1') != '0'`)
  survives, so `b12x_autotune: false` still skips the tuning shard.

**Contract surfaces survive at 5bca5a5:** B12xPreparationCoordinator,
_authorize_ready, and StatelessProcessGroup remain in
vllm/v1/worker/b12x_startup.py; the gpu_worker coordinator RPCs
(begin/advance/abort_b12x_preparation) survive; prepare_b12x_locally is
renamed prepare_b12x_profile upstream and gpu_worker calls it with
stage="state" — consistent with roce-check.py's coordinator use.
enable_b12x_autotune remains a KernelConfig ignored_factor.

**Checked revisions (2026-09-15 17:29 UTC):** JJ dev/jovian-judgement
5bca5a58d970216bd46be82575e824c6e424c465, b12x HEAD
92cd3800932539c947c9a8e06123fe5f36c9eae4 (not adopted); check-upstream.py
reports the vllm pin equal to the upstream head.

**Windows audit-tree note:** the local pinned tree must be cloned with
`git -c core.autocrlf=false clone --depth 1 --branch dev/jovian-judgement ...`.
A plain Windows checkout rewrites the files that differ from the default
branch with CRLF line endings, which corrupts the blob-level sha identities
the hash guards compare; the Linux build checkout is unaffected. The audit
tree is re-cloned at 5bca5a5 (tmp/jj-audit/local-inference-lab-vllm-5bca5a5)
and the vllm test SOURCE defaults point at it; the b12x tree is unchanged.

46 CPU tests pass against the new tree, including the tuning patch gates
against the rebased SOURCE_SHA, the roce_collective and prefill hash guards
against the unchanged targets, and the versions.env/recipe image-tag
consistency gate. This is a full-image change (new vLLM revision): copy the
updated recipe and run bash build-image.sh on the Spark — a new vLLM commit
invalidates the vLLM build layers, so expect a full vLLM rebuild rather than
a cached repair — then share, stop and start the fleet as in the command at
the top. GPU qualification is still required: the rebase must establish that
profile_run sees prepared V4.1 attention metadata and that the four ranks
prime the RoCE collective in the same round.
