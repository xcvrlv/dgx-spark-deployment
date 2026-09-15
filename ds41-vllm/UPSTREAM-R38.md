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
| Graph token capture sizes | Powers of two plus the cap: 1,2,4,8,16,32,48 |
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
therefore still applies after guarding the new file hash (only the upstream
memory-profile preparation callback changed). It is applied in the main image;
no graph child image is needed. At c8/k5, FULL shapes are188 instead of58.
All legal mixed verification totals at each request count are covered. This
costs more graph objects and startup time; the profiler includes them when
budgeting KV memory. The new upstream pricing follows these same shapes.
Set graph_request_buckets=false and restart to compare upstream-only coverage
on the same image. Old configs without swa_block_size can also compare geometry,
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
cap becomes64 at c8/k7). Old c16 JSON profiles are retained, but are not the
current launch default. The new main image carries graph patch label
local-inference.graph-requests=ds41-graphs-v1 for fail-closed launcher preflight.

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

Second, patches/b12x_tuning.py reduces the serving tuning budget and bounds
its resident memory: get_b12x_session passed only autotune and
compile_workers, while PreparationSession defaults to SURVIVOR_ROUNDS=3
rounds with 8 timed samples per candidate per round, race_batch=32 resident
candidates per batch, and race_budget = half of free GPU memory. The patch
passes rounds=1, samples=4, race_batch=8, race_budget=4GiB, cutting timed GPU
work about 6x per candidate and capping resident candidate memory per race
batch instead of letting it reach half the device. b12x's own recorded race
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
opts out of the extra request-bucket experiment and reduced-tuning label gate;
it restores the full earlier token-size ladder (1..48 at k5). R38 cache geometry,
384Ki context, c8/k5, 0.85 memory utilization and OMP2 remain. Cached/default
kernel preparation and graph capture still run; this removes candidate racing,
not all startup allocations. It is not a guarantee against every possible OOM.

Existing repaired R38 images need no rebuild. Copy fleet.py, then update the
operator config to b12x_autotune=false, graph_request_buckets=false,
reduced_tuning=false, max_num_seqs=8, draft_tokens=5. Keep its existing image,
checkpoint path and cache path; stop and start the fleet. Extra request buckets
can be explicitly enabled with configure-r38.py --graph-coverage.
