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
| Graph token capture sizes | Every integer1..48 |
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
