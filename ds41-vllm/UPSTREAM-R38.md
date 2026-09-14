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
