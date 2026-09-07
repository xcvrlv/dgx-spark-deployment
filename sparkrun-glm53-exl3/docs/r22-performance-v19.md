# v19: mixed-K M16 support and SM121 FC2 scheduling

```bash
bash sparkrun-glm53-exl3/scripts/build-r22-v19-image.sh WORKER1 WORKER2 WORKER3
```

Use `recipes/glm53-exl3-v19-4x.yaml`. This candidate needs the GB10 build,
cumulative GPU smoke and serving measurements. Local CPU validation cannot
establish CUDA compilation, numerical correctness or a throughput improvement.
The builder runs the cumulative smoke on all four nodes, including the new
two/three-tier M16 and grouping tests. Earlier version tests remain included.

## M16 fix

The generic routed-block list accepted 16, but that did not establish support
for our mixed-K tile geometry. Both mixed-Trellis compilers only split FC2
into M8 subtiles when the parent block was 32 or 64. Setting 16 therefore
selected an unsplit FC2 with N512/K32. Its register-table key
`(256, 1, 32, 2, False)` is missing in this pinned B12X version.

v19 also splits M16 into the existing M8 FC2 specialization. FC1 and route
packing still use M16. Two adjacent M8 subtiles cover one M16 packed block;
they cannot cross an expert boundary. The existing M16 FC1 and M8 FC2
register entries are used without adding guessed register counts. The same
change applies to both two-tier and three-tier mixed-K compilers.

The new recipe retains `VLLM_EXL3_PREFILL_BLOCK_M: "32"`. After the build
passes, compare it with `"16"` and restart every worker. M16 reduces the
maximum per-expert padding from 31 to 15 routes and uses a smaller FC1 row
tile. It also processes fewer tokens per weight tile, so it can increase
weight traffic. Which effect wins depends on the routed batch. It is not
guaranteed to beat 32. Your observation that 64 is slower remains consistent
with keeping 32 as the baseline.

This patch qualifies a specific mixed-K configuration, not every combination
of block size, weight layout and tile geometry. In particular, the generic
presence of 48 in the host allowed-values list does not qualify its wide FC2
path. KV-cache `--block-size` is a separate setting.

## Optional FC2 scheduling comparison

`VLLM_GB10_EXL3_FC2_GROUP` controls how many adjacent M8 FC2 subtiles one
scheduler job handles. It is read during compilation; restart all workers
after changing it.

| Prefill block M | GROUP=1 | GROUP=2 (default) | GROUP=4 |
| --- | --- | --- | --- |
| 16 | One subtile/job | Two subtiles/job | Capped at two |
| 32 | One subtile/job | Two subtiles/job | Four subtiles/job |
| 64 | One subtile/job | Two subtiles/job | Four subtiles/job |

Only 1, 2 and 4 are accepted on SM121. The override does not affect the M8
decode path; other architectures retain grouping two for these prefill sizes.
At M32, grouping four halves the number of FC2 scheduler jobs relative to
grouping two. It executes the same matrix tiles, reusing the existing shared
scratch sequentially. Longer runs over one weight tile may improve locality
and amortize scheduling, but may hurt load balance or instruction-cache use.
It does not fuse four matrix operations into one or halve the required math.
Grouping one provides the opposite comparison. The existing compiled kernel
cache key includes this factor.

Suggested comparison order: M32/group2, M16/group2, then M32/group4. Keep
prompt, cache state, concurrency and other settings fixed. The synthetic smoke
prints timings for groups 1/2/4 and actual device SM/shared-memory properties;
serving prefill measurements should decide the recipe setting.

## SM121 resource findings

The fused mixed-K grid uses the runtime device's SM count and the lowest
resident-block limit across FC1/FC2 and bitrate tiers. The current wide
geometry uses 256 threads per block. The pinned register-table entries for
FC1 are 158 registers/thread at M16, 175 at M32, and 255 at M64. These are
launch-model entries, not a fresh measurement of every generated variant.
Even M16's existing estimate admits only one such FC1 block per SM under
the register budget. Lower shared-memory usage alone therefore does not
unlock two resident blocks.

The kernel has cooperative grid barriers. Raising its grid size without
qualifying actual registers and shared memory could prevent all participants
from becoming resident. v19 leaves that occupancy policy intact and reports
the launch resource values in the smoke instead. A smaller-thread-count FC1
and FC2 tile pair is a larger next experiment: it needs new prepared weight
layouts and measured occupancy, and narrower output tiles increase tile work.
The pinned mixed implementation also rejects FC1 K tiles below 128 because
they lose large-M cross-tier partial reductions.

No activation dtype or dequantization path is changed by v19. It retains the
existing FP16 MMA/intermediates and fused output policy, along with inherited
shared-input rotation and activation patches.

## Expert parallelism assessment

The pinned EXL3 adapter explicitly rejects `layer.expert_map` and EPLB. Its
streaming loader and prepared artifacts currently slice every expert across
TP ranks. This is an adapter/loader limitation; a CLI flag alone cannot supply
the missing ownership, token dispatch and output combination implementation.
vLLM's separate [expert-parallel deployment documentation](https://docs.vllm.ai/en/v0.22.0/serving/expert_parallel_deployment/)
describes the dispatch backends and load-balancing machinery for supported
implementations; it does not establish EXL3 compatibility.

For this 256-expert, top-8 model, TP4 stores a quarter of every expert on each
node (intermediate width 512). EP4 would instead give a node 64 complete
experts (width 2048). At balanced load the total expert weight storage and
useful math per node remain similar. The opportunity is in full-width matrix
efficiency and reducing work replicated across ranks, balanced against
routing skew and activation transport.

The switched QSFP fabric permits direct communication with every peer, which
is useful for an all-to-all design. Current RoCEnante all-reduce/all-gather
support is not an expert-token dispatch/combine protocol. Under a simple
independent uniform-owner model, top-8 routing visits about
`4 * (1 - (3/4)^8) = 3.60` distinct nodes per token. Actual routing is not
uniform, but this illustrates why EP does not automatically remove most
cross-node traffic. Hot-expert imbalance can also leave GPUs idle.

EP is worth a separate prefill-oriented implementation study, but is outside
the requested small-patch scope. It would need expert-aware streaming loads,
quantized layout/rotation ownership, global-to-local route mappings, an
all-to-all transport, and target/MTP correctness tests. None is enabled in
v19, and no EP speedup is claimed.

## Validation and rollback

CPU tests reproduce the original M16 register failure using the pinned
constructors, check that both compilers select supported specializations,
preserve existing default/decode constructors, cover grouping boundaries and
architecture gating, and verify hash preflight/idempotence and inherited
smoke manifests. GPU gates compare M16 with M32, require exact results for
group-only changes at M32, and test graph replay with changed activations,
router weights, ragged routes and empty experts. M16 permits bounded rounding
differences from the changed FC1 row geometry; its relative RMS error must
stay below 0.5%, with a separate per-element check.

Use M32/group2 to recover the previous schedule, or use the v18 image/recipe
for full rollback. v19 preserves utilization, KV/graph budgeting, batching,
prefix policy, dense quantization, MTP and network settings.
