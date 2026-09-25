# Integrating display-reserved KV into the ds41-vllm serving target

How to give the current R38 TP4 serving target **1.75 GiB of extra usable KV headroom per
Spark** using the display-reserve memory technique, plus how to prove we actually got it.

- Technique analysis and verdict: [`../gb10-display-ram/README.md`](../gb10-display-ram/README.md)
- Operator procedure for the host side: [`../gb10-display-ram/runbook.md`](../gb10-display-ram/runbook.md)
- Current serving target: `cluster-r38-c8.json`, image tag `...-hashes-v1-dtype-v1-programs-v1-tuning-v3-engram-v1-collective-v1`

**Status: design only. Nothing here has been applied, built, or run on a Spark.**
This document specifies work; it does not claim results. Revise before duplicating the work.

## 1. Why this is worth doing here

Our KV headroom is the binding constraint on long-context capacity, and our own
notes already frame capacity this way: `UPSTREAM-R38.md` says *"Compare reported KV token
capacity at the same utilization to measure the fleet gain"*, and it records that
*"Capture memory can partially offset the allocator-capacity gain."* A per-GPU
reclaim of 1.75 GiB at TP4 is **7 GiB of additional KV backing across the fleet**,
obtained without changing `--gpu-memory-utilization` (0.85) and therefore without
tightening the ordinary-memory margin the launcher deliberately keeps.

The technique is per GPU and independent of parallelism, so TP4 needs no topology change.
What is *not* reusable is the author's integration: he patches
`vllm.v1.worker.utils.allocate_kv_cache` by SHA-256 against **his** vLLM build. Our pin
is a different tree, so the anchors differ and the patch must be re-derived (see §4).

## 2. What has to change, at three levels

| Level | Change | Lives in |
| --- | --- | --- |
| Host | Headless boot target; `nvidia_drm modeset=1 fbdev=0`; udev rule for sysfs readability | operator, per node; checked by `fleet.py`, never changed by it |
| Image | Ship the allocator; patch vLLM's KV backing and KV budget; label the overlay | `Dockerfile`, `patches/` |
| Launch | Expose `/dev/dri/card0` + group; add the display capability; add the credit flags; preflight the state | `fleet.py`, `cluster-r38-c8.json`, `configure-r38.py` |

The launcher must keep its existing principle that it checks but never changes drivers,
boot settings or networking. That is what makes the host side an operator step.

## 3. The two separate effects, and why we need both

This distinction is the core of the design and the easiest thing to get wrong:

1. **Substituting the backing** — the KV tensors are backed by the display region instead of
   `torch.zeros`. This moves KV *out of* ordinary RAM, but it does not change `num_blocks`,
   so capacity is unchanged, only ordinary RAM is freed.
2. **Crediting the budget** — the profiled KV budget is *increased* by the display bytes. Since
   `num_blocks = available_memory // bytes_per_block` (`kv_cache_utils.py:1568`), a larger budget
   is what produces more blocks and therefore **more usable KV headroom**.

Doing only (1) buys ordinary RAM back. Doing only (2) makes vLLM ask for more blocks than
ordinary RAM can back and the allocation fails or OOMs. Both are required, and they must be
applied together, which is why they are one workstream but two separately revertible patches.

## 4. Exact hook points at our pin (vLLM 5bca5a5)

Both anchors below were read from the pinned tree and their source hashes computed with the same
method our existing patches use (validated against two pins already recorded in
`UPSTREAM-R38.md`: `model_executor/warmup/b12x_prepare.py` → `9521a4f8…`,
`v1/core/block_pool.py` → `a0932da6…`).

### 4a. Backing substitution — `v1/worker/utils.py`

`allocate_kv_cache()` already asserts that every tensor shares **one backing allocation**, which
is exactly the property a single contiguous display span provides:

```python
    sizes = {tensor.size for tensor in kv_cache_config.kv_cache_tensors}
    assert len(sizes) == 1, "KV cache tensors must share one backing allocation."
    buf = torch.zeros(sizes.pop(), dtype=torch.int8, device=device)
```

- file sha256 at 5bca5a5: `066555c3f4a5a9df55d2eaea22db40e3c98523a2279964649251537c0a923a4f`
- The author's anchor is `buf = torch.zeros(buf_size, ...)`; **ours is `sizes.pop()`**, so his patch
  does not apply here and the anchor must be re-derived. Recording this is the point of §8.

### 4b. Budget credit — `v1/worker/gpu_worker.py`

The profiled KV budget is computed here, after the memory profiler and the CUDA-graph
accounting, and is then handed to the planner:

```python
        self.available_kv_cache_memory_bytes = (
            self.requested_memory
            - profile_result.non_kv_cache_memory
            - late_persistent_memory
            - cudagraph_memory_estimate_applied
        )
```

- file sha256 at 5bca5a5: `392133a37ce23bed46e2d898c7752a59be208ed302da43ac902cec68e127d360`

### 4c. Upstream precedent — we do not need to invent the decoupling

`kv_cache_utils.py:2518-2538` already separates the *allocated* cache size from the
*profiled* budget for `num_gpu_blocks_override`, and says why:

> *"If `num_gpu_blocks_override` is set, the cache size that will actually be allocated is
> decoupled from the profiled `available_memory` … Reflect that in `available_memory` here so
> auto-fit, the admission check, and the per-worker config builder all plan against the same
> effective capacity."*

That comment is the user-side approximation of the credit. It also means the credit has exactly
one required property: **every consumer of the budget must see the same effective capacity**, or
auto-fit, the admission check and the per-worker config will plan against different numbers.

This gives us a second, independent way to obtain the same effect — set
`--num-gpu-blocks-override` to the block count that the display region supports, with the patched
backing supplying it. That route needs no change to the budget computation at all, and may be the
lower-risk first experiment. It is also static rather than automatic, so it is a fallback, not the target.

### 4d. Consumers of the credited budget to keep consistent

Because the credit is applied at 4b, these consumers all see the inflated budget by construction,
which is the property 4c demands:

- `reserve_mm_ipc_gpu_memory(int(self.available_kv_cache_memory_bytes), ...)` —
  `gpu_worker.py:721`. **Open question:** this reserves multimodal IPC memory *out of* the KV
  budget, so a credited budget could enlarge that reservation beyond ordinary memory. This is the
  first thing the hardware test must check; if it does, credit the display bytes *after* this call.
- `_check_enough_kv_cache_memory` — `kv_cache_utils.py:2555`, planned against
  `available_memory` minus the null block. With the credit this check plans against the larger
  effective capacity, which is intended.
- auto-fit for `original_max_model_len == -1` — not our case, we pin `--max-model-len 393216`.
- `vllm:cache_config_info` — `config/cache.py:315` returns every `CacheConfig` field as a label,
  including `kv_cache_size_tokens` (`kv_cache_utils.py:2248`). This is the metric the headroom test
  reads.

## 5. Guardrails the helper must carry

1. **Fail closed, never fall back.** If the DRM buffer or the CUDA registration fails, raise. Never
   silently substitute `torch.zeros`: at 0.85 utilization a silent fallback would allocate the credited
   block count out of ordinary RAM and OOM later, which is worse than a clean startup failure.
2. **Disable without divergence.** With `display_kv` disabled the credited budget must be *byte-identical*
   to today's value; the credit function returns `available` unchanged.
3. **Ownership for the process lifetime.** Keep the span alive while tensors, views and CUDA graphs are
   live; release only at exit. A freed-but-referenced span is a documented failure mode.
4. **One pool per worker.** Refuse a second real allocation so the block math cannot silently double-count.
5. **Never claim `num_blocks` the span cannot hold.** Clamp the credit so the total KV size never exceeds
   the contiguous span; the author keeps 256 MiB of the 2 GiB back, and we should keep a margin too.
6. **Do not touch the ordinary admission path.** The credit must not widen `requested_memory`, the
   profiler's `non_kv_cache_memory`, the cudagraph estimate, or the no-swap cgroup limits used to
   compute them. Those stay ordinary-memory policy.

## 6. Failure modes and rollback

| Symptom | Meaning | Action |
| --- | --- | --- |
| Display allocation fails at startup | DRM state or permissions wrong, or the driver rejected the geometry | Launcher preflight should have caught it; fix host state, do not fall back |
| Backing registered but ordinary RAM also drops 1.75 GiB | Buffer came from ordinary RAM, not the carveout | Roll back the credit; the technique is worth nothing here |
| Startup OOM after enabling the credit | Credit exceeds what the span can hold, or the multimodal reservation absorbed it | Lower `display_mib`, then re-test |
| KV capacity unchanged with the credit enabled | The credit never reached the planner, or the backing replaced the allocation without changing `num_blocks` | Check patch labels and the two log anchors in the headroom test |
| Disabled config differs from baseline at all | The patch is not inert when disabled | Roll back both patches independently |

Both patches carry `revert=True` and `--check` paths, so each can be reverted on its own and
re-verified against the pin. The Dockerfile applies with `--check` after applying, in the existing
idiom, so an unexpected state fails the build rather than producing a silently different image.

## 7. The headroom test — proving we actually got more usable KV

The point of the test is the assertion in §3: a bigger *reported* KV capacity, backed by the
display region rather than by ordinary RAM. A test that only proves "the allocator did not crash"
would pass even if the credit never reached the planner, so the test asserts capacity directly.

All numbers come from surfaces that already exist at our pin, so no new instrumentation is needed:

| Number | Surface | Anchor |
| --- | --- | --- |
| Reported KV tokens | `/metrics` label `vllm:cache_config_info{kv_cache_size_tokens="N"}` | `config/cache.py:315`, set at `kv_cache_utils.py:2248` |
| Effective KV budget | `Available KV cache memory: X GiB` | `gpu_worker.py:678` |
| Derived KV tokens | `GPU KV cache size: N tokens, ...` | `kv_cache_utils.py:2251` |
| Ordinary host RAM | `/proc/meminfo` (`MemTotal`, `MemAvailable`) | reuse `observe.py`'s `HOST_PROBE` |

`bytes_per_token` is derived from each run's own two log lines rather than assumed, so the
expected gain is computed from the same run that produced it and cannot be tuned to pass.

### Assertions

| # | Assertion | What it rules out |
| --- | --- | --- |
| A1 | `T1 > T0` — strictly more reported KV tokens with the display KV enabled | A credit that never reached the planner, and a backing swap with unchanged `num_blocks` |
| A2 | `(T1 - T0) * bytes_per_token ≈ display_mib`, within 10% | A capacity gain that came from somewhere other than the display region |
| A3 | Steady-state host `MemAvailable` in the display run is not lower than the baseline by the display bytes | A "gain" taken out of ordinary RAM, which would be the opposite of the claim |
| A4 | `cuMemGetInfo` / `nvidia-smi` totals unchanged | A misunderstanding that this enlarges `cudaMalloc` budget or reported VRAM |
| A5 | With `display_kv` disabled, `kv_cache_size_tokens` equals the recorded baseline exactly | A patch that is not inert when disabled, i.e. ordinary capacity silently changed |

A1 and A2 together are the test the request asks for: **more usable KV, and of the predicted size.**
A3 and A4 are what stop the test from being satisfiable by simply taking ordinary memory, which is
the failure mode that would look like success. A5 is the rollback guarantee.

### Levels

1. **Offline, runs today without a Spark** — `tests/test_display_kv.py`:
   anchors present in the pinned tree, hashes match, patch idempotency and independent
   rollback, credit arithmetic and the disabled-is-inert property. This level cannot show
   capacity; it shows the patch is correct and reversible.
2. **Container preflight, needs one Spark** — `display-check.py`, run by `fleet.py preflight`
   with the DRM device exposed, in the idiom of the existing `roce-check.py`. This is the level
   that can prove A3/A4 on real hardware and fail closed before a serving start.
   Note it cannot run at image *build* time: unlike `probe_roce.c`, which stubs libibverbs, this
   needs a real DRM node and a real CUDA context.
3. **Fleet capacity, needs the four-node target** — the A1/A2/A5 test, run once per configuration
   against the R38 target at the same `gpu_memory_utilization` (0.85), comparing a
   `display_kv` disabled run with an enabled run. Until this level runs, **no capacity claim is
   made** — consistent with `UPSTREAM-R38.md`, which records memory-capacity improvements as
   unmeasured.

Level 3 is the one that answers the request. Levels 1–2 exist so that level 3 has a meaning:
without them, a capacity difference would not be attributable to the technique.

## 8. Open questions to settle before writing the credit path

1. Does `reserve_mm_ipc_gpu_memory` draw on the credited bytes? If yes, credit after that call.
2. Does the DRM driver accept one 4096 × 114688 dumb buffer on our driver version, or must the span
   be split into several buffers mapped back-to-back with `MAP_FIXED`?
3. What `NVIDIA_DRIVER_CAPABILITIES` and device/group exposure does our launcher need, given it is
   not a privileged container today?
4. What is the measured read penalty at our KV geometry under B12X?
5. Does B12X's attention path accept a host-registered device pointer for KV, or does it assume
   `torch.zeros`-style aligned device memory?

## 9. What was verified offline, and what was not

Performed from this Windows workspace on 2026-09-19. **No Spark was contacted; no image was built;
no server was started; nothing was applied to a running deployment.**

| Check | Result |
| --- | --- |
| Hashing method for `SOURCE_SHA` | Validated: reproduces `b12x_prepare.py` → `9521a4f8…` and `block_pool.py` → `a0932da6…`, both already recorded in `UPSTREAM-R38.md` |
| Audit checkout line endings | 0 CRLF, so blob identity is not corrupted (the `UPSTREAM-R38.md` Windows caveat) |
| Both anchors present exactly once at the pin | Yes; the computed hashes are recorded in §4 |
| Backing patch: re-application, `--check`, rollback, drift refusal | Passes, and the patched file still compiles |
| Credit patch: re-application, `--check`, independent rollback | Passes; the two patches target **different files**, so independent rollback is structural |
| Credit is inert while disabled | Yes: 0 bytes with `DS41_DISPLAY_KV_MIB` unset or `0`, exactly the credit when set, refused above 1792 |
| Backing refuses instead of falling back | Yes: raises rather than allocating from ordinary RAM |
| `configure-r38.py` carries `display_kv` forward | Yes: the generated config keeps `display_kv`, utilization 0.85 and 8 sequences |
| Disabled launch plan unchanged | Yes: byte-identical to before, with no device, group, capability or credit |
| Enabled launch plan | Adds only the DRM card, `--group-add "$DS41_DISPLAY_KV_GID"`, the node-side group discovery, `NVIDIA_DRIVER_CAPABILITIES` and `DS41_DISPLAY_KV_MIB`; **`serve_args` is untouched** |
| Headroom assertions A1–A5 | **Not verified.** All four headroom tests skip without a `display-kv-{baseline,enabled}` run pair. |

Result: 8 offline tests pass and 4 hardware tests skip. The existing suite's errors in this
workspace are all `PermissionError` on the system temp directory, which is not writable here; they are
unrelated to these changes, and `FleetTests` covers the launcher paths that were changed.

**Explicitly not verified: the capacity gain itself.** Nothing here shows that a Spark reports more
usable KV headroom; that is level 3 in §7 and requires the fleet.

## 10. Pin policy for these patches

`check-upstream.py` ran on 2026-09-19 14:40 UTC. Both pins have moved and **neither is advanced
here**: a pin bump must adopt the upstream execution changes first, exactly as the `VLLM pin rebase to
5bca5a5` section of `UPSTREAM-R38.md` did.

| Upstream | Pin here | Current head | Effect on these patches |
| --- | --- | --- | --- |
| local-inference-lab/vllm | `5bca5a5` | `8e1f1e58` | `v1/worker/utils.py` is **unchanged** (`066555c3…`), so the backing patch applies as is |
| local-inference-lab/vllm | `5bca5a5` | `8e1f1e58` | `v1/worker/gpu_worker.py` **changed** (`392133a3…` → `a04163b2…`), but the credit anchor survives exactly once, so the patch applies with `SOURCE_SHA` rebased |
| local-inference-lab/b12x | `92cd380` | `0f3a8cbf` | Not re-audited; the technique does not touch b12x |

Both patches are therefore pin-portable, which is the useful result: not because the pins are current,
but because one target file did not move and the other kept its anchor. Do not advance either pin on the
strength of this check alone, and re-audit both anchors whenever the vLLM pin moves.

## 11. Per-node usable memory, and why never to claim 100%

`kv-budget.py` computes the whole per-node chain. Every term is the one vLLM itself uses
at this pin, so the table explains the reported capacity rather than restating it:

| Term | Source |
| --- | --- |
| `requested = ceil(total_memory × gpu_memory_utilization)` | `v1/worker/utils.py:505` |
| `ordinary_kv = requested − non_kv_cache − late_persistent − cudagraph_estimate` | `v1/worker/gpu_worker.py:657` |
| `num_blocks = (ordinary_kv + credit) // bytes_per_block` | `v1/core/kv_cache_utils.py:1568` |
| `concurrency = num_blocks / blocks_per_request` | `v1/core/kv_cache_utils.py:1001` |
| `capacity = int(concurrency) × max_model_len` | `v1/core/kv_cache_utils.py:2240` |

So the answer to "what is the actual total usable memory per node" is:

```
usable = ceil(total_memory × utilization) − non_kv_cache − late_persistent − cudagraph  [+ display_credit]
```

**Why the display credit does not cause the 100% problem.** The credit is *additive* and the
display reserve is not part of `total_memory` at all, so the credited claim against
`total_memory` is `utilization + credit/total` — with a 1.5 GiB credit on a 121.66 GiB device
that is 85.00% → 86.23%, and the ordinary residual margin is untouched at 18.25 GiB. The
technique therefore needs **no** change to `gpu_memory_utilization`, and the 0.85 factor keeps
its meaning for the ordinary budget.

**`--kv-cache-memory-bytes` is the wrong tool for this.** Per `entrypoints/llm.py:121`,
`kv_cache_memory_bytes` (when not-None) **ignores `gpu_memory_utilization`**. Setting it to the
reported maximum would claim the whole free budget with no utilization guard at all, which is exactly
the 100% case to avoid. Keep the utilization path and add the credit instead.

**The floor effect, and what to measure.** `capacity` is `int(concurrency) × max_model_len`, so it is
quantised to whole max-length requests, while the credit is a fixed byte amount. At a 393,216-token
limit one full request costs ~14.3 GiB of KV, so a 1.5 GiB credit is ~10% of one request: it buys a
whole extra max-length request only if the baseline happens to sit within 1.5 GiB of the next boundary,
and otherwise buys none while still enlarging the backing.

| `--ordinary-kv-gib` | one max-length request | credited | full requests gained |
| --- | --- | --- | --- |
| 100.0 | 14.33 GiB | 101.5 GiB | **+1** (6.98 → 7.08 requests) |
| 51.0 | 14.40 GiB | 52.5 GiB | **0** (3.54 → 3.64 requests) |

Both rows are correct: the same 1.5 GiB credit buys one full request in the first case and none in
the second, purely because of where the floor falls. Consequences for §7:

- The **unquantised** surface is the logged `Available KV cache memory` (`gpu_worker.py:678`), which
  is logged from the credited budget. That is the primary assertion, and it shows the credit directly.
- The **quantised** token capacity (`vllm:cache_config_info`) can legitimately stay flat, so a flat
  reading is reported as the floor effect, not treated as a failure. Asserting a strict token increase
  would fail a correct implementation at some baselines.
- The prediction uses each run's own logged budget and token count, so it is a derivation from one log
  rather than a fitted constant. The fleet total is ×4 nodes.

So `kv-budget.py --log rank-0.log` gives the per-node and fleet numbers before anything is enabled,
and the same numbers are what the headroom test checks afterwards.


