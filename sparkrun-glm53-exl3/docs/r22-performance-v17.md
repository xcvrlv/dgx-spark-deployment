# v17: launch memory reclamation and prefix-cache metadata

```bash
bash sparkrun-glm53-exl3/scripts/build-r22-v17-image.sh WORKER1 WORKER2 WORKER3
```

Use `recipes/glm53-exl3-v17-4x.yaml`. Its working default remains **0.87**.
The aim is to test **0.89**, not to assert that it now fits. Increasing the
fraction by 0.02 adds roughly 2.5 GiB on a 128 GiB device; the exact increment
depends on the memory total reported by CUDA. If earlyoom fires because of
steady-state host headroom rather than a transient, startup cleanup may not
solve it. The last startup and earlyoom log lines are still needed to establish
the actual phase and threshold.

## Launch changes

`VLLM_GB10_STARTUP_RECLAIM=1` enables SM121-only cleanup after model loading,
before production KV allocation, after kernel warmup, and after graph capture.
Each boundary synchronizes CUDA, collects unreachable Python objects, releases
unused PyTorch CUDA allocator cache, then calls glibc `malloc_trim(0)` to return
eligible unused CPU heap pages. The latter can reclaim pages that Python GC
and CUDA `empty_cache()` alone leave resident. It cannot release live weights,
KV blocks, pinned RoCE slots, or live graph allocations.

The operation occurs only at startup, outside capture and outside inference.
It preserves graph memory profiling, KV budgeting, earlyoom configuration and
all live workspaces. It does not increase utilization by secretly reducing the
KV budget. Set `VLLM_GB10_STARTUP_RECLAIM=0` on every node to disable it.

Each `GB10 startup memory` log records stage, PID and before/after values:

- `/proc/meminfo` MemAvailable, MemFree, Cached and SwapFree, in KiB;
- process VmRSS and VmHWM, in KiB;
- CUDA free, PyTorch allocated and reserved memory, in bytes;
- whether the allocator reported reclaiming pages.

These are boundary snapshots, not a continuous peak trace. Compare process RSS
and host MemAvailable along with the startup phase. Cleanup cannot rescue a
process killed before reaching its next boundary, and malloc_trim may reclaim
nothing. There is no demonstrated GiB saving or successful 0.89 launch yet.
CUDA execution errors are propagated; unavailable glibc trimming is a no-op.

The existing R22 graph profiler already tears down temporary KV bindings and
profiling graphs, and v10 already releases EXL3 preparation ballast. v17 does
not duplicate those mechanisms or weaken their lifetime protections. It also
keeps the existing CPU thread counts, mmap safetensors loading, graph sizes,
prefill capacity, block size and RoCE slot limits.

## Prefix-cache finding

When registering newly computed cache blocks, R22 copied
`block_hashes[num_cached_blocks:]`, including the entire uncomputed prompt
suffix. v17 limits this to `[num_cached_blocks:num_full_blocks]`. This matters
for long prompts processed in chunks and for lazy DCP hash views: work follows
the new blocks rather than repeatedly materializing the remaining prompt's
hashes. For a million-token prompt, many successive updates can otherwise copy
thousands of future hash entries each time.

This is a CPU/allocation optimization. It preserves hash values, cache salts,
block lookup, DCP grouping, masking, events, eviction and partial-tail promotion.
It does **not** increase the cache hit rate or persist prefixes across restarts.
Prefix caching is already enabled. Fine-grained partial hits already have
copy-on-write support in the pinned implementation, so v17 does not introduce
another prefix-sharing mechanism or change the match granularity.

## Validation and next measurements

Five new CPU tests check pinned-source application, rejection without partial
writes, cleanup ordering, capture exclusion, disabled/non-SM121 cases, proc
parsing, optional glibc availability, bounded lazy hash access, masked/null
block registration, startup-only hooks and the unchanged default utilization.
The build runs the inherited v16 tests and adds a GPU test that retains tensors
and a captured graph through reclamation, changes inputs, and checks replay.
Local CPU checks cannot validate ARM allocator reclamation or launch peaks.

First collect the boundary logs at 0.87. Then try 0.88 and 0.89 with the same
workload and observe both startup and inference headroom. Keep v16 available.
If earlyoom still intervenes, retain its message and the last startup log;
those determine whether to target model preparation, warmup/capture, or the
steady-state budget next. The build's synthetic GPU smoke does not load the
model or prove that 0.89 is safe.

Sources: [glibc malloc_trim behavior](https://man7.org/linux/man-pages/man3/malloc_trim.3.html),
[pinned worker startup](https://github.com/local-inference-lab/vllm/blob/70b3c1c7f1c76fcf0847fcbb4a0b8b5583b78d19/vllm/v1/worker/gpu_worker.py),
[pinned prefix block pool](https://github.com/local-inference-lab/vllm/blob/70b3c1c7f1c76fcf0847fcbb4a0b8b5583b78d19/vllm/v1/core/block_pool.py).
