# v18: continuation-prefill indexer coalescing and compact metadata

```bash
bash sparkrun-glm53-exl3/scripts/build-r22-v18-image.sh WORKER1 WORKER2 WORKER3
```

Use `recipes/glm53-exl3-v18-4x.yaml`. This is a performance candidate awaiting
GB10 build, GPU smoke and serving measurements. No tokens/sec gain is claimed.
The builder includes all preceding overlays and runs the cumulative GPU smoke
on every node. You do not need to run the v15 smoke separately.

## What the linked 8K patch does

[SparkRing #240](https://github.com/FujitsuPolycom/sparkring/pull/240) shards
Flash's mHC prefill. Its prerequisite,
[SparkRing #239](https://github.com/FujitsuPolycom/sparkring/pull/239), contains
the continuation change: eligible final cold-prefill chunks retain internal
recurrent checkpoints without splitting the model pass for checkpoint export.
The chunk limit is 8,192 tokens. This is specific to Flash's recurrent state
and its checkpoint integration; our full GLM-5.3 model does not run those paths.
Neither patch is copied into this image.

v18 applies the related idea of eliminating unnecessary subdivision to the
full model's B12X DSA indexer. It changes indexer calls within an already
scheduled model batch. It does not merge scheduler batches or turn on Flash
checkpointing, mHC, or an 8K model batch.

## Changes

1. **Coalesce long-context indexer query chunks.** R22's B12X builder inherits
   the generic `query_rows * full_context * 4` logits limit. Its actual B12X
   implementation streams K in 32,768-token supertiles and reuses a bounded
   logits slab. v18 uses the supertile width for chunk sizing. This does not
   truncate attention context or change top-k selection. Coalescing is enabled
   only on SM121 with the existing 32K supertile configuration; other supertile
   values use the old splitter.
2. **Build only the metadata the paged indexer consumes.** A fused kernel
   computes causal lengths and compact page tables. This removes the unused
   context-sized `token_to_seq` map and the two device-to-host `.item()` calls
   per DCP chunk. CPU upper bounds size allocations; live device lengths
   determine causality and mask trailing page entries to -1 before B12X can
   read them. The single-request, uncompressed path is specialized; generic
   metadata construction remains the fallback for other shapes.
3. **Reuse causal lengths across layers.** The indexer uses the lengths built
   in metadata directly, avoiding repeated subtraction/allocation in every
   layer that actually runs the indexer. Decode metadata and target/MTP decode
   execution are unchanged.

For a 4,096-row query batch with a 256 MiB logits setting, deterministic CPU
tests execute the real old and new splitting functions and obtain:

| Total context | Old indexer calls/request | v18 calls/request |
| --- | ---: | ---: |
| 8,192 | 1 | 1 |
| 32,768 | 2 | 2 |
| 65,536 | 4 | 2 |
| 131,072 | 8 | 2 |
| 1,048,576 | 64 | 2 |

These are call counts, **not speedup multipliers**. Arithmetic still covers all
valid keys. v16's separate 256-row DCP candidate-merge limit remains in force,
so larger indexer chunks do not imply proportionally fewer network collectives.
Benefits should come from fewer repeated K gathers, launches and metadata work.
Short cold prefills may benefit only from compact metadata and length reuse.

## Memory and rollback

At long contexts, the coalesced 2,048 rows fit the existing profiled prefill
plan. v18 does not increase that plan's row capacity. The unchanged short-context
path can still create the same larger plans as v17. A million-token context
previously produced 64 separate 4 MiB token maps for a 4K query batch, or 256 MiB
per such metadata construction. v18 allocates no token maps. Its two compact
chunks use approximately 64 KiB for causal bounds and TP4 page tables combined.
These are source-derived allocation counts, not measured whole-worker savings;
other allocations and allocator caching still determine RSS and peak usage.

The recipe preserves v17's 0.87 utilization, scheduler batch size, EXL3 capacity,
graph estimation, KV sizing, prefix policy, quantization configuration and CPU
thread counts. It does not establish that 0.89 fits.

Two independent switches allow A/B tests; restart all ranks after changing:

- `VLLM_GB10_INDEXER_COALESCE=0`: restore the original chunk sizing.
- `VLLM_GB10_INDEXER_METADATA=0`: restore generic metadata and per-layer length
  subtraction. Set both to 0 to compare the v17 inference paths in the v18 image.

## Validation

Seven new CPU tests cover exact-source preflight and idempotence, rejection without
partial writes, chunk coverage across requests and tails, sizing-only changes,
the actual metadata kernel interpreted over NumPy with GPU `.item()` forbidden,
DCP ownership for worlds 1/2/4 and interleave 1/16/64, conservative CPU bounds,
trailing-page masking, fallbacks, retained decode code and cumulative smoke
source verification. The top-k oracle accepts alternative boundary-tied IDs
and rejects lower-score selections, duplicates, invalid IDs and mismatched scores.

The GPU smoke adds exact legacy/compact metadata comparisons, independent
enumerated ownership checks, changing live device lengths under graph replay,
and real B12X prefill partitioned/coalesced top-k comparisons over two K
supertiles. Both runs are independently checked for valid unique IDs, scores
matching their selected IDs and the correct top-k score multiset. The deliberately
tie-heavy fixture may select different equally scored boundary IDs: the local
B12X tiled selector uses atomics and does not promise stable-ID tie breaking.
This differs from the separate stable DCP candidate reducer. The original v18
smoke incorrectly required identical ID sets and stopped before the reference
checks; this smoke-only correction preserves the score tolerances and changes
no inference code. Kernel timings
are diagnostic. All inherited v11-v17 GPU tests remain enabled, with the latest
source hashes checked where v18 intentionally overlays an older file.

The local host has no CUDA/PyTorch runtime, so these new GPU checks and the ARM
image build must run on the Sparks. After smoke passes, compare cold and
prefix-hit continuations at 8K, 32K, 64K and 128K with the same generated output,
cache state and inference settings. Include 1M only if it fits your KV budget.
Check retrieval across the cached-prefix boundary as well as throughput.

Pinned implementation references:
[R22 generic indexer](https://github.com/local-inference-lab/vllm/blob/70b3c1c7f1c76fcf0847fcbb4a0b8b5583b78d19/vllm/v1/attention/backends/mla/indexer.py),
[R22 B12X adapter](https://github.com/local-inference-lab/vllm/blob/70b3c1c7f1c76fcf0847fcbb4a0b8b5583b78d19/vllm/v1/attention/backends/mla/b12x_indexer.py),
[pinned B12X paged supertiles](https://github.com/local-inference-lab/b12x/blob/1e59a1fd09f782d302b1068b15c8a0bd66103894/b12x/attention/dsa_indexer/paged.py).
