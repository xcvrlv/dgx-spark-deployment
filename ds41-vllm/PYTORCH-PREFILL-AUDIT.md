# Text prefill audit: JJ + b12x on SM121

Checked 2026-09-13 at 11:35 UTC against live upstream heads: JJ
`b40673cd006bf3496fdd70361dad2ad29eff54e7` and b12x
`9043b448622764a598969518d413b3fd8b3c0c07`. Both match our pins; no newer
upstream revision supersedes these findings. This is a source audit, not a
Spark profiler capture or an audit of all PyTorch internals.

The claim that quadratic prefill work can become linear is true for one
avoidable prefix-hash copy in this stack. That operation is Python in vLLM,
not a PyTorch kernel. I did not find a growing `torch.cat` KV cache, dense
PyTorch causal mask, or generic PyTorch attention implementation in the native
V4.1 **text** route. GPU page-table duplication is the stronger next kernel
candidate. Exact full-scan indexer scoring still performs quadratic work over
a growing prefix; changing its complexity would require a separate algorithmic
argument, not just replacing a PyTorch operation.

Let N be prompt tokens, M the configured context cap, C the prefill chunk size,
and B the KV block size. Comparisons below assume fixed C and B. Distinguish
quadratic scaling when M grows with N from linear scaling at a fixed M.

| Path | Actual implementation and cost | Assessment |
| --- | --- | --- |
| Prefix-cache hash registration | Python copies all remaining prompt hashes every chunk: O(N² / (CB)) cumulative | Exact bounded-copy replacement gives O(N/B) copied references |
| Indexer page-table expansion | Triton replicates a cap-width page row for every query: O(NM/B) stores | Redundant metadata; candidate for shared rows/request-indexed lookup |
| Runner block-table gather | Triton copies valid pages and clears the remaining cap-width row each scheduler step | Incremental updates already exist upstream; changing gathers requires request-lifetime and graph-padding tests |
| Full-scan sparse indexer | Native b12x compares each query with its visible compressed prefix | O(N²) total scoring in full-scan layers; top-512 output does not avoid candidate scoring |
| Engram ID transfer | PyTorch CUDA-to-pinned-CPU copy plus event synchronization, O(current batch lookups) | Potential latency bottleneck, not quadratic prefix work |
| Engram n-gram lookback | Fixed-depth three-token history and native hashing | Already bounded per token |

## Ready, isolated CPU patch

`patches/prefill_hashes.py` ports just the bounded slice from the preserved
`sparkrun-glm53-exl3/overlay/patch_r22_v17.py`. It applies equally to this
MXFP4 deployment: it changes cache bookkeeping, not EXL3 compute.

In `BlockPool.cache_full_blocks`, change:

```python
new_block_hashes = block_hashes[num_cached_blocks:]
# to
new_block_hashes = block_hashes[num_cached_blocks:num_full_blocks]
```

Only newly full blocks consume these hashes. Null blocks, sparse block masks,
partial-to-full promotions and event indices retain their original behavior.
The same bound also avoids unnecessary Python calls in the scaled block-hash
view. [JJ cache method](https://github.com/local-inference-lab/vllm/blob/b40673cd006bf3496fdd70361dad2ad29eff54e7/vllm/v1/core/block_pool.py#L241),
[scaled hash view](https://github.com/local-inference-lab/vllm/blob/b40673cd006bf3496fdd70361dad2ad29eff54e7/vllm/v1/core/kv_cache_utils.py#L2630).

At 393216 tokens, 4096-token chunks and 256-token blocks, the CPU test measures
74496 copied hash references before versus 1536 after: **48.5 times fewer
references**, not 48.5 times faster prefill. This is for one group's full
uncached prefill with regular chunks. Mixed scheduling, cache hits, sliding
windows and other block sizes change the counts. The absolute cost may be
small compared with GPU scoring and Engram I/O.

The patch is prepared separately; the default Dockerfile and image tag still
contain only the previous communication overlay. It rejects any source drift,
supports idempotent application and exact `--revert`. To evaluate, create a
separately tagged child image from the communication image with this Dockerfile
in the `ds41-vllm` build context:

```dockerfile
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY patches/prefill_hashes.py /opt/ds41/patches/prefill_hashes.py
RUN package="$(python3 -c "from importlib.metadata import distribution; print(distribution('vllm').locate_file('vllm'))")" \
    && python3 /opt/ds41/patches/prefill_hashes.py "$package" \
    && python3 /opt/ds41/patches/prefill_hashes.py "$package" --check
LABEL local-inference.prefill-hash-overlay="ds41-bounded-hashes-v1"
```

Run `python3 check-upstream.py` before building this candidate. Use a new tag
such as the existing communication tag plus `-hashes-v1`; distribute the same
image to all four hosts. Roll back by selecting the communication-only image,
or use `prefill_hashes.py <vllm-package-directory> --revert` in a separate
rollback image build. Do not modify files in a running server.

Validation: 20 repository CPU tests passed, including actual upstream cache
method execution before/after with recorded insertions, removals and event
arguments; null/masked blocks; promotions; unscaled and scaled hash views;
copied-reference counts; source drift rejection; reapplication and rollback.
Dependencies are recording doubles, not a live cache allocator. No ARM64
image build, CUDA execution or fleet speed test was performed for this patch.

## Most promising next GPU investigation: shared page rows

JJ `_pages` materializes an identical page-table row separately for every
query belonging to the same request. Prefill uses indexer chunks of 256 rows.
The width is derived from the configured context cap, not the current prompt
length. At our cap, that is 1536 int32 entries per query. Over a full 384K
prompt this represents about 2.25 GiB of table writes **per layer that runs
this expansion**, excluding reads and other work. This is an operation-count
estimate, not measured traffic or a latency estimate.
[JJ expansion and indexer dispatch](https://github.com/local-inference-lab/vllm/blob/b40673cd006bf3496fdd70361dad2ad29eff54e7/vllm/models/deepseek_v4_1/attention.py#L698).

b12x already accepts either one shared page-table row or one row per query,
and uses zero row stride for the shared case. Homogeneous single-request
chunks could exploit this without changing selected tokens. Simply sharing
within each chunk removes up to a factor of 256 in table copies; it does
**not** change the asymptotic N-versus-M scaling. Moving to persistent
per-request tables or direct request-indexed access is the larger candidate.
[b12x shape validation](https://github.com/local-inference-lab/b12x/blob/9043b448622764a598969518d413b3fd8b3c0c07/b12x/attention/dsa_indexer/mxfp4.py#L1164),
[shared-row stride](https://github.com/local-inference-lab/b12x/blob/9043b448622764a598969518d413b3fd8b3c0c07/b12x/attention/dsa_indexer/mxfp4.py#L1252).

Do not pass the original block table through blindly: JJ normalizes reserved
page zero to -1, handles invalid/padded request IDs, and can mix requests in a
chunk. Any implementation must preserve those semantics, causal cache lengths,
prefix reuse, preemption and request-slot reuse. Test single and mixed requests,
partial final chunks, all Full/Reindex/Reuse layer paths and graph replay.
For c16, a single-request fast path alone may miss much of the workload.

## What is already optimized, and what cannot just be removed

JJ narrows full-scan `score_width` to the active prefix during non-captured
extend execution. Reindex layers operate on bounded candidates and reuse
layers reuse selections. Full-scan score work remains: returning only 512
indices does not mean evaluating only 512 candidates. Increasing index_topk
to 1024 is not a prefill optimization.
[active score width](https://github.com/local-inference-lab/vllm/blob/b40673cd006bf3496fdd70361dad2ad29eff54e7/vllm/models/deepseek_v4_1/attention.py#L698).

The worker already stages newly appended block IDs. Its subsequent gather
and zero-padding maintain persistent buffers for scheduling and capture.
An incremental rewrite needs validity tracking rather than deleting clears.
[worker block tables](https://github.com/local-inference-lab/vllm/blob/b40673cd006bf3496fdd70361dad2ad29eff54e7/vllm/v1/worker/gpu/block_table.py).

Engram's `ids_host.copy_(..., non_blocking=True)` is followed by an event wait
before native storage reads. That wait establishes readiness of IDs and prior
cache readers; removing it creates a race. Pipelining needs explicit buffer
ownership and events. First measure wait time and disk/cache hit behavior.
[Engram storage transaction](https://github.com/local-inference-lab/b12x/blob/9043b448622764a598969518d413b3fd8b3c0c07/b12x/sequence/_shared/disk_table.py#L269).

## Fleet measurement before promoting changes

Hold communication options, context cap, memory utilization and chunk size
constant while comparing the hash-only child image with the base. Use uncached
prompts at 32K, 128K, 256K and near 384K (leave room for output); start at c1
and repeat at c16 within the available KV capacity. Use `benchmark.py` for
TTFT/throughput and capture a short PyTorch CPU/CUDA profiler or Nsight trace
of worker prefill windows. Include worker scheduling and Engram wait time:
GPU kernel totals alone miss both CPU hash overhead and host I/O stalls.
Inspect `_pages`, block-table gathers, indexer kernels and ID-transfer waits.
Profile small windows separately from full throughput runs to avoid trace
overhead. A/B 4096 versus 8192 batch tokens separately, after establishing the
same-image baseline; c16 plus 0.88 remains an unvalidated memory-capacity trial.
