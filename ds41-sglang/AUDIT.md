# Source audit — 2026-09-12

## Top-k path

The audit uses SGLang commit **da64c5cbb8cf6bfd39be19da43573fdfd484c43a**, identified from the ARM64 image's OCI revision label, rather than assuming the preview branch tip equals the image. The image installs FlashInfer **0.6.18**. Source-level findings below apply to this pinned path; hardware correctness remains to be checked with `check-topk` and model serving.

| Location | Finding | Action |
|---|---|---|
| Checkpoint `text_config.index_topk` | Default is 512. `candidate_topk_blocks=2048` is a different setting. | Explicitly override index_topk for each script. |
| `configs/deepseek_v41.py`, HF config loader | SGLang flattens the nested text config; top-level override takes effect. | Preserve checkpoint files; pass a JSON override and validate loaded config. |
| `deepseek_v4_backend.py:826` | **Hard gate: `index_topk in (512, 1024)`** in `init_flashmla_related`. | Extend the allowed set to 2048. |
| Same file, low-ratio decode at line 3420 | **`metadata.use_topk_v2 and raw_indices is None`** sends raw-index calls back to v1. | Use v2's supported raw-output argument; retain candidate filtering. |
| `topk_v1.cuh` | **Maximum 1024**. Raising a model setting does not raise this limit. | Force `SGLANG_OPT_USE_TOPK_V2=1` and both target/draft SGL kernel selectors. |
| `topk_impl.cuh`, `topk_v2.cuh` | Runtime maximum **2048**, supported for paged and ragged output. | No kernel capacity edit needed. Build checks capacity. |
| Ratio-1/2 metadata and `dsv41_sparse.py` | Buffers, selected widths and length clamps use `index_topk`; no separate 512 truncation found. | Keep dynamic sizing. |
| Dense prefill | Allocates `(num_tokens, index_topk)`, calls ragged v2. | No lower gate found. |
| Prefill CUDA graphs and Torch fallback | `min(index_topk, width/lmax)` limits selection to existing history. | Keep these bounds; they are required for short histories. |
| Decode graph variants | Full-attention shortcuts end at thresholds derived from configured top-k and compression ratio. | No fixed 512 threshold found. |
| Candidate stage | Selects 2048 blocks of eight positions; subsequent token selection uses index_topk. Fewer reachable positions can legitimately produce fewer valid indices. | Keep the model's candidate policy. |
| SM120 FlashInfer bridge | SWA indices are the primary input (128); compressed indices go through `extra_indices`. Scratch splits include the full extra width. | No bridge truncation found. |
| FlashInfer 0.6.18 decode | Primary DSv4 specialization list stops at 1024, **but this path's primary width is 128**. `extra_topk` comes from the extra tensor and is traversed in chunks. | A 2048 primary specialization is unnecessary for this two-cache path. |
| FlashInfer prefill | Extra-cache width is a runtime length, separate from the primary top-k specialization. | No fixed 512/1024 cap on compressed selection found. |

The other 512-only metadata gate in `deepseek_v4_backend_hip_radix.py` belongs to AMD and is not used on GB10. Attention head dimension 512, SWA length 128, expert routing top-k, and candidate block count are separate quantities and are not changed.

Sources: [checkpoint config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/fb2764a5cf321eaa5070ca8f9e892818f477c16d/config.json), [SGLang attention backend](https://github.com/sgl-project/sglang/blob/da64c5cbb8cf6bfd39be19da43573fdfd484c43a/python/sglang/srt/layers/attention/deepseek_v4_backend.py), [v2 kernel limits](https://github.com/sgl-project/sglang/blob/da64c5cbb8cf6bfd39be19da43573fdfd484c43a/python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/topk_impl.cuh), [graph variants](https://github.com/sgl-project/sglang/blob/da64c5cbb8cf6bfd39be19da43573fdfd484c43a/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py), [FlashInfer dispatcher](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.18/flashinfer/mla/_sparse_mla_sm120.py), [FlashInfer DSv4 kernel](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.18/include/flashinfer/attention/sparse_mla_sm120/decode_dsv4_kernel.cuh).

## RoCEnante

**Feasible as a separate SGLang adapter; not a drop-in environment switch.** B12x's `b12x.comm.roce` API provides all-reduce and all-gather independently of vLLM. The maintained serving integration and reported measurements target the local-inference-lab vLLM adapter. Setting `VLLM_ENABLE_ROCE_ALLREDUCE` in this SGLang image would not connect that adapter.

The natural integration point is SGLang's `GroupCoordinator` in `distributed/parallel_state.py`: initialize transport using its CPU/Gloo group, route eligible TP all-reduces and all-gathers, and keep other collectives on the existing backend. Initial bounds could match the existing fleet: 2 MiB reduce / 16 MiB gather. Group identity, rank-invariant eligibility, CUDA capture preparation, stream ordering and teardown need explicit handling. Runtime transport failures must stop the rank; independent fallback to NCCL can hang the fleet. Health must be checked before returning tokens, including DSpark asynchronous output paths.

The expected benefit is reduced latency for small decode collectives; this is an inference, not a measured DeepSeek/SGLang result. Acceptance requires TP4 correctness and graph-replay tests, failure injection, then the same 512/2048 serving benchmarks with NCCL and RoCEnante. The delivered profiles use NCCL over RoCE.

Sources: [B12x transport API and qualification](https://github.com/local-inference-lab/b12x/blob/081b235931dbbcedcf0eb5899bae990c5dec5238/docs/rocenante.md), [SGLang GroupCoordinator](https://github.com/sgl-project/sglang/blob/da64c5cbb8cf6bfd39be19da43573fdfd484c43a/python/sglang/srt/distributed/parallel_state.py).
