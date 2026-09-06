# v15: SM121 compute and switched RoCE

Build on the head Spark from the repository root:

```bash
bash sparkrun-glm53-exl3/scripts/build-r22-v15-image.sh WORKER1 WORKER2 WORKER3
```

Use `recipes/glm53-exl3-v15-4x.yaml`. Memory utilization remains **0.87**.
This is a late overlay on v14, with five independent switches enabled in the
image and recipe. Restart all workers after changing a switch.

| Change, in expected significance order | Disable with |
| --- | --- |
| Compact shared input rotations: write one gate/up row per token and have FC1 map each top-k route to that token. At top-k 8, rotation output writes fall from eight copies to one. The existing route-sized allocation is retained because FC2 aliases it. | `VLLM_GB10_COMPACT_INPUT=0` |
| SM121-specific BF16 skinny projection profiles for M=1 and M=2. Covers eligible unquantized QKV-A projections and the MTP EH projection; keeps the existing K6 weight path. Larger batches use the original backend. | `VLLM_GB10_SKINNY_GEMM=0` |
| Corrected SFU reciprocal in routed-expert sigmoid, using an approximate reciprocal and two fused FP32 correction operations. Exceptional/subnormal-range denominators retain division. | `VLLM_GB10_SIGMOID=0` |
| Inline small RoCE payload stripes into the work request, avoiding the NIC payload fetch. Requests 64-byte inline capacity, retries 16 if unavailable, and checks the provider-returned limit. | `B12X_ROCE_INLINE_PAYLOAD=0` |
| Rank-rotated destination order for switched-fabric fanout, spreading the first posts across destinations instead of converging on low-numbered ranks. | `B12X_ROCE_BALANCED_FANOUT=0` |

All five disabled recover v14's paths. Compact input additionally requires
v14's shared-input rotation path, SM121, shared scales, non-coupled rotations,
and compiled capacity above 64. It changes FC1 read addresses and rotation
write addresses together. FC2 addresses, routed output buffers, all phase
barriers and quantized payload bytes remain as before. The compile keys
include layout and arithmetic variants.

RoCEnante already posts asynchronously to every peer over both HCAs. It is
not a ring, and a one-hop QSFP switch does not make port bandwidth unlimited.
The proxy changes preserve the payload-then-flag work-request chain, per-rail
completion flags, sequence/slot protocol, outstanding-write accounting and
fixed-rank floating-point reduction order. They do not remove fences or wait
for each peer before posting to the next. The changed C proxy is compiled
and loaded during image construction; its cache name includes its source hash.

The activation audit found an already fused routed-expert rotation/SiLU
epilogue with fast exponential, and a native fused `SiluAndMul` path for the
ordinary/shared MLP. There is no evidence here of a multi-operation Python
activation fallback. The reciprocal change targets smaller arithmetic overhead.
The skinny kernel uses SIMT FP32 accumulation, with a separate SM121 profile;
it does not enable the SM103-only fused-A operation. Both changes can alter
last-bit rounding, so acceptance and model output quality still need observation.

Changing expert tile dimensions is not automatically a utilization win:
FC1/FC2 share a cooperative grid, thread count and register/shared-memory
budget. Smaller N tiles can increase tile count or conflict with FC2 geometry.
This release preserves the established K128/N128 FC1 and K32/N512 FC2 tiles
and the prefill block M32 setting.

Validation: five new CPU tests pass, covering compact buffer writes and the
three actual FC1 route address expressions, reciprocal correction, SM121
profile/fallback selection, pinned-source preflight and build integration.
The reciprocal CPU check models the instruction's seed-error envelope; it
does not execute the GPU instruction. GPU tests retain v13/v14 exact checks,
compare full mixed-K compact/noncompact layouts, check sigmoid values and
full mixed-K activation outputs, and compare skinny GEMM to FP32 reference
matmuls with graph replay. Floating-point changes use bounded numerical
tolerances; layout changes retain exact equality. Timings are diagnostic.

The optional four-node command from [v10](r22-performance-v10.md#four-node-communication-check)
can use image `r22-dflash2-sm121-v15` and script
`/opt/compose/smoke_r22_v15.py --distributed`. It checks MTP packets and DCP
collectives with the new proxy. The normal per-node build tests do not exercise
RDMA between hosts. GPU execution, native ARM compilation, throughput and
MTP acceptance have not been measured on the development machine.

Sources: [pinned cooperative kernel](https://github.com/local-inference-lab/b12x/blob/1e59a1fd09f782d302b1068b15c8a0bd66103894/b12x/moe/_shared/kernels/w4a16/kernel.py),
[pinned RoCE proxy](https://github.com/local-inference-lab/b12x/blob/1a7e3ec286b0ff0b7c2aabee22dce08daab7e011/b12x/comm/roce/_roce_proxy.c),
[projection dispatcher](https://github.com/local-inference-lab/vllm/blob/70b3c1c7f1c76fcf0847fcbb4a0b8b5583b78d19/vllm/models/deepseek_v32/nvidia/glm52_low_latency_gemm.py),
[NVIDIA PTX reciprocal and FMA semantics](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html).
