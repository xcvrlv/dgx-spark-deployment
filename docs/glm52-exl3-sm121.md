# Full GLM-5.2 EXL3 on four DGX Sparks

## Status

This is an **experimental ARM64/SM121 port with fail-fast source and runtime
guards**. The deployment files are complete and statically tested, but the
container build, cold model load, and four-node GPU run still require execution
on the Spark fleet. Do not call it qualified until every gate below has passed.

As of 2026-08-28, full GLM-5.3 is an upcoming release and no compatible full
GLM-5.3 EXL3 checkpoint has been published. GLM-5.3 Flash is a different 321B
`glm5_next` architecture and remains covered by the existing Flash recipe.
The immediately runnable full-model target is therefore GLM-5.2.

## Upstream base

The recipe follows local-inference-lab's r34 GLM-5.2 R7 profile:

- checkpoint `brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78` at revision
  `9ab9579774cc432df91567a36f6e9e863e0d4c9f`;
- vLLM integration tree `b0f8c85c7b96497e0148a18230f43d18854ae04a`;
- B12X integration tree `cd3ce190f0f1917402cdfd5773724267cc9a63f8`;
- ExLlamaV3 commit `704aefd743b390af4bd0fb429d1906f9b964c7d8`;
- InstantTensor commit `49b4010afc1cae0441e71fe0b0bffc24fa05e932`.

The source locks and patches come from
[`local-inference-lab/blackwell-llm-docker@7302862`](https://github.com/local-inference-lab/blackwell-llm-docker/tree/7302862b8fcfdc7c06a411a61e1f0fb072258880).
Its r34 checkpoint profile was qualified on four RTX PRO 6000 Blackwell GPUs
in one x86 host. The published r34 image is amd64, and the later clean r34 tag
named by the current Compose file was not present in Docker Hub when checked.
Neither artifact can run directly on ARM64 DGX Spark.

## Spark adaptations

[`docker/Dockerfile.glm52-exl3-sm121`](../docker/Dockerfile.glm52-exl3-sm121)
reconstructs the immutable r34 source trees on the pinned ARM64 CUDA base. It
compiles with `TORCH_CUDA_ARCH_LIST=12.1a`, `CUTE_DSL_ARCH=sm_121a`, and
`FLASHINFER_CUDA_ARCH_LIST=12.1f`; the build fails if either reconstructed Git
tree differs from its release lock.

The node launcher makes four topology changes:

1. one visible GPU and one native vLLM rank per Spark (`TP=4`, `nnodes=4`);
2. NCCL/Gloo rendezvous over the selected CX RoCE interface;
3. B12X PCIe DMA and vLLM custom all-reduce disabled across nodes;
4. B12X retained only for rank-local EXL3 MoE and sparse MLA kernels.

The checkpoint's routed experts stay in their native mixed K3/K4/K5 Trellis
payloads. Eligible BF16 dense and shared projections are converted once to K6
and stored under `/cache/exl3-online`. Deleting that cache forces the expensive
conversion to run again.

## Bring-up sequence

```bash
bash scripts/launch-glm52-exl3.sh build
bash scripts/launch-glm52-exl3.sh prepare
bash scripts/launch-glm52-exl3.sh preflight
bash scripts/launch-glm52-exl3.sh start
```

The base recipe intentionally uses eager execution and no MTP. After it passes,
run the profiles in this order and record a benchmark after each:

1. `profiles/glm52-exl3-cudagraph.env`;
2. `profiles/glm52-exl3-mtp-1.env`;
3. `profiles/glm52-exl3-mtp-3.env`.

Keep MTP3 only if it improves generic decode after accounting for draft-token
acceptance. The r34 SM120 result does not predict Spark/RoCE performance.

## Qualification gates

- image builds natively on ARM64 and reports both pinned integration-tree
  labels;
- every rank detects SM121, imports vLLM/B12X/ExLlamaV3, and sees every model
  shard at the pinned revision;
- eager target-only startup reaches `/health` with no OOM, NCCL error, CUDA
  error, non-finite logprob, or repeated-token corruption;
- tool calling, reasoning output, prefix caching, 8k uncached prefill, and a
  256-token decode pass complete coherently;
- CUDA graph profiles show actual capture in rank-0 logs;
- MTP1 and MTP3 pass the same gates and expose a useful acceptance rate;
- a warm restart reuses the K6 cache instead of re-encoding it.

## GLM-5.3 full migration

Do not point this recipe at GLM-5.3 merely by changing `MODEL_ID`. First confirm
the released architecture, tokenizer/tool parser, tensor-parallel slice layout,
EXL3 metadata schema, MTP layer layout, and B12X sparse-attention contract. A
new checkpoint also needs its own immutable revision, served name, cache path,
and cold/warm qualification record.
