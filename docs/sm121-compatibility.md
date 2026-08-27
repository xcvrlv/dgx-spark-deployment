# GLM-5.3 Flash SM121 compatibility notes

The initial recipe deliberately separates the proven non-ring serving profile
from speculative decoding. It serves the pinned
`local-inference-lab/GLM-5.3-Flash-NVFP4` checkpoint with TP4, the model-native
1,048,576-token context, an explicit 8 GiB FP8 KV slab, FlashInfer B12X linear
kernels, FlashInfer CUTLASS MoE, FlashKDA prefill, eager execution, and no MTP.
Once coherent output, finite logprobs, tool calling, and a real long prefill
pass, MTP can be enabled independently in the recipe.

The backend, scheduler, loader, and checkpoint settings are adapted from the
[FujitsuPolycom TP4 Spark service contract](https://github.com/FujitsuPolycom/glm53-flash-tp4-spark/tree/c822fae8394fcaacd47db1fd6fdd7df67cb822b0).
Its SparkRing transport, paired-HCA topology, forced NCCL ring algorithm,
patched NCCL preload, and ring-specific QP assertions are intentionally not
used here. This repository retains its switch-fabric CX0 transport settings.

## Patch inventory

`docker/Dockerfile.glm53-sm121` consolidates the current GLM-specific fixes on
top of vLLM's pinned ARM64 CUDA 13.0 day-zero image:

1. Expose the NoPE sparse-MLA backend on SM121 and select its FA2 kernel away
   from Hopper. GLM's `pe_dim=0` is incompatible with the DeepSeek-specific
   packed SM120 cache path.
2. Pin FlashInfer 0.6.18. The earlier FA2 MLA path can produce non-finite
   results for ordinary SM121 batch shapes.
3. Restore NCCL 2.30.7 and CUTLASS DSL 4.6.2 after the FlashInfer install.
   Unchecked dependency resolution can otherwise break RoCE rendezvous or
   CuTeDSL warmup.
4. Disable PDL on SM12x for GLM's recurrent KDA state kernels.
5. Initialize sparse-indexer top-k tails to `-1` and bounds-check expanded pool
   IDs to prevent uninitialized KV gathers.
6. Cap the FP8 MLA tile to the shared-memory size selected for GB10 and admit
   capability 12 in FlashInfer's wrapper.
7. Install InstantTensor for direct-I/O weight loading, then re-pin NCCL.
8. Raise the container `nofile` limit to 1,048,576. InstantTensor shard loading
   and NCCL connection setup can otherwise exhaust Docker's inherited limit and
   fail the entire TP group with `ncclOsSocketTryAccept: Too many open files`.
9. Use Marlin for NVFP4 MoE in the first iteration. FlashInfer CUTLASS triggers
   a 97-object `fused_moe_120` JIT on SM121 and has also produced silently
   incorrect repeated-token output in independent GLM-5.3 Spark deployments.

Every source edit is guarded and the image build fails when its expected
upstream source no longer matches. That is intentional: a changed upstream
file needs review rather than silently receiving a stale patch.

## References

- [Official vLLM GLM-5.3 Flash recipe](https://recipes.vllm.ai/zai-org/GLM-5.3-Flash)
- [NVIDIA multi-node vLLM guide](https://build.nvidia.com/spark/vllm/multi-node)
- [local-inference-lab b12x](https://github.com/local-inference-lab/b12x)
- [SM121 deployment investigation used for the guarded patches](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-2x-DGX-Spark)
- [Proven TP4 profile used for non-ring serving settings](https://github.com/FujitsuPolycom/glm53-flash-tp4-spark)

The local-inference-lab `dev/infernal-invocation` vLLM branch is still the
preferred future B12X base, but at the pinned revision it does not yet contain
`glm5_next`. The first deploy therefore uses vLLM's GLM-5.3 integration and the
known Marlin fallback while retaining the local-inference-lab checkpoint.
