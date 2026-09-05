# v11 experimental mixed-K and MTP candidate

v11 extends v10. It is ready for a Spark build, **not benchmark-qualified**.
No CUDA compilation, GPU numerical test, cluster test, or throughput result
has been obtained on the Windows development host.

## Findings

[MiaAI PR77](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/pull/77)
adds sorted/batched fat-expert execution and a dedicated K4 MCG CUDA kernel.
Its native batched path requires compatible shared input rotations. The final
reported cold-prefill improvement is about 20–21% in its two-Spark setup;
earlier cache-contaminated results were withdrawn. This is not an estimate for
our four-Spark mixed-K/MTP3 setup. Its CUDA kernel explicitly rejects other K.

Our pinned B12X mixed-Trellis implementation already packs routes by expert,
dispatches K3/K4/K5 per projection, and combines FC1, activation and FC2 in a
cooperative grid. It supports separate gate/up/down tier assignments. Replacing
it with PR77's K4 kernel would not preserve the checkpoint. The existing
prefill block size is already 32; the recipe's old prefill-chunk value of 128
does not divide this mixed path into 128-token launches. Its capacity is 4096.

The B12X mixed-Trellis path history checked through September 6 has no newer
mixed-K implementation than the one included here. New dense NVFP4/A16 tuning
does not directly accelerate the packed Trellis main-model path. The EXL3 fork
also has newer native GEMV and standalone sampling work, but the active mixed
MoE and online-K6 dense paths use B12X. The native FP16 accumulation change is
SM86-specific; the experimental int8 GEMV path is not graph-capturable. v11
therefore retains the extension pin and the weight-loading/quantization paths.

## Implemented changes

1. **Mixed-K final-store fusion.** Both two-tier and three-tier launch plans
   can store the final MoE result directly as BF16/FP16. Route values remain
   FP16, rotations and weighted reduction retain their existing arithmetic,
   and accumulation remains FP32. Conversion occurs only at the final store.
   The existing vLLM `.to(x.dtype)` then needs no conversion launch or copy.
   This applies to prefill and decode, including MTP calls that use this path.
   Shared and per-expert rotation tables use the same implementation.
2. **[RoCEnante PR315](https://github.com/local-inference-lab/b12x/pull/315)**,
   pinned to `b58f34eaf978277621efced6678e6713fd7122e4`. All-gather uses a
   size-dependent launch grid with matching arrival counters, and one system
   fence per block after staging. This targets small gathers such as MTP
   logits/top-k and DCP metadata. The two changed files match upstream exactly.

For a 4096×6144 output, fusion avoids approximately 192 MiB of output memory
traffic per affected layer per rank (FP32 write plus subsequent FP32 read).
This is a traffic calculation, **not a measured latency gain**. Matmuls,
rotation scratch, route packing and communication can still dominate.

The output scratch dtype follows immutable compiled-plan metadata, including
on compiler-cache hits. BF16, FP16 and legacy FP32 have separate cache keys.
The final output is now a view of reusable per-layer scratch; the current
chunked vLLM caller copies each chunk before reuse. Callers must consume it
before reusing that plan. No intermediate is rounded earlier to obtain speed.

## Build and run

On the head Spark:

```bash
bash sparkrun-glm53-exl3/scripts/build-r22-v11-image.sh WORKER1 WORKER2 WORKER3
```

The script builds/reuses the pinned v10 base, builds the small v11 layer, then
runs v10 and v11 GPU smoke checks on each Spark and verifies distributed image
identity. Use `recipes/glm53-exl3-v11-4x.yaml`. The existing v10 recipe and build
script retain their defaults. Weight location, TP4/DCP4, MTP3, batch capacity,
graph sizes and online K6 policy remain the same for comparison.

The v11 GPU smoke compares fused BF16 and FP16 stores with the FP32 result
followed by PyTorch conversion, requiring exact equality. It covers M=1,4,32,
128,2048, shared/per-expert output rotations, permuted/missing expert routes,
both route-id dtypes, and graph replay with changing inputs. It tests the
changed reduction kernel, not full mixed-K weight decoding or model quality.
The build stops if these checks fail. First use may compile CuTe kernels.

For four-node transport qualification, run `/opt/compose/smoke_r22_v11.py
--distributed` under the cluster's four-node `torchrun` launch, with the same
host networking, IPC, RDMA device mounts and fabric environment as serving.
This exercises DCP gathers and reductions at different message sizes inside
graphs. It is an explicit cluster check, not part of the per-node build smoke.

## Compare and roll back

- Run v10, v11 with `VLLM_EXL3_MIXED_FUSED_OUTPUT=0`, and v11 with it set to `1`.
  Restart every worker after changing this variable; it is read at plan setup.
  The off setting restores the old FP32 output path while retaining PR315.
- Use unique cold prompts, identical input/output lengths, MTP3 and concurrency.
  Keep prefill capacity and graph shapes fixed. Collect several alternating
  runs at short, medium and long contexts; report medians and variation.
- Record prefill tokens/s, decode tokens/s, TTFT, accepted speculative tokens
  per draft cycle and draft/verification time. A decode gain with falling MTP
  acceptance is not sufficient evidence of improvement. Check generated output
  and server logs for fallback or graph errors.
- Revert to the v10 recipe/image to remove both v11 changes. Keep the v10 image
  on the workers if immediate rollback is required.

Potential regressions include CuTe compiler/typed-store behavior, altered
scratch lifetime assumptions, and size-dependent RoCE synchronization. The
numerical intent is unchanged, but end-to-end acceptance and speed remain
unverified. There is no claimed PR77-sized mixed-K speedup.

CPU validation: five v11 tests cover hash/idempotence/preflight behavior, actual
compiler-function dtype/cache handling, both tier callers and RoCE counter/grid
selection, plus the separate build/recipe GPU gates. The existing seven v10 tests and nine recipe/image tests also pass.
Set `GLM53_V11_BASELINE` to the pinned composed-v10 B12X package to run the v11
source tests outside this checkout's ignored research fixture directory.
