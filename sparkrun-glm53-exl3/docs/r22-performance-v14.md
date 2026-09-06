# v14: mixed-K prefill and greedy MTP on GB10

Build this candidate on the head Spark, from the repository root:

```bash
bash sparkrun-glm53-exl3/scripts/build-r22-v14-image.sh WORKER1 WORKER2 WORKER3
```

Then use `recipes/glm53-exl3-v14-4x.yaml`. The image is
`spark-vllm-glm53-exl3:r22-dflash2-sm121-v14`. The wrapper builds/reuses v10
through v13, applies the late v14 overlay, and distributes the exact image ID
to the three workers. GPU memory utilization remains **0.87**. The supplied
recipe retains MTP3, prefill block M32, and the v13 CKV settings.

The user observed that v13 recovered some prefill throughput, leaving a gap
of less than 100 tokens/s to the earlier safe image. This is an experimental
candidate; no new end-to-end speedup or explanation of that remaining gap
has been established locally.

## Changes

1. **Shared-H mixed-K prefill rotation reuse.** The existing cooperative
   kernel repeats the same gate/up input rotations for each selected expert
   when SUH is shared. The new branch computes them once per token/H128 block
   and fills all existing route rows. With top-k 8, this removes seven of
   eight identical butterflies, but does not reduce route-buffer stores or
   the expert GEMM work. FP16 rounding boundaries, FP32 butterflies, distinct
   gate/up scales, route layouts and phase barriers are preserved. It applies
   only to SM121, shared input scales, and compiled row capacity above 64.
   Per-expert scales, coupled rotations and small decode plans retain their
   original path. Both two-tier K3/K4 and three-tier K3/K4/K5 use the same
   changed driver. The compile key includes the variant to prevent stale
   persistent kernels. The restored TR3/R7 loader marks its routed layers
   as using shared H-side rotations; arbitrary EXL3 checkpoints need not do so.
2. **Enable the existing local argmax protocol for greedy MTP.** The active
   V2 runner and NVIDIA DeepseekV32 MTP model already implement it, but v13's
   recipe explicitly disabled it. Each rank still evaluates its own LM-head
   weight shard; token selection gathers only local winners. No target-model
   sampling or MTP normalization changes are introduced. This is a recipe
   activation of existing support, rather than a new MTP algorithm.
3. **Use 16-byte MTP winner packets.** The stock FP32 value/token pair is
   eight bytes per row. RoCEnante accepts it through its padded path, which
   needs staging, layout conversion and a separate scratch arena. Four FP32
   fields per row (value, token ID, zero, zero) meet its direct-gather layout.
   Both the stock PyTorch reduction and the new reduction use this layout.
   The padded arena is otherwise allocated at the configured maximum gather
   size, not the tiny packet size, and first allocation inside graph capture
   is refused. This change avoids that dependency for MTP winner gathers.
4. **Split-vocabulary SM121 argmax.** A new Triton path distributes a small
   draft batch's vocabulary reduction across more CTAs, then emits the aligned
   packet and reduces gathered winners. It retains the existing LM-head GEMM,
   scale and soft-cap operations. It handles original-vocabulary padding,
   first-index ties, NaNs, and all-negative-infinity inputs. Token IDs remain
   exact in FP32 within the guarded range. Other devices/dtypes and unsupported
   shard layouts keep the PyTorch reduction. More launches can outweigh the
   parallelism benefit; the smoke prints isolated timings without imposing
   a speed threshold.
5. **Repair and extend the smoke checks.** The v13 smoke directly imported
   `exllamav3_ext`, whereas only the serving recipe supplied its directory on
   `PYTHONPATH`. Separate `docker run` smoke processes do not inherit an
   earlier smoke's `sys.path`. The v14 entrypoint invokes the serving extension
   loader with `/opt/exllamav3` as the default path, honoring the ABI-shim
   setting. It then runs v13's numerical checks, including v11's fused-output
   checks. Its own source verifier handles the changed kernel hash while
   still verifying unchanged inherited files. Missing symbols, ABI errors,
   and numerical mismatches remain fatal.

The two new GPU kernel switches are evaluated before compilation/capture;
restart **all four workers** after changing them.

| Control | Result |
| --- | --- |
| `VLLM_GB10_SHARED_INPUT_ROTATION: "0"` | Original per-route prefill rotations. |
| `VLLM_GB10_DRAFT_ARGMAX: "0"` | PyTorch local reduction, retaining aligned winner packets. |
| `"use_local_argmax_reduction": false` in speculative config | Original full-vocabulary draft-logit gather; bypasses both winner optimizations. |

For a v13-equivalent code-path control within v14, turn off the shared-input
switch and set local argmax reduction to false. Keep the same model snapshot,
MTP steps, graph sizes, prompt, concurrency, output length and cold-prefix-cache
state. Compare acceptance and accepted tokens per cycle alongside decode
throughput. A change in MTP acceptance can confound a speed comparison even
when all new arithmetic is equivalent.

## Validation and limits

Local validation: **25 targeted CPU tests pass** across v10–v14, using the
exact pinned source fixtures. Python compilation, Bash syntax and whitespace
checks pass. The argmax tests execute the actual Triton functions through a
NumPy operation shim, including padding across block boundaries, NaNs, ties
and simulated TP4 rank order. A separate test executes the actual shared and
legacy route loops to check address coverage and missing/padded routes.
Source checks reject unknown inputs before writing any file and verify
reapplication. The new logits source was independently checked against the
R22 Git blob `09ae52b9d982f8d119afd4cb5a79fcf953fd7fc4`.

The image build checks extension loading without a GPU. The distribution
script runs the inherited image/CKV checks and v14's GPU checks on every
Spark before reporting success. The new checks compare:

- FP16/BF16 shared input rotations against the original driver, including
  padding, non-local routes, tail rows, and changed inputs during graph replay.
- Full K3/K4 and K3/K4/K5 cooperative kernels at hidden size 6144, block M32,
  top-k 8, with separate compiled variants and graph replay after changing
  input values and route order.
- Argmax against full-vocabulary PyTorch selection at multiple dtypes and
  batch sizes, plus graph replay and isolated timing.

No CUDA compiler, GB10 GPU, ARM64 Docker build, model throughput test or
four-node transport test was available on the development machine. These
CPU results cannot establish GPU compilation, race freedom, or a speedup.
The new GPU smoke code itself must still run on the Sparks.

An optional real TP4 transport check is available after distributing v14.
Use the four-host command in [the v10 transport instructions](r22-performance-v10.md#four-node-communication-check),
with the image changed to `r22-dflash2-sm121-v14` and the script changed to
`/opt/compose/smoke_r22_v14.py --distributed`. It runs the installed logits
processor with both local reductions through RoCEnante, replays graphs with
changing winning ranks, and checks that the padded gather arena was never
allocated. The ordinary per-node build smoke simulates TP4 locally and does
not replace this collective test.

## PR #131 and source basis

[PR #131](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/pull/131)
reports isolated K4 fat-expert kernel and indexer temporary-memory improvements.
Its M64, paired projections, fused activation and asynchronous staging changes
target a different implementation; they are not a patch that can be applied
unchanged to this mixed-K megakernel. It explicitly does not establish a matched
full-model speedup. Our mixed-K implementation already pairs projection work
within the cooperative grid and fuses activation. The R22 B12X indexer uses its
paged top-k path, not that PR's Python loop over a full logits tensor, so its
`del logits` patch is not applied here.

- [Pinned mixed-K driver and input rotations](https://github.com/local-inference-lab/b12x/blob/1e59a1fd09f782d302b1068b15c8a0bd66103894/b12x/moe/_shared/kernels/w4a16/kernel.py)
- [V2 draft sampling protocol](https://github.com/local-inference-lab/vllm/blob/70b3c1c7f1c76fcf0847fcbb4a0b8b5583b78d19/vllm/v1/worker/gpu/spec_decode/speculator.py)
- [NVIDIA MTP logits and normalization](https://github.com/local-inference-lab/vllm/blob/70b3c1c7f1c76fcf0847fcbb4a0b8b5583b78d19/vllm/models/deepseek_v32/nvidia/mtp.py)
- [RoCEnante direct and padded gather contracts](https://github.com/local-inference-lab/b12x/blob/b58f34eaf978277621efced6678e6713fd7122e4/b12x/comm/roce/roce_oneshot.py)
