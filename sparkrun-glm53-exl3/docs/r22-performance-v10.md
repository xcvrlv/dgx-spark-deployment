# GLM-5.3 EXL3 R22 v10 performance candidate

The build script and MTP3 recipe now select
`spark-vllm-glm53-exl3:r22-dflash2-sm121-v10`. The image retains DFlash2 support;
the recipe uses the checkpoint's native MTP3 heads.

## Changes

1. **Restore full-CKV prefill for GLM-DSA.** Stock R22 only enables its gathered
   cache path for `glm5_next`/`glm5_next_text`. This checkpoint is `glm_moe_dsa`.
   The overlay adds unpooled top-k metadata, uses the existing native 656-byte
   FP8 cache gather and index mapping, and reserves its workspace before KV
   profiling. The NVIDIA attention caller keeps its 16 local query heads and
   skips the DCP query gather and output combination when the complete cache
   is available. Ordinary decode, mixed batches, and oversized contexts keep
   the existing DCP path. GLM-DSA NVFP4 and prefill-context parallelism are not
   enabled by this change.
2. **Use RoCEnante for DCP communication.** The original communicator only
   created RoCEnante for TP groups. The overlay also allows the separate DCP
   group when explicitly enabled. Other single-host custom transports remain
   restricted to TP. This makes small query, FP32 LSE and the B12X indexer's
   global top-k candidate gathers eligible for the existing RoCEnante
   implementation.
3. **Gather query heads without a layout copy.** RoCEnante accepts dimension
   zero or the last dimension. A contiguous `[tokens, heads, width]` query is
   viewed as `[tokens, heads*width]`, gathered directly into the concatenated
   layout, then viewed back. This preserves rank/head order and avoids the
   NCCL gather's intermediate layout conversion.
4. **Bounded DCP reduce-scatter substitution.** Up to 256 KiB, the DCP
   communicator uses RoCEnante all-reduce and extracts its head slice. That
   covers a single MTP3 request's four BF16 verification rows at 64 heads and
   value width 512. Larger outputs retain NCCL reduce-scatter. Batch-invariant
   mode also retains NCCL. The 256 KiB crossover is a conservative candidate,
   not a measured optimum; setting it to zero independently disables this
   substitution.

The mixed-K EXL3 implementation already uses its one-grid prefill plan for
batches up to `VLLM_EXL3_PREFILL_CAPACITY=4096`. Its older `PREFILL_CHUNK=128`
setting does not split this path into 128-token launches. The K3/K4 weight
formats, K6 conversion, Trellis tile choices, MTP depth and graph sizes remain
as in v9. The NVIDIA GLM-DSA implementation does not consume the older
`VLLM_DCP_Q_REPLICATE` setting, so the comparison recipe removes it.

The full-CKV gather capacity is reduced from 524,288 to 131,072 logical tokens
to match the safe recipe's envelope. Attention remains available beyond this
envelope through ordinary DCP. The model length limit remains 1,048,576.

## Build and validation

Run on the head Spark, with the other three SSH host names:

```bash
bash sparkrun-glm53-exl3/scripts/build-r22-dflash2-image.sh host2 host3 host4
```

The Python overlay is a late image layer, preserving the existing wheel-build
cache when available. The original R22/R7 composed-tree identity describes the
compiled base; `local-inference.performance-overlay=glm53-r22-performance-v1`
and the overlay's input/output SHA256 manifest describe the subsequent Python
changes. Both the retained source and installed vLLM package are patched and
verified. Unknown input or output hashes stop the build before patch writes.

The build script verifies image IDs and runs both smoke programs on each
node. The new GPU smoke uses random BF16 inputs and the native packed-FP8
cache writer. It compares gathered-cache attention against an ordinary global
cache and four simulated DCP shards with an LSE-weighted merge. It checks
unequal shard lengths, reversed physical page tables, and causal query
lengths. This test needs no model weights or four-node rendezvous.

The local CPU suite applies the overlay to the exact v9 source and checks
hashes, rejection without partial writes, idempotence, eligibility, caller
dispatch, query-head order, all four reduction slices, and the NCCL fallback.

```bash
# BASELINE_PACKAGE is a v9-composed vllm package directory, before this overlay.
GLM53_R22_PERF_BASELINE="$BASELINE_PACKAGE" \
  python3 sparkrun-glm53-exl3/tests/test_r22_performance.py
python3 sparkrun-glm53-exl3/tests/test_r22_dflash2.py
```

CPU checks passed during implementation. The ARM64 image build, GPU smoke,
four-node communication, throughput and model-level quality checks require
the Sparks and were not run during local implementation. The original
400 versus 700+ prefill figures are user observations, not new measurements.

## Four-node communication check

After distributing the image and while the model service is stopped, run the
following on each Spark. Set `HEAD_ADDR` to a head-node address reachable from
all four hosts and `NODE_RANK` to a distinct value from 0 through 3. Launch all
four commands together. This tests the new DCP communicator with real RDMA,
the middle-dimension gather, the small-output reduction and larger NCCL
fallback, including graph replay after changing the input tensors.

```bash
docker run --rm --gpus all --network host --ipc host \
  --device /dev/infiniband:/dev/infiniband --ulimit memlock=-1:-1 \
  -e LD_PRELOAD=/opt/sparkring/nccl/libnccl.so.2 \
  -e VLLM_NCCL_SO_PATH=/opt/sparkring/nccl/libnccl.so.2 \
  -e NCCL_NET=IB -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 \
  -e B12X_ROCE_HCA=rocep1s0f0,roceP2p1s0f0 -e B12X_ROCE_GID_INDEX=3 \
  --entrypoint torchrun spark-vllm-glm53-exl3:r22-dflash2-sm121-v10 \
  --nnodes=4 --nproc-per-node=1 --node-rank="$NODE_RANK" \
  --master-addr="$HEAD_ADDR" --master-port=29553 \
  /opt/compose/smoke_r22_performance.py --distributed
```

## Benchmark and isolate

Keep model snapshot, host state, prompt tokens, output length, request
concurrency and cache warmth identical. Use uncached prompts for prefill;
record prompt processing separately from compilation and model loading.
Compare decode at concurrency 1 and 8, including MTP acceptance. Verify
finite outputs and model-level long-context behavior before interpreting a
throughput improvement.

Change one recipe environment value at a time for diagnosis:

| Change | Effect |
| --- | --- |
| `VLLM_B12X_MLA_CKV_GATHER: "0"` | Disable gathered-cache prefill. |
| `VLLM_ROCE_DCP_RS_MAX_BYTES: "0"` | Keep DCP RoCEnante gathers, use NCCL reduce-scatter. |
| `VLLM_ROCE_DCP_ENABLE: "0"` | Keep existing TP RoCEnante, use NCCL for DCP. |
| `VLLM_ENABLE_ROCE_ALLREDUCE: "0"` | Disable RoCEnante for both groups. |

All switches require a restart. Preserve the v9 image tag for a full rollback.
No throughput increase is claimed until the resulting image is benchmarked.

## Source basis

- [Pinned R22 B12X attention](https://github.com/local-inference-lab/vllm/blob/70b3c1c7f1c76fcf0847fcbb4a0b8b5583b78d19/vllm/v1/attention/backends/mla/b12x_mla_sparse.py)
- [Pinned NVIDIA GLM-DSA attention](https://github.com/local-inference-lab/vllm/blob/70b3c1c7f1c76fcf0847fcbb4a0b8b5583b78d19/vllm/models/deepseek_v32/attention.py)
- [Pinned CUDA communicator](https://github.com/local-inference-lab/vllm/blob/70b3c1c7f1c76fcf0847fcbb4a0b8b5583b78d19/vllm/distributed/device_communicators/cuda_communicator.py)
- [Pinned RoCEnante runtime](https://github.com/local-inference-lab/b12x/blob/1a7e3ec286b0ff0b7c2aabee22dce08daab7e011/b12x/comm/roce/roce_oneshot.py)
- [Restored R7 EXL3 implementation](https://github.com/local-inference-lab/vllm/blob/3c0a496caf9339f396b0be8da6910b1887920709/vllm/model_executor/layers/quantization/exl3.py)
