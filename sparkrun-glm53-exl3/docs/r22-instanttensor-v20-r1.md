# v20 InstantTensor revision 1

Experimental image: `spark-vllm-glm53-exl3:r22-dflash2-sm121-v20-instanttensor-r1`.
Recipe: `recipes/glm53-exl3-v20-instanttensor-r1-4x.yaml`.
This is a loading-policy experiment on the complete v20 image. All compute,
communication and MTP settings are retained. It is not yet a confirmed
segfault fix or a measured inference improvement.

## Evidence and changes

Inspection of the installed safe and v20 images found InstantTensor 0.1.9
with identical `_impl.py` bytes. This does **not** establish identical native
binaries or ABI compatibility: safe has Torch 2.12.0+cu132; v20 has
2.13.0+cu130. Safe forces buffered I/O, a 512 MiB tensor buffer and a 5%
free-memory budget, and patches the loader's process group to `None`.
The inspected v20 image supplies none of those environment defaults.

R22 already supports explicit `instanttensor_copy` and
`instanttensor_distributed` loader options. Revision 1 therefore needs no
replacement wheel, native rebuild, or vLLM source patch:

- Use `--load-format instanttensor` and
  `--model-loader-extra-config '{"instanttensor_copy":true,"instanttensor_distributed":false}'`.
  This avoids handing the native loader a distributed NCCL communicator and
  keeps tensor storage owned across staging-buffer reuse. The latter is a
  conservative deviation from safe's `INSTANTTENSOR_COPY=0`; R22 reads the
  explicit option rather than that old environment variable.
- Set `INSTANTTENSOR_BACKEND=BUFFERED`, buffer size `536870912`, and
  `INSTANTTENSOR_MAX_FREE_MEM_USAGE=0.05`, matching safe's controls.
  BUFFERED selects buffered backend candidates; it does not promise mmap.
- Bound disk staging with 8 MiB chunks, one I/O worker and depth 8.
  These are an additional conservative experiment, not settings established
  by the safe baseline. Buffer size is not a cap on total process memory.
- Enable Python fault context with `PYTHONFAULTHANDLER=1`. A native backtrace
  is still needed to diagnose a recurring segmentation fault.

R22's existing iterator filters the checkpoint index before physical I/O
and routes tensors larger than the configured buffer through CPU
safetensors. Both paths remain intact. The image build checks exact loader
source hashes so a different base fails visibly.

## Build and qualification

On the head Spark, with the existing v20 image and this checkout:

```bash
bash sparkrun-glm53-exl3/scripts/build-r22-v20-instanttensor-image.sh worker1 worker2 worker3
```

The builder requires idle GPU capacity for its default small loader test.
`GLM53_IT_GPU_SMOKE=0` allows building/distributing during serving, but does
not qualify GPU loading. The GPU test retains tensors through repeated
ring reuse and loader close, checks values/dtypes, exercises oversized CPU
fallback, and checks index-restricted loading. It is not a full-model test.

After freeing the four serving GPUs, run the builder with its default GPU
gate, validate the recipe, and launch using the usual cache-flusher wrapper:

```bash
sparkrun recipe validate sparkrun-glm53-exl3/recipes/glm53-exl3-v20-instanttensor-r1-4x.yaml
bash sparkrun-glm53-exl3/scripts/run-with-cache-flusher.sh \
  sparkrun-glm53-exl3/recipes/glm53-exl3-v20-instanttensor-r1-4x.yaml \
  --hosts host1,host2,host3,host4 --dry-run
bash sparkrun-glm53-exl3/scripts/run-with-cache-flusher.sh \
  sparkrun-glm53-exl3/recipes/glm53-exl3-v20-instanttensor-r1-4x.yaml \
  --hosts host1,host2,host3,host4 --no-follow
```

Require all ranks to finish loading, graph capture and readiness, followed
by deterministic output checks against v20 and repeated requests without
corruption or crashes. Record loading duration, peak memory, final KV-cache
blocks and native libraries. If it still segfaults, capture the native
stack and isolate buffered backend/binary compatibility before changing
inference kernels. Do not interpret the earlier CUDA allocation failure as
the original segmentation fault.

For performance, compare v20/safetensors against this revision first with
MTP disabled **in both launch commands** (the supplied recipes default to
MTP3). Repeat the uncached 8k/64k/128k prefill and context-0 decode C1/C2/C4/C8
workloads in `TARGET-ONLY-PERFORMANCE.md`. Use identical tokens, output
lengths, cache contents, graph settings, KV capacity and machine state;
warm up and repeat runs in alternating order. Then repeat with MTP3 and
record acceptance as well as throughput. Keep loading time separate from
TTFT and steady-state decode. A surviving gain could motivate examining
weight placement/layout and memory pressure; loading success alone does
not establish an inference benefit. Roll back with the unchanged original
v20 image and recipe.

## Validation on 2026-09-09

- Built on head `10.3.10.1`, image ID
  `sha256:c526ee07fe082bb81f9f2c525c4da39cccd87d9ffd2411056a53fe08122738ff`.
  Parent v20 remains
  `sha256:26c29fa42233c38f8e2cc9a3a580ae97b230a0ecb51b4c59c577db9a63db86f6`.
- Exact installed source gate passed. Python syntax and builder shell syntax
  passed. Parsed YAML comparison passed: only image identity, revision
  metadata and the specified loader policy differ from v20.
- A preliminary isolated GPU test failed at `torch.cuda.mem_get_info()` with
  CUDA out-of-memory before entering the loader while v20 was serving.
  The existing serving container was left running.
- Worker distribution, GPU loader smoke, full-model loading, SparkRun launch
  validation and performance measurements are pending an idle test window.
