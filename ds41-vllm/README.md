# DeepSeek V4.1 Flash — JJ + b12x, four Sparks, c8

Native ARM64 source build of local-inference-lab vLLM `dev/jovian-judgement`,
with b12x, SM121 kernels and dual-rail RoCEnante. Independent of the preserved
GLM images. `c8` means **eight concurrent sequences**.

Defaults: TP4, one GPU per node, DCP1, **1,048,576-token native context cap**,
4096-token prefill chunks, 0.80 GPU memory utilization, FP8 KV, native MXFP4
experts and **original FP8 Engram on local SSD**. CUDA graphs and prefix caching
are enabled. Target-only is the initial profile; optional DSpark is below.
The context cap is a per-request limit, not a promise that eight simultaneous
1M-token requests fit. No GPU-memory or cgroup limit is used to force a fit;
startup must establish the available KV capacity on the fleet.

## Build and run on Spark 1

Copy this directory to Spark 1. Python 3, Git, Docker with NVIDIA Container
Toolkit, and passwordless SSH from Spark 1 to all four nodes (including itself)
are required. Stop the other inference service before running GPU checks.

```bash
cd ds41-vllm
cp cluster-c8.json fleet.local.json
# Edit fleet.local.json for your local SSD checkpoint/cache paths and SSH key.
# model_path is the HF repository cache ROOT, containing snapshots/ and blobs/.

bash build-image.sh
python3 fleet.py --config fleet.local.json share
python3 fleet.py --config fleet.local.json plan
python3 fleet.py --config fleet.local.json preflight
python3 fleet.py --config fleet.local.json start
```

`start` repeats preflight, runs the four-node fabric qualification, starts
workers 3/2/1 and then head 0, waits for health, sends eight concurrent finite
logprob requests, and requires a log confirming live RoCEnante dispatch.
Failures retain serving containers for diagnosis. Fabric test containers are
removed after their logs are collected. Existing serving containers must be
stopped explicitly before another start; other deployments are not removed.

```bash
python3 fleet.py --config fleet.local.json status
python3 fleet.py --config fleet.local.json logs --rank 0
python3 fleet.py --config fleet.local.json smoke
python3 fleet.py --config fleet.local.json stop
# Fabric-only qualification while the model service is stopped:
python3 fleet.py --config fleet.local.json fabric
```

API: `http://192.168.0.1:8000/v1`, served name `DeepSeek-V4.1-Flash`.
The endpoint is bound to all interfaces, without API authentication; use the
cluster's private network access controls.

## Checkpoint and Engram storage

Use `deepseek-ai/DeepSeek-V4.1-Flash`, revision
`fb2764a5cf321eaa5070ca8f9e892818f477c16d`, on **every node's local SSD**.
For example, run this on each Spark with the Hugging Face CLI installed:

```bash
hf download deepseek-ai/DeepSeek-V4.1-Flash \
  --revision fb2764a5cf321eaa5070ca8f9e892818f477c16d
```

The cache root is mounted read-only, including `blobs/`, so snapshot symlinks
work. The checker verifies the original config hash, all indexed tensors and
shard sizes, and the Engram FP8 datatype/256-byte row width. It rejects the old
FP4-Engram hybrid. This is a header/completeness check, not a full weight checksum.

JJ now binds Engram directly to immutable checkpoint file offsets and b12x
reads selected rows through io_uring. No conversion or separate row-file pack
step is needed. The older dsv41/SGLang row files are incompatible with this path.
`preflight` rejects network filesystems for the model root. Verify that the
underlying local filesystem actually resides on SSD/NVMe. All four nodes need
space for the complete checkpoint and a writable local compilation cache.

Docker's default seccomp profile blocks io_uring, so the recipe uses
`seccomp=unconfined` and explicitly tests `io_uring_queue_init` in the container.
It retains normal Docker capabilities and exposes `/dev/infiniband`; memlock is
unlimited. A deployment-specific seccomp profile allowing the io_uring syscalls
can replace this setting later.

## RoCEnante and SM121

The preserved GLM fabric settings carried over are the `f0` HCA pair
`rocep1s0f0,roceP2p1s0f0`, GID 3, two-rail operation, 2 MiB all-reduce and
16 MiB **per-shard** all-gather bounds, and four NCCL channels. Socket interfaces
are discovered separately on every node from its CX0 address. PCIe all-reduce
is disabled. **Do not add `--disable-custom-all-reduce`: it disables RoCEnante
too.** Larger/ineligible collectives use NCCL over IB/RoCE.

Current b12x already contains dual-rail striping, size-aware gather behavior,
and the SM121 integrated-memory transport. Current JJ exchanges its RoCEnante
setup through Gloo, avoiding an extra Torch NCCL communicator in serving.
The image compiles and loads the C verbs proxy and compiles the io_uring storage
extension during build. CUDA extension imports, storage loading and the vLLM
adapter ABI check run after build with `docker run --gpus all`; Docker build
does not have the host's `libcuda.so.1`. The four-node test constructs the actual vLLM RoCEnante adapter,
rejects initialization fallback, compares reductions/gathers with NCCL,
checks traffic counters on both rails, and checks mixed CUDA-graph replay with
changing inputs. A passing import or health endpoint alone is not qualification.

JJ's source already includes SM121 CMake targets; build flags select `12.1a`
and the runtime selects `sm_121a`. The image uses the upstream NCCL build,
without depending on the deleted SparkRing image or preloading its libraries.
It lets NVIDIA Container Toolkit supply the host driver; no compat `libcuda`
preload is added. Actual driver/CUDA compatibility is checked on the GPU.

## Can index_topk be 1024?

**Not as a configuration-only change in these pins. Keep 512.** The original
checkpoint's indexer uses MXFP4; this is distinct from the FP8 Engram tables.

* [JJ attention.py](https://github.com/local-inference-lab/vllm/blob/35601be19df6be8be33f05aa482139e4b0a12bff/vllm/models/deepseek_v4_1/attention.py#L489)
  fixes indexed attention width, output buffers and decode/prefill indexer plans
  at 512. A separate model-level buffer reads `config.index_topk`, so changing
  only the config can create inconsistent widths.
* [JJ sparse_mla.py](https://github.com/local-inference-lab/vllm/blob/35601be19df6be8be33f05aa482139e4b0a12bff/vllm/models/deepseek_v4_1/sparse_mla.py#L141)
  clamps selected lengths to 512.
* [b12x indexer API](https://github.com/local-inference-lab/b12x/blob/323107ff948ca532f1f7c793b4b550c30ba5212b/b12x/attention/dsa_indexer/api.py#L111)
  explicitly rejects MXFP4 `topk != 512`; its scratch planner repeats the gate.

Some older FP8 indexer/DSV4 kernels accept 1024, which does **not** establish
support for this V4.1 MXFP4 path. Enabling it needs coordinated buffer/metadata,
attention-plan and indexer changes, plus prefill/decode/candidate-reindex and
graph tests. It also changes model behavior and needs quality/performance
measurement. No 1024 override or compute patch has been added.

## Optional DSpark

After the target-only run passes, set `draft_tokens` in `fleet.local.json` to
`3`, stop and start again. `1`, `5` and `7` are also accepted for comparison.
This uses JJ's native DSpark configuration, DCP1 and capture sizes spanning
all c8 verification depths. No GLM MTP heads or EXL3 quantization are involved.
The initial setting is 0 because the exact fleet has not been qualified.

## GLM patch candidates — inventory only

| Preserved work | Applicability here |
| --- | --- |
| v17 startup allocator reclamation and memory accounting | Potentially useful on unified-memory GB10; reassess current JJ lifecycle before porting. |
| v17 incremental prefix-cache hash copies | Potentially generic; compare current upstream code before porting. |
| v15 inline RoCE payload/balanced fanout; v16 empty-CQ skipping/lazy payload initialization | Fabric candidates for a measured A/B; local overlays are not present in this image. |
| v10–18 DCP/CKV gather, metadata reuse and GLM sparse-indexer batching | GLM-DSA-specific paths; this recipe uses V4.1 attention and DCP1. Not direct ports. |
| v14 greedy local argmax / aligned winner packets | Revisit only if profiling DSpark identifies the corresponding reduction path. |
| EXL3 K6 rotations, mixed-K output fusion, M8/M16 FC2, FC1 tail splitting, InstantTensor EXL3 loading | EXL3-specific; not applied to native MXFP4/FP8. |

No old compute overlays were implemented. The available native launcher avoids
an additional dependency; no LOL launcher was found in the remaining working
tree during this audit.

## Provenance and validation limits

Resolved 2026-09-12: [JJ `35601be`](https://github.com/local-inference-lab/vllm/commit/35601be19df6be8be33f05aa482139e4b0a12bff),
[b12x `323107f`](https://github.com/local-inference-lab/b12x/commit/323107ff948ca532f1f7c793b4b550c30ba5212b).
The build uses JJ's complete Dockerfile with its nightly Torch resolution,
matching Torch/vision/audio across stages. Source pins are fixed; base tags,
nightly packages and apt resolution are not fully immutable. Archive the
resulting image for exact reproduction. `.build/image-inspect.json` and
`.build/pip-freeze.txt` record the result; fleet distribution verifies identical
image IDs. Rebuilding later can resolve different dependencies.

The compile-stage builder is pinned separately in `versions.env` to the ARM64
`pytorch/manylinuxaarch64-builder` image. JJ's default
`pytorch/manylinux2_28-builder` tag is AMD64-only and fails on Spark with
`exec /bin/sh: exec format error`, even with `--platform linux/arm64`.
The launcher explicitly passes `BUILD_BASE_IMAGE` and checks its platform.
After updating `build-image.sh` and `versions.env`, rerun `bash build-image.sh`;
there is no need to prune caches or edit the pinned source checkout.

The Windows development host has no running Docker daemon or accessible Spark
GPU. **The image has not been compiled here, and native-context c8 serving,
RoCEnante throughput and model quality remain unmeasured.** Run the supplied
build and fleet gates on the Sparks. CPU validation:

```bash
python3 -m unittest discover -s tests -v
```
