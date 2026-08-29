# DGX Spark deployment

This repository is the development and test workspace for deploying vLLM on
DGX Spark systems. It is organized around three related activities:

1. **Image compatibility:** patch and maintain vLLM container images so they
   run reliably and efficiently on NVIDIA SM121 / GB10 hardware.
2. **Deployment testing:** hold deployment configurations and run controlled
   A/B tests across image, runtime, and serving changes.
3. **Optimization:** measure and improve performance, stability, and resource
   utilization of the deployed stack.

## Development target

The primary target is **TP4**: a four-node DGX Spark system. Configurations
and test results should therefore be reproducible across all four nodes and
should clearly record the image, model, runtime settings, and node topology
used for each test.

## Node network inventory

The initial four-node inventory is in [`sparks.env`](sparks.env). It defines:

- management addresses in the `10.3.10.0/24` network;
- ConnectX (CX) interface 0 addresses in `192.168.0.0/24`;
- ConnectX (CX) interface 1 addresses in `192.168.1.0/24`.

Load it in a shell or deployment tool as appropriate for that tool. The file
contains topology information, not credentials. Keep passwords, tokens, and
other secrets in a local `.env` file or secret manager; local `.env` files are
ignored by Git.

## Working principles

- Keep deployment changes versioned and reviewable.
- Record benchmark methodology and results alongside the configuration they
  evaluate.
- Treat image patches and runtime workarounds as temporary until validated on
  the TP4 topology.
- Prefer repeatable scripts and pinned versions for A/B comparisons.

## Current roadmap

1. Get `local-inference-lab/GLM-5.3-Flash-NVFP4` working.
2. Prepare for GLM 5.3 by getting an EXL3 version of GLM 5 working. Use
   `brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78` for testing before GLM 5.3
   is available.

## Preferred base and upstream guidance

- Always try to use [local-inference-lab](https://github.com/local-inference-lab)
  work as the base, specifically `vLLM` and `b12x`.
- GLM 5.3 Flash may require waiting for MTP fixes from `local-inference-lab`,
  but the model should be brought up and running without MTP first.
- The organization’s other repositories may also provide valuable solutions
  and implementation guidance.

## GLM-5.3 Flash TP4 first iteration

The first runnable deployment is in
[`recipes/glm-5.3-flash-nvfp4.env`](recipes/glm-5.3-flash-nvfp4.env). It pins
the local-inference-lab checkpoint and starts one vLLM rank on each Spark using
native multi-node tensor parallelism. The default profile exposes the full
1,048,576-token model context, uses an explicit 8 GiB FP8 KV slab per rank,
FlashInfer B12X/CUTLASS kernels, FlashKDA, chunked prefill, and prefix caching.
The base recipe remains an eager, non-MTP control. CUDA graphs and MTP are
enabled through staged profiles so their gains and memory costs can be measured
independently.

Run the fleet launcher from a Linux machine with passwordless SSH access to all
four management addresses (normally Spark 1):

```bash
# Optional SSH settings belong in the ignored .env file.
cp .env.example .env

# Build the pinned SM121 image on Spark 1 and distribute it to every node.
bash scripts/launch-glm53-flash.sh build

# Or build the image, then download/verify the model on every node.
bash scripts/launch-glm53-flash.sh prepare

# Validate Docker, SM121, RoCE, image, model, and revision on every node.
bash scripts/launch-glm53-flash.sh preflight

# Tear down stale ranks, launch workers 3 -> 2 -> 1, then head rank 0.
bash scripts/launch-glm53-flash.sh start

# Record the eager baseline before changing performance settings.
bash scripts/launch-glm53-flash.sh benchmark
```

Operational commands are `build`, `status`, `verify`, `benchmark`,
`logs [0-3]`, and `stop`.
Use `build` when model weights already exist at `MODEL_HOST_PATH`; unlike
`prepare`, it does not run `hf download`. A successful start waits for the health
endpoint, verifies finite token logprobs and tool calling, checks every container
for OOM/restarts/fatal logs, then prints the OpenAI-compatible API URL.
The model is downloaded independently to `MODEL_HOST_PATH` on each Spark; a
shared mount can be used by changing that one recipe value.

## CUDA graph and MTP rollout

Apply one profile at a time and use the same profile for `preflight`, `start`,
and `verify`. The graph profile requests vLLM `FULL_AND_PIECEWISE` capture:
full-model graphs for compatible decode batches, piecewise graphs for prefill
and mixed batches, and eager fallback for shapes which cannot be captured. It
also raises the chunked-prefill budget from 8,192 to 32,768 tokens.

```bash
# Stage 1: CUDA graphs, no MTP. Startup will take longer while graphs capture.
PROFILE_FILE=profiles/glm53-cudagraph.env \
  bash scripts/launch-glm53-flash.sh start
PROFILE_FILE=profiles/glm53-cudagraph.env \
  bash scripts/launch-glm53-flash.sh benchmark

# Stage 2: one MTP draft token for correctness, memory, and acceptance testing.
PROFILE_FILE=profiles/glm53-mtp-1.env \
  bash scripts/launch-glm53-flash.sh start
PROFILE_FILE=profiles/glm53-mtp-1.env \
  bash scripts/launch-glm53-flash.sh benchmark

# Stage 3: three draft tokens after mtp-1 passes verify and the smoke test.
PROFILE_FILE=profiles/glm53-mtp-3.env \
  bash scripts/launch-glm53-flash.sh start
PROFILE_FILE=profiles/glm53-mtp-3.env \
  bash scripts/launch-glm53-flash.sh benchmark
```

`verify` rejects the graph profiles unless rank 0 logs show that CUDA graph
capture ran. The MTP profiles use TP4 draft execution, greedy draft sampling,
and local argmax reduction. They use Marlin for both the target model's NVFP4
experts and the checkpoint's MXFP8 MTP experts. The image patches vLLM's
mixed-precision resolver so the renamed draft layer retains its MXFP8 method.
The pinned engine supports async scheduling for its EAGLE/MTP path, so it
remains enabled. Benchmark JSON is written under `benchmarks/results/`; it
reports an uncached long-prefill approximation plus generic and predictable
structured decode rates. Compare medians, not the first request after startup.

When the vLLM `/metrics` endpoint exports speculative-decoding counters, the
result also records the drafted-token acceptance rate for tuning MTP depth.

For an existing Hugging Face cache snapshot, set `MODEL_HOST_PATH` to the exact
snapshot for host-side validation, set `MODEL_MOUNT_HOST_PATH` to the parent
`models--ORG--NAME` directory so its `blobs` symlinks remain valid, set
`MODEL_MOUNT_CONTAINER_PATH` to `/model-mount`, and set `MODEL_CONTAINER_PATH`
to `/model-mount/snapshots/REVISION`.

The default path builds the pinned SM121 image on Spark 1 and streams that exact
image to the other ranks. To use an already published image instead, override
`IMAGE`, `BUILD_IMAGE=0`, and `PULL_IMAGE=1` in `.env`. To build manually on an
ARM64 Spark:

```bash
docker build -f docker/Dockerfile.glm53-sm121 \
  -t spark-vllm-glm53:sm121-v5 docker
```

See [`docs/sm121-compatibility.md`](docs/sm121-compatibility.md) for the patch
inventory, upstream status, and the gates for enabling FP8 KV and MTP.

## Full GLM-5.2 EXL3 TP4 bring-up

The full-model path is now staged separately from GLM-5.3 Flash. It ports the
current local-inference-lab r34 GLM-5.2 R7 source composition to ARM64/SM121,
serves the pinned 3.5 bpw checkpoint across four single-GPU Sparks, and uses the
same fleet lifecycle as the Flash recipe. The upstream images are amd64/SM120
single-host artifacts, so this repository rebuilds their pinned vLLM, B12X,
ExLlamaV3, and InstantTensor sources on a Spark and uses NCCL/RoCE instead of
B12X's single-host PCIe collectives.

```bash
# Build the ARM64 image, distribute it, and download the TP4 checkpoint.
bash scripts/launch-glm52-exl3.sh prepare

# Validate the eager, target-only baseline first.
bash scripts/launch-glm52-exl3.sh start
bash scripts/launch-glm52-exl3.sh benchmark

# Then stage graphs, MTP1, and finally the upstream-proven MTP3 depth.
PROFILE_FILE=profiles/glm52-exl3-cudagraph.env \
  bash scripts/launch-glm52-exl3.sh start
PROFILE_FILE=profiles/glm52-exl3-mtp-1.env \
  bash scripts/launch-glm52-exl3.sh start
PROFILE_FILE=profiles/glm52-exl3-mtp-3.env \
  bash scripts/launch-glm52-exl3.sh start
```

The first start performs persistent online K6 encoding and can take well over
15 minutes; keep `/var/tmp/glm52-exl3-cache` on every rank. GLM-5.3 full is not
treated as a checkpoint swap: its release and EXL3 tensor contract must be
verified before changing this recipe. See
[`docs/glm52-exl3-sm121.md`](docs/glm52-exl3-sm121.md) for provenance,
limitations, and the promotion gates.

## SparkRing-compatible switched GLM serving target

The target topology now has a separate launch path which preserves
SparkRing's GLM-5.2 serving contract while replacing its direct-cycle
SIRCL transport with NCCL over the switched, dual-rail RoCE fabric. It uses
TP4/DCP4 `ag_rs`, fixed MTP4, 1M context, 16 sequences, dynamic NVFP4 DS-MLA,
a 9.25 GB/rank KV slab, full-CKV gather, Q1-Q40 graphs, and image-bound
exact-Q40 overlays.

```bash
# Create .env.sparkring-switch from .env.example and set its SSH user. The
# default model is already expected in juho's Hugging Face cache on every node.
# `prepare` is offline: it neither builds/pulls an image nor downloads a model.
bash scripts/launch-glm52-sparkring-switch.sh prepare
bash scripts/launch-glm52-sparkring-switch.sh preflight
bash scripts/launch-glm52-sparkring-switch.sh start
```

Run `bash scripts/launch-glm52-sparkring-switch.sh build` separately if the
pinned runtime image and Q40 overlay are not already installed. That explicit
action requires network access. It records the immutable parent's OCI license
label automatically; `SPARKRING_BASE_IMAGE_LICENSES` is an optional audited
override. Because the pinned flattened parent has no such label, this private
fleet profile records `LicenseRef-Unknown-Operator-Supplied`; do not
redistribute the derived image without a parent-license audit. Model selection
is operator-managed and lenient: point `MODEL_HOST_PATH`
at another ready EXL model directory or Hugging Face repository-cache root.
Optional hashes and shard-count fields can be set when strict model pinning is
wanted. This path is implemented but not fleet-qualified. See
[`docs/glm52-sparkring-switch.md`](docs/glm52-sparkring-switch.md).
