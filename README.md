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
MTP remains disabled until ordinary decode has passed the fleet smoke tests.

Run the fleet launcher from a Linux machine with passwordless SSH access to all
four management addresses (normally Spark 1):

```bash
# Optional SSH settings belong in the ignored .env file.
cp .env.example .env

# Pull the image and download the pinned model onto every node.
bash scripts/launch-glm53-flash.sh prepare

# Validate Docker, SM121, RoCE, image, model, and revision on every node.
bash scripts/launch-glm53-flash.sh preflight

# Tear down stale ranks, launch workers 3 -> 2 -> 1, then head rank 0.
bash scripts/launch-glm53-flash.sh start
```

Operational commands are `status`, `verify`, `logs [0-3]`, and `stop`. A successful
start waits for the health endpoint, verifies finite token logprobs and tool
calling, checks every container for OOM/restarts/fatal logs, then prints the
OpenAI-compatible API URL.
The model is downloaded independently to `MODEL_HOST_PATH` on each Spark; a
shared mount can be used by changing that one recipe value.

The default path builds the pinned SM121 image on Spark 1 and streams that exact
image to the other ranks. To use an already published image instead, override
`IMAGE`, `BUILD_IMAGE=0`, and `PULL_IMAGE=1` in `.env`. To build manually on an
ARM64 Spark:

```bash
docker build -f docker/Dockerfile.glm53-sm121 \
  -t spark-vllm-glm53:sm121-v1 docker
```

See [`docs/sm121-compatibility.md`](docs/sm121-compatibility.md) for the patch
inventory, upstream status, and the gates for enabling FP8 KV and MTP.
