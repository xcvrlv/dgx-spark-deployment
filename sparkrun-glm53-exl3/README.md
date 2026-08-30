# GLM-5.3 EXL3 SparkRun bring-up

This is a small, independent SparkRun recipe for the already-built
`spark-vllm-glm52-exl3:sparkring-switch-v1` image and the existing local model
cache. It does not build or pull an image, download a model, create a fixed
large KV cache, or enable MTP. It captures only the batch-size-one uniform
decode CUDA graph; prefill and mixed batches stay eager.

Run the commands from a Spark that can SSH to all four nodes:

```bash
# Install the CLI only if it is not already installed. This does not alter the
# current cluster configuration.
uvx sparkrun setup install

# One-time host and network validation; this is separate from launching the
# model and does not alter the existing switched fabric.
uvx sparkrun setup check

# Confirm SparkRun sees the cache root and render the exact launch first.
sparkrun recipe validate sparkrun-glm53-exl3/recipes/glm53-exl3-4x-safe.yaml
sparkrun show sparkrun-glm53-exl3/recipes/glm53-exl3-4x-safe.yaml
sparkrun run sparkrun-glm53-exl3/recipes/glm53-exl3-4x-safe.yaml \
  --hosts host1,host2,host3,host4 --dry-run

# The first host is the head. This starts in the background; Ctrl-C only
# detaches from log following.
sparkrun run sparkrun-glm53-exl3/recipes/glm53-exl3-4x-safe.yaml \
  --hosts host1,host2,host3,host4
```

Replace `host1` through `host4` with the management hostnames or IPs already
used by the cluster. SparkRun manages the per-node native-vLLM arguments,
container/image availability, RDMA discovery, lifecycle, and logs. The model
path is deliberately the Hugging Face cache root: the command reads `refs/main`
and the complete root is mounted, so snapshot-to-blob symlinks work without
another copy of the weights.

The initialized worker reported 101.76 GiB free out of 121.63 GiB. A 0.85
target requests 103.38 GiB and therefore fails vLLM's admission check before
model loading. The 0.83 target requests about 100.95 GiB, leaving roughly
0.81 GiB of startup margin while adding about 3.65 GiB to the proven 0.80
allocation. `gpu_memory_utilization` is an allocation budget, not a complete
host-RAM limit: model loading, graph capture,
communication buffers, Python processes, and page cache can consume additional
memory. Raising it does not itself make MTP fit; it primarily gives vLLM more
KV-cache budget, while MTP weights, activations, and graph memory are profiled
before the remaining KV capacity is chosen. Unlike the previous 0.85 recipe,
this one does not reserve a fixed
16.5 GB KV slab or capture Q1–Q40. `FULL_DECODE_ONLY` with capture size `[1]`
covers the current single-request ordinary decode path while leaving prefill
and mixed batches eager. CUDA-graph memory estimation is enabled so KV sizing
accounts for graph headroom.

The recipe intentionally does not set `executor_config.memory_limit`, so
SparkRun does not add Docker's `--memory` cgroup cap. That cap is not a useful
proxy for the dynamically shared GB10 CPU/GPU memory pool and can turn a
recoverable allocation failure into a prematurely killed container. SparkRun's
host setup and lifecycle controls remain responsible for node protection.

### Clean up a failed launch before retrying

SparkRun's native executor is an otherwise-idle `sleep infinity` container;
vLLM itself runs through `docker exec`. If that vLLM command fails after model
loading, the executor can remain alive with its cgroup still charged for model
file cache and shared memory. Stop the SparkRun job before retrying so Docker
removes that executor:

```bash
sparkrun stop glm53-exl3-4x-safe --hosts host1,host2,host3,host4
```

If the job is already failed and the exact executor is still present, first
confirm it has no live vLLM process (`docker top <container>`), then remove
only that named stale container with `docker rm -f <container>`. Do this on
each affected node. Do not use a blanket Docker prune on the DGX Spark.

Keep MTP disabled until this decode-graph profile completes a cold start and
sustained requests. Confirm the logs report `FULL_DECODE_ONLY` and capture size
1, then record KV capacity and generation throughput before adding MTP. Capture
`sparkrun logs glm53-exl3-4x-safe`, `sparkrun status`, host `dmesg`, and
`memory.events` on any failed node. Do not restore the old 1M context, 16.5 GB
fixed KV cache, and Q1–Q40 graph capture.

For optional name-based registration, initialize and commit this folder as its
own Git repository, then run `sparkrun registry add <its-git-url>`. The local
file path above is the recommended first launch because it needs no registry
installation.

If no SparkRun cluster has been configured yet, run `sparkrun setup wizard`
interactively before the launch. It can set up the SSH mesh, RDMA detection,
and early-OOM protection, so review its detected networking and sudo changes
instead of accepting them blindly on this existing switched fabric.
