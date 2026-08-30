# GLM-5.3 EXL3 SparkRun bring-up

This is a small, independent SparkRun recipe for the already-built
`spark-vllm-glm52-exl3:sparkring-switch-v1` image and the existing local model
cache. It does not build or pull an image, download a model, create a fixed
large KV cache, enable MTP, or capture CUDA graphs.

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

The 0.80 utilization leaves roughly 26 GiB of the 128 GiB unified-memory pool
outside vLLM's target allocation. `gpu_memory_utilization` is an allocation
budget, not a complete host-RAM limit: model loading, graph capture,
communication buffers, Python processes, and page cache can consume additional
memory. The previous recipe had only about 19 GiB outside its 0.85 target while
also reserving a 16.5 GB KV slab and capturing 40 graph sizes. Its shared-memory
broadcast waits started after that phase, which is consistent with an EngineCore
worker being starved or killed; Docker's `OOMKilled=false` does not rule out a
child process or host-level pressure event.

The recipe intentionally does not set `executor_config.memory_limit`, so
SparkRun does not add Docker's `--memory` cgroup cap. That cap is not a useful
proxy for the dynamically shared GB10 CPU/GPU memory pool and can turn a
recoverable allocation failure into a prematurely killed container. SparkRun's
host setup and lifecycle controls remain responsible for node protection.

Keep this recipe eager until it completes a cold start and sustained requests.
Capture `sparkrun logs glm53-exl3-4x-safe`, `sparkrun status`, host `dmesg`, and
`memory.events` on any failed node before changing one dimension at a time:
context/KV capacity first, then CUDA graphs, then MTP. Do not restore the old
1M context, 16.5 GB fixed KV cache, and Q1–Q40 graph capture in one step.

For optional name-based registration, initialize and commit this folder as its
own Git repository, then run `sparkrun registry add <its-git-url>`. The local
file path above is the recommended first launch because it needs no registry
installation.

If no SparkRun cluster has been configured yet, run `sparkrun setup wizard`
interactively before the launch. It can set up the SSH mesh, RDMA detection,
and early-OOM protection, so review its detected networking and sudo changes
instead of accepting them blindly on this existing switched fabric.
