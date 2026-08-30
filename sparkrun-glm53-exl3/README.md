# GLM-5.3 EXL3 SparkRun bring-up

This is a small, independent SparkRun recipe for the already-built
`spark-vllm-glm52-exl3:sparkring-switch-v1` image and the existing local model
cache. It does not build or pull an image, download a model, create a fixed
oversized KV cache, or capture prefill graphs. It enables the
checkpoint-qualified MTP3 path for up to eight sequences; prefill and mixed
batches stay eager.

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

Before page-cache reclamation, the initialized worker reported only 101.49 GiB
free out of 121.63 GiB. Immediately clearing the cluster page cache raised the
same startup reading to about 112.9 GiB. The current 0.89 startup target is
108.25 GiB, leaving about 4.65 GiB of admission margin after a clean start.
`gpu_memory_utilization` is an allocation budget, not a complete host-RAM
limit: model loading, graph capture,
communication buffers, Python processes, and page cache can consume additional
memory. This recipe fixes the per-node KV pool at 14,000,000,000 bytes. That is
rounded down from vLLM's measured 14,191,165,010-byte recommendation for
fitting inside the 0.89 envelope. The alternative 19,186,056,704-byte message
means consuming essentially all memory free in that one startup snapshot; even
19,000,000,000 bytes would leave only about 186 MB relative to it and is not a
safe GB10 UMA margin. MTP3 verifies four tokens per sequence, so
`FULL_DECODE_ONLY` captures `[4,8,12,16,20,24,28,32]` for one through eight
uniform speculative-decode requests while leaving prefill and mixed batches
eager. CUDA-graph memory estimation is enabled so KV sizing accounts for graph
headroom during profiling; the explicit KV pool prevents a conservative graph
estimate from silently shrinking the cache. The 32 GiB `/dev/shm` setting is
capacity, not preallocated memory or a container memory limit.

The 4096-token prefill batch is paired with
`VLLM_EXL3_PREFILL_CAPACITY=4096`; increasing only the scheduler limit would
leave the EXL3 scratch path sized for the earlier 1024-token baseline. The
recipe also pins the proven SM121/B12X controls: V2 model runner, B12X sparse
indexer, MTP verification through the sparse decode path, CKV gather disabled,
DCP global top-k with a sharded draft, a 256 MiB sparse-indexer logits bound,
four NCCL channels, and MTP acceptance instrumentation. Async scheduling stays
enabled because this exact V2 MTP3 path has already launched successfully and
decode-aware prefill is not enabled.

The GLM-5.2 `index_topk_pattern` override is intentionally absent. This GLM-5.3
checkpoint already carries the complete 78-entry `indexer_types` topology with
`index_topk_freq=4` and `index_skip_topk_offset=3`; replacing it with a copied
GLM-5.2 string would be a checkpoint-specific override. Adaptive MTP depths,
long-context permission overrides, decode-aware prefill, and its associated
scheduler budgets also remain absent.

Do not copy the v1 `mods/drop-caches` field into this native v2 recipe: the
presence of `mods` selects SparkRun's legacy recipe path. To test whether page
cache is what prevents 0.85 admission, use the supported cluster operation
immediately before one controlled launch:

```bash
# One-time setup if SparkRun has not saved the required sudo permission:
sparkrun setup clear-cache --cluster <cluster-name> --save-sudo

# Controlled A/B launch:
sparkrun setup clear-cache --cluster <cluster-name>
```

Then compare vLLM's `Free memory on device (.../121.63 GiB) on startup` line.
The 0.89 gate needs about 108.25 GiB free. The measured increase from 101.49 to
about 112.9 GiB confirms that reclaimable cache—not the sparse-indexer
workspace—caused the earlier ceiling. Repeat the cache clear before cold
launches; the fixed 14 GB KV pool assumes that clean-start margin.

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

Confirm the logs report MTP with three speculative tokens, `FULL_DECODE_ONLY`,
capture sizes through 32, and a 14,000,000,000-byte KV pool. Then record KV
capacity, drafted-token acceptance,
and generation throughput at concurrency 1 and 8. Capture `sparkrun logs
glm53-exl3-4x-safe`, `sparkrun status`, host `dmesg`, and `memory.events` on any
failed node. Do not restore the old 1M context, 16.5 GB KV cache, and Q1–Q40
graph capture.

For optional name-based registration, initialize and commit this folder as its
own Git repository, then run `sparkrun registry add <its-git-url>`. The local
file path above is the recommended first launch because it needs no registry
installation.

If no SparkRun cluster has been configured yet, run `sparkrun setup wizard`
interactively before the launch. It can set up the SSH mesh, RDMA detection,
and early-OOM protection, so review its detected networking and sudo changes
instead of accepting them blindly on this existing switched fabric.
