# SparkRing-compatible GLM-5.2 serving on a switched fabric

## Status

This is an **implementation-ready, unqualified switch port** of SparkRing's
R7 serving runtime. The runtime and performance topology remain pinned, while
the checkpoint is deliberately operator-managed so compatible EXL variants
can be swapped without editing the launcher. The image build and four-node GPU
run still have to be executed on the Spark fleet.

The reference is SparkRing commit
`510556275ed3b77fc56a14367d319417072eeb8c`. Pinning is important because the
upstream repository changes rapidly.

## What remains identical

The switched recipe preserves the execution contract from SparkRing's GLM-5.2
quickstart while deliberately relaxing checkpoint identity:

| Setting | Switched recipe |
|---|---|
| Default checkpoint | Local `davidsyoung/GLM-5.3-EXL3-TR3-3.42bpw` Hugging Face cache on every node |
| Model ownership | Operator-provisioned and offline; config/index and referenced shard presence are checked, while hashes/counts are optional |
| Parallelism | TP4, DCP4, `ag_rs`, KV interleave 1 |
| Speculation | Fixed MTP4, draft TP4, B12X target and draft MoE |
| Context and concurrency | 1,048,576 tokens, 16 sequences, 4,096 batched tokens |
| KV cache | `nvfp4_ds_mla`, dynamic per-token scale, FP8 RoPE, 9,250,000,000 bytes/rank |
| KV blocks | 64 tokens |
| Prefill | EXL3 capacity 4,096 and bounded full-CKV gather to 1,048,576 tokens |
| Execution | `FULL_AND_PIECEWISE`, capture sizes Q1 through Q40 |
| Q40 | Target-only exact-Q40 state, capacity 40, route block 8, runtime attestation |
| Loading | InstantTensor plus persistent online target K6 conversion |

The builder uses SparkRing's own pinned R7 build rather than attempting to
approximate its runtime with this repository's older local-inference r34
Dockerfile. This is necessary: the older tree does not contain the complete
dynamic-NVFP4, full-CKV-gather, shared-capture, and exact-Q40 contracts.

## The deliberate topology difference

SparkRing's qualified profile uses SIRCL for supported tensor-parallel paths
and patched NCCL fallback over a physical direct cable cycle. This port:

- does not enable or mount the SIRCL TP4 backend;
- uses patched NCCL over a non-blocking RoCE switch for all distributed paths;
- uses CX0 for rendezvous and discovers both switch-facing RDMA HCAs per node
  from its CX0/CX1 inventory addresses;
- enables cross-NIC routing and subnet-aware routing;
- does not force `NCCL_ALGO=Ring`, fixed channel counts, or
  `NCCL_SKIP_TREE_CONNECT`, allowing NCCL to choose switch-appropriate
  algorithms.

The two rails must both terminate on the intended lossless switch fabric, use
MTU 9000, and have working RoCEv2/PFC/ECN configuration. DGX Spark HCA names
are not assumed to be `mlx5_*`: the launcher maps each inventory IP to its
Linux netdev and then uses `ibdev2netdev` or sysfs to find its RDMA device. If
only one adapter is cabled, use a single-rail topology instead of this profile.

The switch may improve collective performance, but that is a hypothesis, not
a property of the topology. Preserve the exact serving knobs and compare the
normalized 2K through 128K prefill and C1/C2/C4/C8 decode matrix against the
published SparkRing numbers.

## Build and launch

The default `prepare` action is completely offline. It does not build or pull
an image and does not contact Hugging Face. It checks that the existing runtime
image and local model are present on every node. The configured path is a
Hugging Face repository-cache root; the node launcher follows local `refs/main`
and mounts the whole repository so snapshot-to-blob symlinks remain valid.

```bash
cp .env.example .env.sparkring-switch
# Set SPARK_SSH_USER in that file.
# Normally leave NCCL_IB_HCA empty for automatic per-node discovery.

# Offline local checks only. Nothing is downloaded.
bash scripts/launch-glm52-sparkring-switch.sh prepare
bash scripts/launch-glm52-sparkring-switch.sh preflight
bash scripts/launch-glm52-sparkring-switch.sh start
bash scripts/launch-glm52-sparkring-switch.sh benchmark
```

`start` does not run preflight implicitly. Run the explicit `preflight` action
when configuration or hardware state has changed; repeated starts go directly
to teardown and worker-first launch. The default GPU utilization remains 0.85;
each node must expose at least about 104 GiB free before launch. A lower reading
is a node-health or competing-workload condition and is not masked by reducing
the serving envelope.

If the runtime image is not present, explicitly run `build` first. That action
checks out the pinned SparkRing source, pulls its immutable parent, builds the
runtime and Q40 overlays on rank 0, and distributes them. Audit and set
`SPARKRING_BASE_IMAGE_LICENSES` only if the immutable parent's own
`org.opencontainers.image.licenses` label needs an audited override. The pinned
parent currently has no such label, so the private-fleet default is the honest
SPDX placeholder `LicenseRef-Unknown-Operator-Supplied`. It is not a license
grant; do not redistribute the derived image until the parent is audited.

The repository's `benchmark` action is a functional deployment diagnostic. It
is not the `llm_decode_bench.py` normalized harness used for SparkRing's
published table, which is referenced by hash in upstream receipts but is not
distributed in the upstream repository. Its JSON must not be compared
directly with that table. A performance claim requires obtaining and
hash-verifying that external harness, then reproducing its cold, unique-prompt
2K/8K/16K/32K/64K/128K and C1/C2/C4/C8 matrix on both topologies.

The custom build path runs on rank 0. It checks out the pinned SparkRing
revision, uses SparkRing's receipt-gated R7 image builder, generates the two
exact-Q40 overlays for the produced immutable image ID, and then distributes
the image and overlays to ranks 1-3. Preflight checks the local model index and
all referenced shards but does not require a repository identity, revision
marker, fixed shard count, or content hashes unless the operator supplies
those optional values.

The wrapper deliberately uses `.env.sparkring-switch`, not the shared `.env`,
so paths from another deployment do not accidentally override this target.
Set `LOCAL_ENV_FILE` explicitly if another ignored local file is preferred.

Exact-Q40 receipts are create-once upstream. On restart, the node launcher
moves an existing receipt to a timestamped backup before starting, preserving
the evidence while giving the new process a fresh namespace.

## Changing the EXL model

The user owns model provisioning. Set `MODEL_HOST_PATH` and
`MODEL_MOUNT_HOST_PATH` to either a ready model directory or its Hugging Face
repository-cache root, then update `MODEL_ID` and `SERVED_MODEL_NAME` for
reporting. No download is attempted. Optional config/index hashes, shard count,
and indexed byte size can be supplied to tighten validation.

Lenient selection does not imply universal compatibility. Qualify a materially
different model for:

1. architecture and TP4 checkpoint slicing;
2. tokenizer, reasoning parser, tool parser, and chat template;
3. DCP4 `ag_rs`, global TopK, draft sharding, and full-CKV gather;
4. EXL3 mixed-trellis target layers and online-quantization ignore rules;
5. MTP layer count, draft quantization, and the fixed speculative depth;
6. Q1-Q40 graph shapes and exact-Q40 target/draft state assumptions;
7. dynamic NVFP4 KV record size, cache capacity, and long-context correctness.

The exact-Q40 overlay remains image-bound and its runtime gates are enforced.
If a model's layer geometry does not satisfy those gates, disable Q40 for an
initial bring-up or rebuild/adjust the overlay instead of weakening its runtime
attestation. Upstream requires the receipt's checkpoint field to be 40
lowercase hexadecimal characters. The local-model profile therefore uses a
stable synthetic `Q40_CHECKPOINT_REVISION`; it identifies the operator-managed
model slot and is not presented as a model revision or content hash.

## Qualification gates

- image labels identify the pinned SparkRing revision and its runtime verifier
  passes on all four ranks with the image's CUDA forward-compatibility driver
  library preloaded;
- config/index readability and every referenced local shard pass on all ranks;
- the GPU memory health probe reports at least the 0.85 envelope (about 104 GiB)
  free on every rank, with no competing GPU workload;
- both switch rails pass RoCE validation without retries, fallback to sockets,
  or asymmetric HCA selection;
- all four exact-Q40 attestations are fresh and consistent;
- `/health` and `/v1/models` report the expected model and 1M context;
- finite decode, tool calling, reasoning output, prefix caching, and long
  prefill pass with no OOM, NCCL, CUDA, or repeated-token failures;
- graph capture includes Q1-Q40 and post-workload rank/transport health passes;
- the normalized benchmark is run with cold unique prompts and compared with
  SparkRing's published baseline before any speed claim is made.
