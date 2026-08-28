# SparkRing-compatible GLM-5.2 serving on a switched fabric

## Status

This is an **implementation-ready, unqualified switch port** of SparkRing's
GLM-5.2 EXL3 3.5-bpw serving contract. The configuration, pinned upstream
runtime builder, image-bound exact-Q40 overlay generation, launcher wiring,
and fail-fast checks are present. The image build and four-node GPU run still
have to be executed on the Spark fleet.

The reference is SparkRing commit
`510556275ed3b77fc56a14367d319417072eeb8c`. Pinning is important because the
upstream repository changes rapidly.

## What remains identical

The switched recipe preserves the model-facing and execution contract from
SparkRing's GLM-5.2 quickstart:

| Setting | Switched recipe |
|---|---|
| Checkpoint | `brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78@9ab9579...` |
| Model integrity | Config/index pins plus every shard's pinned-revision LFS SHA-256, 157 shards, 346,218,639,128 indexed bytes |
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
- uses CX0 for rendezvous and both `mlx5_0,mlx5_1` switch-facing HCAs for
  collectives by default;
- enables cross-NIC routing and subnet-aware routing;
- does not force `NCCL_ALGO=Ring`, fixed channel counts, or
  `NCCL_SKIP_TREE_CONNECT`, allowing NCCL to choose switch-appropriate
  algorithms.

The two rails must both terminate on the intended lossless switch fabric, use
MTU 9000, and have working RoCEv2/PFC/ECN configuration. If only one adapter is
cabled, override `NCCL_IB_HCA` and `NCCL_CROSS_NIC=0` in the local environment
file. HCA enumeration must be checked on every rank;
the default names are not assumed to be universal.

The switch may improve collective performance, but that is a hypothesis, not
a property of the topology. Preserve the exact serving knobs and compare the
normalized 2K through 128K prefill and C1/C2/C4/C8 decode matrix against the
published SparkRing numbers.

## Build and launch

First audit the SPDX license expression for the immutable parent image and set
it locally. It is intentionally not guessed by the repository.

```bash
cp .env.example .env.sparkring-switch
# Set SPARK_SSH_USER and SPARKRING_BASE_IMAGE_LICENSES in that file.
# Override NCCL_IB_HCA if the two adapters are not mlx5_0 and mlx5_1.

# `prepare` builds/distributes the runtime and downloads/verifies the model.
bash scripts/launch-glm52-sparkring-switch.sh prepare
bash scripts/launch-glm52-sparkring-switch.sh preflight
bash scripts/launch-glm52-sparkring-switch.sh start
bash scripts/launch-glm52-sparkring-switch.sh benchmark
```

The repository's `benchmark` action is a functional deployment diagnostic. It
is not the `llm_decode_bench.py` normalized harness used for SparkRing's
published table, which is referenced by hash in upstream receipts but is not
distributed in the upstream repository. Its JSON must not be compared
directly with that table. A performance claim requires obtaining and
hash-verifying that external harness, then reproducing its cold, unique-prompt
2K/8K/16K/32K/64K/128K and C1/C2/C4/C8 matrix on both topologies.

Use the `build` action instead when the pinned model is already present on all
four ranks and only the runtime image/Q40 bundle needs to be rebuilt.

The custom build path runs on rank 0. It checks out the pinned SparkRing
revision, uses SparkRing's receipt-gated R7 image builder, generates the two
exact-Q40 overlays for the produced immutable image ID, and then distributes
the image, exact model verifier, and overlays to ranks 1-3. The `prepare`
action runs SparkRing's full pinned-revision LFS hash verification on every
rank. Preflight rejects image, model, overlay, or revision drift before a
serving container starts.

The wrapper deliberately uses `.env.sparkring-switch`, not the shared `.env`,
so model paths or revisions from the GLM-5.3 deployment cannot override this
hash-bound contract. Set `LOCAL_ENV_FILE` explicitly if another ignored local
file is preferred.

Exact-Q40 receipts are create-once upstream. On restart, the node launcher
moves an existing receipt to a timestamped backup before starting, preserving
the evidence while giving the new process a fresh namespace.

## Future model candidates

Do not replace only `MODEL_ID`. A candidate needs its own immutable revision,
config/index hashes, shard count, indexed byte size, served name, model/cache
paths, and then validation of:

1. architecture and TP4 checkpoint slicing;
2. tokenizer, reasoning parser, tool parser, and chat template;
3. DCP4 `ag_rs`, global TopK, draft sharding, and full-CKV gather;
4. EXL3 mixed-trellis target layers and online-quantization ignore rules;
5. MTP layer count, draft quantization, and the fixed speculative depth;
6. Q1-Q40 graph shapes and exact-Q40 target/draft state assumptions;
7. dynamic NVFP4 KV record size, cache capacity, and long-context correctness.

The existing Q40 overlay is bound specifically to the GLM-5.2 source bytes and
layer geometry. It must be regenerated or replaced with a candidate-specific,
hash-bound overlay; bypassing its guard is not a supported migration path.

## Qualification gates

- image labels identify the pinned SparkRing revision and its runtime verifier
  passes on all four ranks;
- model hashes, shard inventory, and revision marker pass on all ranks;
- both switch rails pass RoCE validation without retries, fallback to sockets,
  or asymmetric HCA selection;
- all four exact-Q40 attestations are fresh and consistent;
- `/health` and `/v1/models` report the expected model and 1M context;
- finite decode, tool calling, reasoning output, prefix caching, and long
  prefill pass with no OOM, NCCL, CUDA, or repeated-token failures;
- graph capture includes Q1-Q40 and post-workload rank/transport health passes;
- the normalized benchmark is run with cold unique prompts and compared with
  SparkRing's published baseline before any speed claim is made.
