# DeepSeek-V4.1-Flash on the fleet, served by the dsv41 stack (temporary)

This directory integrates the [joe-spark-patches dsv41 patch stack](https://github.com/josephdrose/joe-spark-patches/tree/main/dsv41)
as the fleet's temporary DeepSeek-V4.1-Flash serving, pointed at the fleet's
MXFP4+FP4-Engram hybrid checkpoint with the Engram tables on NVMe. The six
files that make that work:

| Added file | What it does |
|---|---|
| `patch/engram-fp4-disk.py` | The FP4 adapter. Adds packed E2M1 Engram rows to the dsv41 stack's disk reader, applied on top of the installed tree. Every substitution asserts its occurrence count. See [docs/INTEGRATION.md](docs/INTEGRATION.md). |
| `build/ds41fp4-build-image.sh` | Fourth image layer: `vlspeed-eng:5` = `vlspeed-eng:4` + the FP4 adapter. Python only, about a second per box. The three upstream build scripts referenced their patch/ editors via `$HERE`, which resolves to `build/` — broken as shipped (the stale md5 entries are exactly these three scripts). Fixed here to `$HERE/../patch/`; upstream files otherwise verbatim. |
| `launch/ds41-flash-vlspeed-up.sh` | The wrapper. Reads `../ds4.1/cluster.json` (the fleet's node config) and launches the dsv41 stack on it. |
| `tests/test_engram_disk_fp4.py` | CPU contract test for the fp4 read path. 14 checks, all passing. |
| `tests/build_real_engram_table_fp4.py` | Builds one rank's real fp4 row file from the hybrid, with a bitwise `--verify`. |
| `tests/check_builder_fp4.py` | End-to-end check of the builder on a synthetic hybrid shard. |

Everything else here is the upstream dsv41 stack, byte-identical to upstream
main at commit `9d592115b721bd9ca5b353af4d86dd6a975b8628`. The upstream
build/launch scripts are kept verbatim for diffing against upstream; the
adaptation is an overlay, the same pattern the ds4.1 stack uses for its FP4
disk adapter.

## Why there is an FP4 adapter

The dsv41 stack's disk Engram reader stages `float8_e4m3fn[256] + ue8m0[8]`
rows — 264 bytes. The hybrid checkpoint stores the same table as
`uint8[rows, 128]` packed E2M1 (low nibble first) with the same-shaped E8M0
scale plane, so an unchanged boot dies at weight load, in
`_stage_disk_shard` -> `build_row_file`:

```
AssertionError: (torch.Size([96000564, 8]), 96000564, 128)
```

The adapter adds a `packed_fp4` path: 136-byte rows, packed nibbles decoded
directly in the CPU-side gather, E8M0 scales applied, bf16 out. This is not
another quantization step; it preserves the values of the downloaded FP4
checkpoint. SSD offload is the dsv41 stack's own `EngramConfig.table_path`
design — the table on NVMe, gather rows on the CPU with O_DIRECT — and the
fp4 path rides it unchanged. See [docs/INTEGRATION.md](docs/INTEGRATION.md)
for the failure verbatim, the cause, the change, and the 14-check CPU
contract test.

## Run on the head Spark

Copy this directory (or update the repository there). Use the existing
passwordless SSH and Docker setup. Stop the ds4.1 service first: co-tenancy
fails startup on these boxes.

```bash
# 0. The checkpoint is already staged on all four boxes
#    (ds4.1/cluster.json model_path, ~383.7 GiB per host).

# 1. Build the image. Four layers, none compiles vLLM from source.
./build/vl41-build-image.sh        # vl41-eng:2
./build/vlpage-build-image.sh      # vlpage-eng:3
./build/vlspeed-build-image.sh     # vlspeed-eng:4
./build/ds41fp4-build-image.sh     # vlspeed-eng:5 = vlspeed-eng:4 + the FP4 adapter

# 2. Optional: build and verify one rank's row files stand-alone, in a
#    scratch dir. The serve builds its own row files at weight load, fp4
#    included, under TABLE_HOST; stand-alone files are not consumed by it.
python3 tests/build_real_engram_table_fp4.py --out ~/cc-scratch/dsv41/table --layer 1  --rank 0 --verify 2000
python3 tests/build_real_engram_table_fp4.py --out ~/cc-scratch/dsv41/table --layer 14 --rank 0 --verify 2000

# 3. Serve. The wrapper reads ds4.1/cluster.json: hosts 192.168.0.1-4 (CX0),
#    HCAs rocep1s0f0,roceP2p1s0f0, GID 3, the hybrid at model_path, port 8000.
DRYRUN=1 ./launch/ds41-flash-vlspeed-up.sh     # print the per-rank scripts
./launch/ds41-flash-vlspeed-up.sh              # bring up at 1,048,576, DSpark k=5

# 4. Confirm.
curl -s localhost:8000/v1/models | grep DeepSeek-V4.1-Flash
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"DeepSeek-V4.1-Flash",
       "messages":[{"role":"user","content":"Name the four inner planets."}]}'
```

A correct response carries `content`, a populated `reasoning` field, and
`finish_reason: "stop"`. A tool call through the `deepseek_v41` parser
returns `finish_reason: "tool_calls"`.

## The serve

The defaults reproduce the upstream measured serve adapted to this fleet:
CTX=1048576, GPU_UTIL=0.78, MAXSEQS=8, MAXBATCH=8192, MOE_BACKEND=b12x with
`B12X_A16=1` (bf16 activations), DSpark k=5, CUDA graphs FULL_AND_PIECEWISE
with capture sizes derived from DSPARK and MAXSEQS. The launcher clears the
docker `--memory` cap above CTX 262144 and runs `drop_caches` +
`compact_memory` on all four boxes before the start — both load-bearing
upstream findings (docs 3 and 4 in the upstream README). `DEFRAG=0` skips
the second one. `DRYRUN=1` prints the resolved node config, the discovered
socket interfaces, and the per-rank docker scripts without starting
anything.

The ds4.1 cluster.json's 0.80 utilization and 16 sequences are tuned for the
ds4.1 stack, not measured with this one; override with `GPU_UTIL`/`MAXSEQS`
if wanted. CTX is mapped from cluster.json because it agrees with the
measured serve.

## The wrapper's node config

| Setting | Source |
|---|---|
| Hosts, SSH/control plane | `ds4.1/cluster.json` `nodes[].host` (192.168.0.1-4) |
| `VLLM_HOST_IP`, rendezvous, master-addr | `nodes[].cx0` (CX0), as in ds4.1 |
| NCCL HCA pair, GID | `hcas` (`rocep1s0f0,roceP2p1s0f0`), `gid_index` (3) |
| NCCL socket interface | Discovered from each node's CX0 IP, as in `ds4.1/cluster.py` |
| Checkpoint | `model_path`, mounted read-only at `/model` |
| Engram row files | `TABLE_HOST` (default `$CACHE_PATH/vlspeed-table`), built by the serve at weight load |
| Port, master port, served name | `port` (8000), `master_port` (29511), `served_model_name` (`DeepSeek-V4.1-Flash`) |
| Row files on NVMe | ~12.2 GiB per layer per rank (136-byte rows), built at weight load |

Rank 0 runs locally when the launch box owns the head's CX0 address, as in
`ds4.1/cluster.py`; remote ranks are reached over SSH. The upstream dsv41
launcher's fabric conventions (CX7 `f1` HCAs, `enP7s7` socket interface) do
not carry over: this cluster's HCAs are the `f0` pair, so the wrapper
discovers the interface per node and overrides the identities.

## Local validation completed

- The fp4 CPU contract test (`tests/test_engram_disk_fp4.py`): 14 checks, all
  passing, including the bitwise gather parity against a reference dequant of
  the packed bytes, four negative controls that had to fire, and the fp8
  regression guard.
- The builder end-to-end on a synthetic hybrid shard
  (`tests/check_builder_fp4.py`): row file, bitwise verify, seam-row picks,
  and the row-count insurance refusing a corrupted checkpoint.
- The editor's anchors verified against the real patched vLLM tree: a fresh
  copy of the PR #56214 tree at `e47aa780`, `engram-disk-table.patch`
  applied, then the FP4 editor against the two produced files. All five
  substitutions matched, the patched tree compiles, and an idempotent reapply
  fails loudly.
- Upstream files byte-identical to upstream main, except the six files this
  integration adds. `md5sum -c patch/md5sums.txt` passes for 18 of 21 files;
  the three build/launch scripts were changed by upstream commits after the
  manifest was written, so their manifest hashes are stale — the copies here
  are byte-identical to upstream main.
- Wrapper and build script syntax checked with `bash -n`.

CUDA kernels and the full-model serve cannot be exercised on the Windows
development host. The cluster checks are in
[docs/INTEGRATION.md](docs/INTEGRATION.md#not-proven).

## Provenance and validation limits

- vLLM PR tree: #56214 at `e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba`; wheel
  `vllm-0.28.1rc1.dev391+g29af8bd67` at the parent `29af8bd67`, per
  [docs/image-build.md](docs/image-build.md).
- Base image: `eugr/spark-vllm-b12x:latest`, pulled on all four boxes.
- The checkpoint: the fleet's MXFP4+FP4-Engram hybrid, staged by
  `scripts/prepare-deepseek-v41-hybrid.py` — upstream shards 1-46 at
  `dba1be0a`, the NVFP4 Engram repo's shards 47-48 at `dfce15b9`.
  `--engram-config '{"table_path":...}'` is required: the fp4 path is
  disk-only.
- The MoE backbone is MXFP4 in both checkpoints, so the b12x MoE path is
  unchanged. The tokenizer is the upstream one, unchanged.
- The dsv41 stack's numbers were measured on the upstream's own cluster with
  the official fp8 checkpoint. No number on the upstream README applies to
  the hybrid unchanged: the fp4 Engram table is about 12.2 GiB per layer per
  rank (136-byte rows) against the official 23.6 (264-byte rows), which
  frees roughly 23 GiB per rank for the KV pool. Concurrency, TTFT and
  quality on the hybrid are unmeasured.
- The manifest hashes in `patch/md5sums.txt` predate the upstream commits
  that changed the three build/launch scripts; the copies here are
  byte-identical to upstream main, verified by a fresh clone diff.

Sources: [dsv41 recipe](https://github.com/josephdrose/joe-spark-patches/tree/main/dsv41),
[ds4.1 cluster config](../ds4.1/cluster.json),
[B12x disk-table API](https://github.com/local-inference-lab/b12x/blob/00b69ac22e21413622c4ecd98f607a2c3e015161/b12x/sequence/engram/api.py).
