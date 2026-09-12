# DeepSeek V4.1 Flash: four Sparks, FP8 Engram on SSD

Based on [MiaAI-Lab's Spark recipe](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/tree/e59e6eb67479aa68f6fa700c600dc90a0729b5ec), using SGLang, TP4/EP4, native MXFP4 experts and FP8 Engram rows packed onto each node's local SSD.

Two entry points select an explicit value on every rank:

| Script | `index_topk` |
|---|---:|
| `serve-2048.sh` | **2048**, forced |
| `serve-512.sh` | **512**, the checkpoint default |

These operate the same four containers. Run one profile at a time. The 2048 setting changes model behavior; quality and throughput need an A/B measurement.

## Run from Spark 1

Copy this entire directory to Spark 1. Configure the original FP8-Engram checkpoint, passwordless SSH, and the interface/key names in the fleet file:

```bash
cd ds41-sglang
cp fleet.env.example .env.fleet
# Edit .env.fleet: MODEL_DIR, SSH_IDENTITY and the network interface names.
bash serve-2048.sh plan
bash serve-2048.sh doctor
bash serve-2048.sh build
bash serve-2048.sh share
bash serve-2048.sh pack
bash serve-2048.sh check-topk
bash serve-2048.sh serve
```

The fleet addresses are `192.168.0.1` through `.4`, user `juho`, HCAs `rocep1s0f0,roceP2p1s0f0`, GID 3, API port 8000. The interface auto-detection in the example expects the head CX0 address to have a `/24` prefix. Set the interface explicitly if that differs; confirm matching interface names on the workers.

`MODEL_DIR` must contain the original `deepseek-ai/DeepSeek-V4.1-Flash` checkpoint (revision `fb2764a5cf321eaa5070ca8f9e892818f477c16d`). The existing `DeepSeek-V4.1-Flash-MXFP4-FP4-Engram` hybrid in the older deployment is incompatible with this FP8 reader. `download` is also an available command. Workers mount the checkpoint from the head over NFS; `pack` writes their rank's FP8 rows locally, about **47.2 GiB per node** for both tables, plus headers. Serving refuses missing or incorrectly sized/formatted local shards instead of silently reading Engram rows from NFS. Use these dedicated shard directories only for this checkpoint revision.

Switch profiles after stopping the running service:

```bash
bash serve-2048.sh stop
bash serve-512.sh serve
```

Other commands: `status`, `logs [worker1|worker2|worker3] [lines]`, `smoke`. Plain `bash serve-2048.sh` and `bash serve-512.sh` default to `serve`. `DSV41_ENV_FILE=/absolute/path/to/env` selects another fleet settings file.

## Enforcement and verification

The image is pinned by ARM64 digest. Its build patches the metadata gate from `(512, 1024)` to `(512, 1024, 2048)` and fixes the raw-index fallback that otherwise selects the v1 kernel. The patch checks the audited backend's SHA-256 before changing anything and writes `/opt/dsv41/topk-patch.json`.

After reading `.env.fleet`, the launcher forces TP4/EP4, NVMe offload and the script's top-k value. Boot adds `--json-model-override-args '{"index_topk":2048}'` (or 512), forces the SGLang v2 selector, and rejects conflicting extra config/backend arguments. A worker import hook checks the effective model config, including draft model construction, and raises on a mismatch. Each rank records `index_topk` and arguments in `launch.json` and logs `[topk-policy] ... index_topk=...`.

`check-topk` runs on all four GPUs without loading weights. It checks paged selection, raw indices, ragged selection, short/empty rows and CUDA graph replay at both 512 and 2048, comparing selected indices against PyTorch. Run it before serving, while GPUs have room for kernel compilation.

The TP4 sizing is inherited: 1,048,576 context, a 4M-token KV pool, eight requests, 4096-token prefill chunks. These are configured limits, not validation of 1M-context serving at top-k 2048. Ramp context and concurrency with the included upstream benchmarks and memory guard. Re-profile DSpark's SPS tables for each top-k setting; absent tables retain the upstream verify-all schedule.

Local validation completed: policy unit tests, Bash syntax checks, both profiles rendered with conflicting environment settings, and the source patch applied to the exact image revision. **No image build, GPU test, or four-node serving benchmark has been run in this Windows workspace.**

CPU regression commands (from this directory): `python3 -m unittest discover -s tests -v` and `bash tests/test_launch.sh`. The image build also executes the metadata-gate and raw/candidate dispatch checks against its patched SGLang source.

See [AUDIT.md](AUDIT.md) for the complete gate findings and RoCEnante assessment, and [UPSTREAM.md](UPSTREAM.md) for pins and modifications.
