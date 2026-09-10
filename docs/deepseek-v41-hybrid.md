# DeepSeek V4.1 Flash: upstream MXFP4 + FP4 Engram

Run `scripts/prepare-deepseek-v41-hybrid.py` on Spark 1 from this repository.
It downloads upstream shards 1–46 at `dba1be0a40aa45a94ad051997016db3960a90277`
and LibertAI shards 47–48 at `dfce15b92ed1fa76e80e2a46ba847e5b5451f12c`.
No model conversion, GPU allocation, or full-table materialization is performed.
The combined checkpoint occupies approximately **383.7 GiB on each host**.

## Run

For a standalone script with authentication and the familiar HF storage location,
copy `scripts/download-deepseek-v41-hybrid.sh` to the head Spark and run:

```bash
bash download-deepseek-v41-hybrid.sh
```

It creates an isolated download environment, reuses existing HF authentication or
prompts for your token, then prepares and distributes the checkpoint. Its default
destination is `~/.cache/huggingface/hub/DeepSeek-V4.1-Flash-MXFP4-FP4-Engram`.
It respects `HF_HUB_CACHE` and `HF_HOME`; `HYBRID_MODEL_DIR` overrides the model
directory and `SPARK_PEERS` overrides the comma-separated peer list. Pass
`--user USERNAME` for a different SSH account. Tokens stay in the normal HF
authentication store and are not copied with the model.

This is a self-contained local model directory under the hub folder, not a
Hub-managed `models--.../snapshots/...` cache entry. Serve its absolute path;
loading by the upstream repository ID would select the original model instead.
The shell file embeds the Python preparer and requires no repository checkout.

Prerequisites: Python 3.10+, `python3-venv`, rsync, and passwordless SSH from
Spark 1 to the other Sparks. Python 3 and rsync must also be installed on peers.
Reserve about 390 GiB free on each destination filesystem. The script checks
available space, though interrupted partial files may require extra headroom.

```bash
python3 -m venv .venv-download
.venv-download/bin/pip install 'huggingface_hub>=1,<2'

# Inspect the selection without network requests or filesystem writes.
.venv-download/bin/python scripts/prepare-deepseek-v41-hybrid.py \
  --model-dir "$HOME/models/DeepSeek-V4.1-Flash-MXFP4-FP4-Engram" --copy --plan

# Download, assemble, validate, then sequentially copy and verify Sparks 2–4.
.venv-download/bin/python scripts/prepare-deepseek-v41-hybrid.py \
  --model-dir "$HOME/models/DeepSeek-V4.1-Flash-MXFP4-FP4-Engram" --copy
```

The default peers are `192.168.0.2,192.168.0.3,192.168.0.4` on the faster CX0 rail,
read from `sparks.env` by the Python preparer and embedded in the standalone script.
SSH defaults to your normal SSH configuration/current username. Override using
`--user YOUR_USER`, `--ssh-key /path/to/key`, or
`--hosts 192.168.0.2,192.168.0.3,192.168.0.4` for a working SSH-accessible CX rail.
`SPARK_SSH_USER` and `SPARK_SSH_KEY` environment variables are also accepted;
the script does not source `.env`. The same absolute model path is used on peers,
so pick a path writable by the SSH account on all four hosts.

Omit `--copy` to prepare only the head. Rerun the same command to resume a failed
download/copy or distribute later. HF retains local download metadata, and rsync
retains partial transfers. Rsync uses content checksums so retries also repair
same-size corrupt files. Verification rereads the weights, so allow time for
full-checkpoint disk reads (additional passes on retries). Peers receive the full checkpoint;
the serving loader is responsible for selecting each rank's resident tensors.

Use a new dedicated directory. The script refuses an unrelated existing local
or remote directory and uses pinned provenance to recognize its own retries.
It never deletes destination files except its own readiness marker during transfer.
Do not run preparation/copy concurrently or point a running service at a directory
being prepared. `HYBRID_READY.json` means preparation finished, not that inference
has been validated. No weights are downloaded by the `--plan` command.

## What is checked

- Downloaded LFS files are SHA256-checked against pinned Hugging Face metadata.
- All 48 safetensors headers are read without loading tensor payloads into RAM.
  Offsets/file lengths, exact upstream tensor-name mapping, and packed FP4 Engram
  weight/scale dtypes and widths are checked.
- `model.safetensors.index.json` is rebuilt, including actual tensor payload bytes
  (safetensors file headers are correctly excluded from `metadata.total_size`).
- Remote payloads are checked using `SHA256SUMS` before publishing readiness.
- Original upstream config/index and source revisions remain in the directory.

## Changes needed in an upstream recipe

Point the recipe at the local hybrid directory and provision locally instead of
downloading upstream again. Preserve the upstream `expert_dtype: fp4`, FP8 dense
settings and other model architecture fields. The script adds these descriptive
fields inside `quantization_config`:

```json
{
  "engram_dtype": "fp4",
  "engram_block_size": 32,
  "engram_scale_fmt": "ue8m0"
}
```

These fields describe LibertAI's format; they are **not a confirmed standard loader
interface**. Do not select global NVFP4 quantization or copy LibertAI's full expert
configuration into the hybrid. The rest of the model stays in upstream formats.

An engine that already serves upstream V4.1 still needs an Engram path that:

1. Accepts `uint8` packed E2M1 tables with logical row width 256, stored width 128,
   plus eight E8M0 scales per row. Both weights and recomputed scales must be used.
2. Unpacks/dequantizes gathered rows on demand instead of expanding the full tables
   to FP8/BF16 at load time, which would lose the intended memory saving.
3. Shards tables across the four ranks with correct distributed lookup semantics;
   keeping all tables on every rank would invalidate the ~95.9 GiB/rank ideal budget.

B12x MoE support does not automatically implement this Engram loader/gather path.
The amount of patching depends on the upstream V4.1 integration that lands; this
script makes the data ready but cannot promise only recipe flag changes will suffice.
Runtime overhead, replicated tensors, graph buffers, and temporary loading memory
remain additional to the ideal weight budget. Quality of this DeepSeek Engram
conversion has not yet been evaluated end to end.

Sources: [upstream](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash),
[LibertAI formats and compatibility](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-NVFP4),
[HF local-directory downloads](https://huggingface.co/docs/huggingface_hub/guides/download).
