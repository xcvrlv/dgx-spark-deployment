# Pins and local overlay

Vendored source: MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks, commit `e59e6eb67479aa68f6fa700c600dc90a0729b5ec`, downloaded as the GitHub source archive on 2026-09-12. Original licenses and NOTICE remain under `upstream/`.

Base: `lmsysorg/sglang@sha256:b4a4745fab5393dc0aca573fe754b86477f57691c9e971390b21102d3d94ebf9` (linux/arm64). OCI label `org.opencontainers.image.revision`: `da64c5cbb8cf6bfd39be19da43573fdfd484c43a`. FlashInfer: 0.6.18. Model revision: `fb2764a5cf321eaa5070ca8f9e892818f477c16d`.

Local modifications in the vendored tree:

- `Dockerfile`: pin ARM64 base and apply the checked top-k patch during build.
- `scripts/patch_topk.py`: extend the metadata gate; remove the raw-index v1 fallback using v2's existing output support; record source hashes.
- `adapter/topk_policy.py` and `adapter/sitecustomize.py`: check effective config in target/draft processes and construct fixed serving arguments.
- `boot.py`: enforce top-k and NVMe/TP4 policy; validate FP8 checkpoint headers and local packed shards; record the selection.
- `start.sh`: enforce policy after environment loading; forward it to head and workers; add `plan` and `check-topk`; invoke stop through Bash for Windows-checkout portability.
- `tests/check_topk_gpu.py`: GPU selection and CUDA graph checks for both k values; `check_patched_backend.py` verifies the actual patched metadata gate and raw/candidate dispatch during image build.

Everything else in `upstream/` is retained from the specified MiaAI revision. Use the outer `serve-2048.sh` / `serve-512.sh` entry points; the original entry points do not select this fleet policy.

Research reference for RoCEnante: local-inference-lab/b12x `081b235931dbbcedcf0eb5899bae990c5dec5238`. It is not installed or enabled by this image.
