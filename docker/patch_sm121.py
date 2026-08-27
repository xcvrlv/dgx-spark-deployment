from importlib.util import find_spec
from pathlib import Path


def package_root(name: str) -> Path:
    spec = find_spec(name)
    if spec is None or spec.submodule_search_locations is None:
        raise SystemExit(f"package not found: {name}")
    return Path(next(iter(spec.submodule_search_locations)))


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    source = path.read_text()
    count = source.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one source match in {path}, found {count}")
    path.write_text(source.replace(old, new))


vllm = package_root("vllm")
flashinfer = package_root("flashinfer")

# Make the NoPE FlashInfer MLA path available on SM121. GLM has pe_dim=0, so
# the SM120 DeepSeek-specific packed-cache backend cannot represent its cache.
replace_once(
    vllm / "platforms/cuda.py",
    """        elif device_capability.major == 12:
            return [
                AttentionBackendEnum.TRITON_MLA,
                AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120,
            ]""",
    """        elif device_capability.major == 12:
            return [
                AttentionBackendEnum.TRITON_MLA,
                AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM90,
                AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120,
            ]""",
    "SM121 MLA candidates",
)

sm90_backend = vllm / "v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py"
replace_once(
    sm90_backend,
    "    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:\n        return capability.major == 9\n",
    "    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:\n        return capability.major in (9, 12)\n",
    "NoPE MLA compute capability",
)
replace_once(
    sm90_backend,
    '            backend="fa3",\n',
    '            backend=("fa3" if torch.cuda.get_device_capability()[0] == 9 else "fa2"),\n',
    "NoPE MLA kernel backend",
)
replace_once(
    sm90_backend,
    """        if not has_flashinfer_sm90_nope_mla():
            return (
                "FLASHINFER_MLA_SPARSE_SM90 requires FlashInfer with SM90 "
                "MLA support (ckv_scale_arr in "
                "BatchMLAPagedAttentionWrapper.run, FlashInfer >= 0.6.18)"
            )""",
    """        if kv_cache_dtype in ("fp8", "fp8_e4m3") and not has_flashinfer_sm90_nope_mla():
            return (
                "FLASHINFER_MLA_SPARSE_SM90 fp8 KV requires FlashInfer with "
                "SM90 MLA support (ckv_scale_arr in "
                "BatchMLAPagedAttentionWrapper.run, FlashInfer >= 0.6.18)"
            )""",
    "NoPE MLA FlashInfer feature gate",
)

# PDL is not reliable for GLM's recurrent KDA state kernels on SM121.
replace_once(
    vllm / "platforms/cuda.py",
    """    @classmethod
    def is_arch_support_pdl(cls) -> bool:
        try:
            device = torch.cuda.current_device()
            major, _ = torch.cuda.get_device_capability(device)
        except Exception:
            return False
        return major >= 9
""",
    """    @classmethod
    def is_arch_support_pdl(cls) -> bool:
        try:
            device = torch.cuda.current_device()
            major, _ = torch.cuda.get_device_capability(device)
        except Exception:
            return False
        # SM12x PDL races with GLM's recurrent KDA state kernels.
        return major in (9, 10)
""",
    "PDL architecture gate",
)

# Sparse-indexer kernels may write fewer than select_k entries. Initialize the
# unused tail and reject pool IDs outside the current pool.
indexer = vllm / "model_executor/layers/sparse_attn_indexer_kpool.py"
replace_once(
    indexer,
    """                pool_topk = torch.empty(
                    (num_rows, select_k), dtype=torch.int32, device=logits.device
                )
""",
    """                pool_topk = torch.full(
                    (num_rows, select_k), -1, dtype=torch.int32, device=logits.device
                )
""",
    "prefill top-k initialization",
)
replace_once(
    indexer,
    """            pool_topk = torch.empty(
                (num_rows, select_k), dtype=torch.int32, device=logits.device
            )
""",
    """            pool_topk = torch.full(
                (num_rows, select_k), -1, dtype=torch.int32, device=logits.device
            )
""",
    "decode top-k initialization",
)

# GB10 exposes 48 resident CTAs at this TopK shape and 101,376 bytes of opt-in
# shared memory per block. The persistent TopK kernel can require more CTAs than
# SM121 can host, while its only C++ fallback requires 128 KiB shared memory.
# Use the existing multi-wave per-row kernel on small-SM devices instead.
replace_once(
    indexer,
    "        if select_k in (512, 1024, 2048):\n",
    """        use_persistent_topk = (
            select_k in (512, 1024, 2048)
            and torch.cuda.get_device_properties(logits.device).multi_processor_count >= 78
        )
        if use_persistent_topk:
""",
    "small-SM persistent TopK gate",
)
replace_once(
    vllm / "models/glm5next/nvidia/ops/kpool_compress.py",
    "    hist_out = tl.where(pid >= 0, hist_val, -1)\n",
    "    hist_out = tl.where((pid >= 0) & (pid < pool_len), hist_val, -1)\n",
    "kpool expansion bounds check",
)

# FP8 MLA on GB10 has less shared memory than Hopper. Honor the device-selected
# tile when it is smaller than the FP8 ceiling, and admit SM12x in the wrapper.
replace_once(
    flashinfer / "data/include/flashinfer/attention/mla.cuh",
    "    constexpr uint32_t EFF_CTA_TILE_KV = std::is_same_v<DTypeKV, __nv_fp8_e4m3> ? 32 : CTA_TILE_KV;\n",
    "    constexpr uint32_t EFF_CTA_TILE_KV = std::is_same_v<DTypeKV, __nv_fp8_e4m3> ? (CTA_TILE_KV < 32u ? CTA_TILE_KV : 32u) : CTA_TILE_KV;\n",
    "FP8 MLA shared-memory tile",
)
replace_once(
    flashinfer / "mla/_core.py",
    "            major, minor = get_compute_capability(self.device)\n            if major != 9:\n",
    "            major, minor = get_compute_capability(self.device)\n            if major not in (9, 12):\n",
    "FlashInfer FP8 MLA architecture gate",
)

print("Applied guarded GLM-5.3 SM121 compatibility patches")
