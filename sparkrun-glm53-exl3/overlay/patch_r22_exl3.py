#!/usr/bin/env python3
"""Compose the qualified GLM-5.3 R22 tree with the R7 EXL3 backend.

The Docker build restores ``exl3.py`` and ``exl3_online_cache.py`` from the
immutable R7 EXL3 commit before running this patch.  Everything else remains
from the R22 vLLM commit.  This script deliberately uses exact, one-shot text
replacements so source drift fails the image build instead of producing a
plausible but unreviewed runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import py_compile
from pathlib import Path


R22_COMMIT = "70b3c1c7f1c76fcf0847fcbb4a0b8b5583b78d19"
EXL3_BLOB = "384d1fb36be6887b251329cc862fc1d00be249b5"
EXL3_CACHE_BLOB = "335b04a5291e0754affbb151cbb532e66e5579ad"


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{path}: expected exactly one source anchor, found {count}: {old[:80]!r}"
        )
    path.write_text(text.replace(old, new, 1), encoding="utf-8", newline="\n")


def git_dir(root: Path) -> Path:
    marker = root / ".git"
    if marker.is_dir():
        return marker
    if marker.is_file():
        prefix = "gitdir: "
        value = marker.read_text(encoding="utf-8").strip()
        if value.startswith(prefix):
            return (root / value[len(prefix) :]).resolve()
    raise RuntimeError(f"cannot resolve Git directory for {root}")


def detached_head(root: Path) -> str:
    value = (git_dir(root) / "HEAD").read_text(encoding="ascii").strip()
    if value.startswith("ref: "):
        raise RuntimeError("vLLM source must be checked out at a detached commit")
    return value


def git_blob_id(path: Path) -> str:
    payload = path.read_bytes()
    framed = f"blob {len(payload)}\0".encode("ascii") + payload
    return hashlib.sha1(framed, usedforsecurity=False).hexdigest()


def verify_source(root: Path) -> None:
    if detached_head(root) != R22_COMMIT:
        raise RuntimeError(f"vLLM source is not pinned to R22 commit {R22_COMMIT}")
    expected = {
        "vllm/model_executor/layers/quantization/exl3.py": EXL3_BLOB,
        "vllm/model_executor/layers/quantization/exl3_online_cache.py": EXL3_CACHE_BLOB,
    }
    for relative, blob in expected.items():
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"missing restored EXL3 source: {path}")
        actual = git_blob_id(path)
        if actual != blob:
            raise RuntimeError(
                f"{relative}: expected restored blob {blob}, found {actual}"
            )


def patch_quant_registry(root: Path) -> None:
    path = root / "vllm/model_executor/layers/quantization/__init__.py"
    replace_once(
        path,
        '    "fp8",\n    "fbgemm_fp8",\n',
        '    "fp8",\n    "fbgemm_fp8",\n    "exl3",\n',
    )
    replace_once(
        path,
        "    from .experts_int8 import ExpertsInt8Config\n",
        "    from .experts_int8 import ExpertsInt8Config\n"
        "    from .exl3 import Exl3Config\n",
    )
    replace_once(
        path,
        '        "fbgemm_fp8": FBGEMMFp8Config,\n',
        '        "fbgemm_fp8": FBGEMMFp8Config,\n'
        '        "exl3": Exl3Config,\n',
    )


def patch_quant_override_order(root: Path) -> None:
    path = root / "vllm/config/model.py"
    replace_once(
        path,
        '                "moe_wna16",\n'
        '                "modelopt",\n',
        '                "moe_wna16",\n'
        "                # Rank-sliced EXL3 checkpoints retain a ModelOpt dispatch tag\n"
        "                # for backward compatibility, so EXL3 must inspect metadata\n"
        "                # before the ModelOpt overrides claim them.\n"
        '                "exl3",\n'
        '                "modelopt",\n',
    )


def patch_sm121_flashmla_build(root: Path) -> None:
    path = root / "CMakeLists.txt"
    replace_once(
        path,
        "    include(cmake/external_projects/flashmla.cmake)\n",
        "    # FlashMLA only supplies sm90/sm100 kernels. This image targets sm121\n"
        "    # and uses B12X_MLA_SPARSE, so retain setup.py's optional targets\n"
        "    # without fetching FlashMLA and its otherwise-unused CUTLASS submodule.\n"
        "    add_custom_target(_flashmla_C)\n"
        "    add_custom_target(_flashmla_extension_C)\n",
    )


def patch_b12x_glm_dsa_fp8_abi(root: Path) -> None:
    path = root / "vllm/v1/attention/backends/mla/b12x_mla_sparse.py"
    replace_once(
        path,
        "_GLM_DSA_NVFP4_CACHE_RECORD_BYTES = 368\n",
        "_GLM_DSA_FP8_CACHE_RECORD_BYTES = 656\n"
        "_GLM_DSA_NVFP4_CACHE_RECORD_BYTES = 368\n",
    )
    replace_once(
        path,
        "        if self._is_glm_next and self._uses_nvfp4_cache:\n"
        "            self._cache_record_bytes = _GLM_NEXT_NVFP4_CACHE_RECORD_BYTES\n"
        "        elif self._uses_glm_dsa_nvfp4_cache:\n"
        "            self._cache_record_bytes = _GLM_DSA_NVFP4_CACHE_RECORD_BYTES\n"
        "        else:\n"
        "            self._cache_record_bytes = _GLM_NEXT_CACHE_RECORD_BYTES\n",
        "        if self._is_glm_next:\n"
        "            self._cache_record_bytes = (\n"
        "                _GLM_NEXT_NVFP4_CACHE_RECORD_BYTES\n"
        "                if self._uses_nvfp4_cache\n"
        "                else _GLM_NEXT_CACHE_RECORD_BYTES\n"
        "            )\n"
        "        elif self._is_glm_dsa:\n"
        "            self._cache_record_bytes = (\n"
        "                _GLM_DSA_NVFP4_CACHE_RECORD_BYTES\n"
        "                if self._uses_nvfp4_cache\n"
        "                else _GLM_DSA_FP8_CACHE_RECORD_BYTES\n"
        "            )\n"
        "        else:\n"
        "            self._cache_record_bytes = _GLM_NEXT_CACHE_RECORD_BYTES\n",
    )
    replace_once(
        path,
        "        elif self._uses_glm_dsa_nvfp4_cache:\n"
        "            self._model_type = int(module.ModelType.GLM_NSA)\n"
        "            self._concat_and_cache_nvfp4_mla_fp8_rope = (\n"
        "                module.concat_and_cache_nvfp4_mla_fp8_rope\n"
        "            )\n",
        "        elif self._is_glm_dsa:\n"
        "            self._model_type = int(module.ModelType.GLM_NSA)\n"
        "            if self._uses_glm_dsa_nvfp4_cache:\n"
        "                self._concat_and_cache_nvfp4_mla_fp8_rope = (\n"
        "                    module.concat_and_cache_nvfp4_mla_fp8_rope\n"
        "                )\n",
    )


def patch_rank_sliced_exl3_loaders(root: Path) -> None:
    target_path = root / "vllm/models/deepseek_v32/nvidia/model.py"
    replace_once(
        target_path,
        "        quant_config = vllm_config.quant_config\n"
        "        self.config = config\n",
        "        quant_config = vllm_config.quant_config\n"
        "        self.config = config\n"
        "        self.quant_config = quant_config\n",
    )
    replace_once(
        target_path,
        "        params_dict = dict(self.named_parameters())\n"
        "        loaded_params: set[str] = set()\n"
        "        _pending_wk_fp8: dict = {}\n"
        "        for name, loaded_weight in weights:\n"
        "            if \"rotary_emb.inv_freq\" in name:\n",
        "        params_dict = dict(self.named_parameters())\n"
        "        loaded_params: set[str] = set()\n"
        "        _pending_wk_fp8: dict = {}\n"
        "        rank_sliced_name = getattr(\n"
        "            self.quant_config,\n"
        '            "normalize_rank_sliced_weight_name",\n'
        "            None,\n"
        "        )\n"
        "        for name, loaded_weight in weights:\n"
        "            if rank_sliced_name is not None:\n"
        "                name = rank_sliced_name(name)\n"
        "                if name is None:\n"
        "                    continue\n"
        "            if \"rotary_emb.inv_freq\" in name:\n",
    )

    mtp_path = root / "vllm/model_executor/models/deepseek_mtp.py"
    replace_once(
        mtp_path,
        "        params_dict = dict(self.named_parameters())\n"
        "        loaded_params: set[str] = set()\n"
        "        _pending_wk_fp8: dict = {}  # FP8 indexer wk dequant buffer\n"
        "        for name, loaded_weight in weights:\n"
        "            if \"rotary_emb.inv_freq\" in name:\n",
        "        params_dict = dict(self.named_parameters())\n"
        "        loaded_params: set[str] = set()\n"
        "        _pending_wk_fp8: dict = {}  # FP8 indexer wk dequant buffer\n"
        "        # Match the target EXL3 loader: discard non-local serialized TP\n"
        "        # payloads and strip the rank segment before expert mapping.\n"
        "        rank_sliced_name = getattr(\n"
        "            self.quant_config,\n"
        '            "normalize_rank_sliced_weight_name",\n'
        "            None,\n"
        "        )\n"
        "        for name, loaded_weight in weights:\n"
        "            if rank_sliced_name is not None:\n"
        "                name = rank_sliced_name(name)\n"
        "                if name is None:\n"
        "                    continue\n"
        "            if \"rotary_emb.inv_freq\" in name:\n",
    )


def patch_quant_overlay(root: Path) -> None:
    path = root / "vllm/config/quantization.py"
    replace_once(
        path,
        "    string forms accepted on `linear` and `moe`.\n",
        "    string forms accepted on `linear`, `moe`, and `shared_experts`.\n",
    )
    replace_once(
        path,
        '    moe: QuantSpec | None = None\n    """Spec applied to ``FusedMoEFactory`` layers."""\n\n'
        "    ignore: list[str] = Field(default_factory=list)\n",
        '    moe: QuantSpec | None = None\n    """Spec applied to ``FusedMoEFactory`` layers."""\n\n'
        "    shared_experts: QuantSpec | None = None\n"
        '    """Spec applied only to shared-expert gate/up/down projections."""\n\n'
        "    ignore: list[str] = Field(default_factory=list)\n",
    )
    replace_once(
        path,
        '@field_validator("linear", "moe", mode="before")',
        '@field_validator("linear", "moe", "shared_experts", mode="before")',
    )

    anchor = "\n\ndef resolve_quantization_config(\n"
    overlay = '''

# Checkpoint backends keep ownership of serialized weights.  EXL3 may overlay
# online MXFP8 (or online Trellis selected by the EXL3 backend) only on BF16
# dense/shared-expert projections which are absent from EXL3 storage.
_CHECKPOINT_ONLINE_OVERLAY_WEIGHTS = {
    "exl3": frozenset({kMxfp8Dynamic}),
}


def _is_checkpoint_online_overlay(
    quantization: str | None, args: QuantizationConfigArgs
) -> bool:
    supported_weights = _CHECKPOINT_ONLINE_OVERLAY_WEIGHTS.get(quantization)
    if supported_weights is None or args.moe is not None:
        return False
    specs = [spec for spec in (args.linear, args.shared_experts) if spec is not None]
    if not specs:
        return False
    if args.ignore and args.linear is None:
        return False
    return all(
        spec.weight in supported_weights and spec.activation is None for spec in specs
    )


def resolve_quantization_config(
'''
    replace_once(path, anchor, overlay)
    replace_once(
        path,
        "    if quantization is not None and quantization not in ONLINE_QUANT_SHORTHAND_NAMES:\n"
        "        if quantization_config is not None:\n"
        "            raise ValueError(\n"
        "                f\"quantization_config is only supported when quantization is \"\n"
        "                f\"one of {sorted(ONLINE_QUANT_SHORTHAND_NAMES)}, \"\n"
        "                f\"got quantization={quantization!r}\"\n"
        "            )\n"
        "        return None\n\n"
        "    base = _ONLINE_SHORTHANDS.get(quantization) if quantization else None\n\n"
        "    if quantization_config is None:\n"
        "        return base\n\n"
        "    if isinstance(quantization_config, dict):\n"
        "        quantization_config = QuantizationConfigArgs(**quantization_config)\n",
        "    if isinstance(quantization_config, dict):\n"
        "        quantization_config = QuantizationConfigArgs(**quantization_config)\n\n"
        "    if quantization is not None and quantization not in ONLINE_QUANT_SHORTHAND_NAMES:\n"
        "        if quantization_config is not None and not _is_checkpoint_online_overlay(\n"
        "            quantization, quantization_config\n"
        "        ):\n"
        "            raise ValueError(\n"
        "                f\"quantization_config is not a supported checkpoint overlay for \"\n"
        "                f\"quantization={quantization!r}\"\n"
        "            )\n"
        "        return quantization_config\n\n"
        "    base = _ONLINE_SHORTHANDS.get(quantization) if quantization else None\n\n"
        "    if quantization_config is None:\n"
        "        return base\n",
    )
    replace_once(
        path,
        "        moe=quantization_config.moe or base.moe,\n"
        "        ignore=quantization_config.ignore or base.ignore,\n",
        "        moe=quantization_config.moe or base.moe,\n"
        "        shared_experts=quantization_config.shared_experts,\n"
        "        ignore=quantization_config.ignore or base.ignore,\n",
    )


def patch_exl3_backend(root: Path) -> None:
    path = root / "vllm/model_executor/layers/quantization/exl3.py"
    replace_once(
        path,
        "from vllm.model_executor.layers.quantization.online.mxfp8 import (\n"
        "    Mxfp8OnlineLinearMethod,\n"
        "    is_shared_expert_projection,\n"
        ")\n",
        "from vllm.model_executor.layers.quantization.online.mxfp8 import (\n"
        "    Mxfp8OnlineLinearMethod,\n"
        ")\n",
    )
    replace_once(
        path,
        "logger = init_logger(__name__)\n",
        "_SHARED_EXPERT_PROJECTIONS = frozenset(\n"
        '    {"gate_proj", "up_proj", "gate_up_proj", "down_proj"}\n'
        ")\n\n\n"
        "def is_shared_expert_projection(prefix: str) -> bool:\n"
        "    parts = prefix.split(\".\")\n"
        "    return (\n"
        "        len(parts) >= 2\n"
        "        and parts[-1] in _SHARED_EXPERT_PROJECTIONS\n"
        '        and parts[-2] in {"shared_expert", "shared_experts"}\n'
        "    )\n\n\n"
        "logger = init_logger(__name__)\n",
    )

    start = path.read_text(encoding="utf-8").index("def _load_b12x_mixed_trellis()")
    text = path.read_text(encoding="utf-8")
    end = text.index("\n\ndef _load_b12x_trellis_linear()", start)
    old = text[start:end]
    new = '''def _load_b12x_mixed_trellis() -> Any:
    """Resolve R22 B12X and adapt its bind-once mixed-Trellis ABI."""

    global _B12X_MIXED_TRELLIS_API
    if _B12X_MIXED_TRELLIS_API is not None:
        return _B12X_MIXED_TRELLIS_API
    try:
        module = importlib.import_module("b12x.moe._shared.kernels.w4a16.mixed_trellis")
        prepare = importlib.import_module("b12x.moe._shared.kernels.w4a16.prepare")
        host = importlib.import_module("b12x.moe._shared.kernels.w4a16.host")
    except Exception as exc:
        raise RuntimeError(
            "Mixed-bitrate rank-sliced EXL3 requires the R22 B12X "
            "mixed_trellis implementation."
        ) from exc

    bindings: dict[tuple[Any, ...], Any] = {}

    def prepare_weights(**kwargs):
        # R22 renamed the same padded preparation layout.  R7 replaces the
        # padded FC1 view with projection-tight storage immediately afterward.
        if kwargs.get("w13_layout") == "trellis3_t256_proj":
            kwargs = dict(kwargs)
            kwargs["w13_layout"] = "trellis_t256_proj"
        return prepare.prepare_trellis256_moe_weights(**kwargs)

    def run_mixed_trellis(
        x, tier0, tier1, topk_weights, topk_ids, global_to_combined,
        descriptor_map, rotations, launch, buffers, **kwargs
    ):
        key = (
            id(tier0), id(tier1), id(global_to_combined), id(descriptor_map),
            id(rotations), id(launch), tuple(sorted(kwargs.items())),
        )
        binding = bindings.get(key)
        if binding is None:
            binding = module.bind_mixed_trellis(
                tier0, tier1, global_to_combined, descriptor_map, rotations,
                launch, **kwargs
            )
            bindings[key] = binding
        return module.run_bound_mixed_trellis(
            x, topk_weights, topk_ids, binding, buffers
        )

    def run_mixed_trellis3(
        x, tier0, tier1, tier2, topk_weights, topk_ids, global_to_combined,
        descriptor_map, rotations, launch, buffers, **kwargs
    ):
        key = (
            id(tier0), id(tier1), id(tier2), id(global_to_combined),
            id(descriptor_map), id(rotations), id(launch),
            tuple(sorted(kwargs.items())),
        )
        binding = bindings.get(key)
        if binding is None:
            binding = module.bind_mixed_trellis3(
                tier0, tier1, tier2, global_to_combined, descriptor_map,
                rotations, launch, **kwargs
            )
            bindings[key] = binding
        return module.run_bound_mixed_trellis3(
            x, topk_weights, topk_ids, binding, buffers
        )

    api = SimpleNamespace(
        build_tiered_maps=module.build_tiered_maps,
        build_projection_tiered_maps=module.build_projection_tiered_maps,
        combine_trellis_rotations=module.combine_trellis_rotations,
        compile_mixed_trellis=module.compile_mixed_trellis,
        compile_mixed_trellis3=module.compile_mixed_trellis3,
        make_mixed_trellis_buffers=module.make_mixed_trellis_buffers,
        make_mixed_trellis3_buffers=module.make_mixed_trellis3_buffers,
        max_packed_route_slots=host.max_packed_route_slots,
        prepare_weights=prepare_weights,
        run_mixed_trellis=run_mixed_trellis,
        run_mixed_trellis3=run_mixed_trellis3,
        warmup_mixed_trellis_route_pack=module.warmup_mixed_trellis_route_pack,
    )
    _B12X_MIXED_TRELLIS_API = api
    return api
'''
    replace_once(path, old, new)


def patch_ballast_release(root: Path) -> None:
    path = root / "vllm/model_executor/model_loader/utils.py"
    replace_once(path, "import inspect\n", "import gc\nimport inspect\n")
    anchor = "\n    # Initialize post-load attention weights for any attention layer and MM\n"
    insertion = '''
    # EXL3's preparation ballast is validated but never read.  Release the one
    # shared pool after every routed-expert layer has been finalized so it does
    # not permanently reduce the KV-cache budget on unified-memory systems.
    try:
        from vllm.model_executor.layers.quantization import exl3 as _exl3_r7

        pool = getattr(_exl3_r7, "_R7_BALLAST_POOL", None)
        if pool:
            pool.clear()
            gc.collect()
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        logger.debug("EXL3 R7 ballast release skipped", exc_info=True)

    # Initialize post-load attention weights for any attention layer and MM
'''
    replace_once(path, anchor, insertion)


def patch_warmup(root: Path) -> None:
    path = root / "vllm/model_executor/warmup/kernel_warmup.py"
    replace_once(
        path,
        "from vllm.logger import init_logger\n",
        "from vllm.logger import init_logger\n"
        "from vllm.model_executor.layers.quantization.exl3 import (\n"
        "    warmup_exl3_mixed_trellis_route_pack,\n"
        ")\n",
    )
    replace_once(
        path,
        "    b12x_warmup(worker, cudagraph_capture_sizes)\n\n"
        "    minimax_m3_msa_warmup(worker)\n",
        "    b12x_warmup(worker, cudagraph_capture_sizes)\n\n"
        "    warmed_exl3 = warmup_exl3_mixed_trellis_route_pack(worker.get_model())\n"
        "    if warmed_exl3:\n"
        "        logger.info_once(\n"
        '            "Warmed up %d EXL3 mixed-Trellis route-pack variants.",\n'
        "            warmed_exl3,\n"
        "        )\n\n"
        "    minimax_m3_msa_warmup(worker)\n",
    )


def verify_patched(root: Path) -> None:
    checks = {
        "CMakeLists.txt": (
            "FlashMLA only supplies sm90/sm100 kernels",
            "add_custom_target(_flashmla_C)",
            "add_custom_target(_flashmla_extension_C)",
        ),
        "vllm/v1/attention/backends/mla/b12x_mla_sparse.py": (
            "_GLM_DSA_FP8_CACHE_RECORD_BYTES = 656",
            "elif self._is_glm_dsa:",
            "else _GLM_DSA_FP8_CACHE_RECORD_BYTES",
            "self._model_type = int(module.ModelType.GLM_NSA)",
        ),
        "vllm/models/deepseek_v32/nvidia/model.py": (
            "self.quant_config = quant_config",
            '"normalize_rank_sliced_weight_name"',
            "name = rank_sliced_name(name)",
        ),
        "vllm/model_executor/models/deepseek_mtp.py": (
            "Match the target EXL3 loader",
            '"normalize_rank_sliced_weight_name"',
            "name = rank_sliced_name(name)",
        ),
        "vllm/model_executor/layers/quantization/__init__.py": (
            '"exl3": Exl3Config',
        ),
        "vllm/config/model.py": (
            '                "exl3",\n                "modelopt",',
        ),
        "vllm/config/quantization.py": (
            "shared_experts: QuantSpec | None",
            '_CHECKPOINT_ONLINE_OVERLAY_WEIGHTS = {\n    "exl3"',
        ),
        "vllm/model_executor/layers/quantization/exl3.py": (
            "def run_mixed_trellis3(",
            "module.bind_mixed_trellis3(",
            'kwargs["w13_layout"] = "trellis_t256_proj"',
        ),
        "vllm/model_executor/model_loader/utils.py": (
            "EXL3 R7 ballast release skipped",
        ),
        "vllm/model_executor/warmup/kernel_warmup.py": (
            "warmup_exl3_mixed_trellis_route_pack(worker.get_model())",
        ),
    }
    for relative, markers in checks.items():
        text = (root / relative).read_text(encoding="utf-8")
        for marker in markers:
            if marker not in text:
                raise RuntimeError(f"{relative}: missing patched marker {marker!r}")

    for relative in (
        "vllm/model_executor/layers/quantization/exl3.py",
        "vllm/model_executor/layers/quantization/exl3_online_cache.py",
        "vllm/config/quantization.py",
        "vllm/v1/attention/backends/mla/b12x_mla_sparse.py",
        "vllm/models/deepseek_v32/nvidia/model.py",
        "vllm/model_executor/models/deepseek_mtp.py",
    ):
        py_compile.compile(str(root / relative), doraise=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = args.source.resolve()

    if args.check:
        verify_patched(root)
        return

    verify_source(root)
    patch_quant_registry(root)
    patch_quant_override_order(root)
    patch_sm121_flashmla_build(root)
    patch_b12x_glm_dsa_fp8_abi(root)
    patch_rank_sliced_exl3_loaders(root)
    patch_quant_overlay(root)
    patch_exl3_backend(root)
    patch_ballast_release(root)
    patch_warmup(root)
    verify_patched(root)


if __name__ == "__main__":
    main()
