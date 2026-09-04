#!/usr/bin/env python3
"""Fail-closed image contract check for the R22 EXL3 + DFlash2 runtime."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path

from packaging.version import Version


VLLM_COMMIT = "70b3c1c7f1c76fcf0847fcbb4a0b8b5583b78d19"
B12X_COMMIT = "1e59a1fd09f782d302b1068b15c8a0bd66103894"
B12X_ROCENANTE_COMMIT = "1a7e3ec286b0ff0b7c2aabee22dce08daab7e011"
EXLLAMAV3_COMMIT = "704aefd743b390af4bd0fb429d1906f9b964c7d8"


def check_static_contract() -> dict[str, object]:
    import torch
    import vllm
    from vllm.model_executor.layers.quantization import get_quantization_config

    torch_version = Version(version("torch"))
    assert torch_version.release[:2] == (2, 13), torch_version
    assert Version(version("instanttensor")) >= Version("0.1.9")
    assert version("nvidia-cutlass-dsl") == "4.6.2"
    assert get_quantization_config("exl3").get_name() == "exl3"

    ext_root = Path(os.environ.get("VLLM_EXL3_EXT_PATH", "/opt/exllamav3"))
    assert ext_root.is_dir(), ext_root
    sys.path.insert(0, str(ext_root))
    ext = importlib.import_module("exllamav3_ext")
    for name in ("exl3_gemm", "exl3_moe_fused", "exl3_moe_fused_retile"):
        assert hasattr(ext, name), name

    mixed = importlib.import_module("b12x.moe._shared.kernels.w4a16.mixed_trellis")
    for name in (
        "bind_mixed_trellis",
        "run_bound_mixed_trellis",
        "bind_mixed_trellis3",
        "run_bound_mixed_trellis3",
    ):
        assert hasattr(mixed, name), name

    b12x = importlib.import_module("b12x")
    from b12x.comm import roce

    assert roce.API_VERSION == 1, roce.API_VERSION
    b12x_root = Path(b12x.__file__).parent
    roce_root = b12x_root / "comm/roce"
    oneshot = (roce_root / "_oneshot_cute.py").read_text(encoding="utf-8")
    runtime = (roce_root / "roce_oneshot.py").read_text(encoding="utf-8")
    proxy = (roce_root / "_roce_proxy.c").read_text(encoding="utf-8")
    assert "hca_count" in oneshot
    assert "def _grid_blocks" in runtime
    assert "#define ROCE_IDLE_SPINS 20000000" in proxy
    proxy_cache = Path(
        os.environ.get("B12X_ROCE_CACHE_DIR", "/opt/b12x-roce-cache")
    )
    assert any(proxy_cache.glob("roce_proxy-*.so")), proxy_cache

    root = Path(vllm.__file__).parent
    assert (root / "model_executor/layers/quantization/exl3.py").is_file()
    assert (root / "v1/worker/gpu/spec_decode/dflash2/speculator.py").is_file()
    assert (root / "model_executor/models/qwen3_dflash2.py").is_file()
    roce_adapter = root / "distributed/device_communicators/b12x_roce_all_reduce.py"
    communicator = root / "distributed/device_communicators/cuda_communicator.py"
    envs = (root / "envs.py").read_text(encoding="utf-8")
    assert roce_adapter.is_file(), roce_adapter
    assert "B12X_ROCENANTE" in communicator.read_text(encoding="utf-8")
    for name in (
        "VLLM_ENABLE_ROCE_ALLREDUCE",
        "VLLM_ROCE_ALLREDUCE_MAX_SIZE",
        "VLLM_ROCE_ALLGATHER_MAX_SIZE",
    ):
        assert name in envs, name
    assert Path("/opt/sparkring/nccl/libnccl.so.2").is_file()
    assert Path("/usr/local/cuda/compat/libcuda.so.1").is_file()

    return {
        "vllm": version("vllm"),
        "b12x": version("b12x"),
        "torch": str(torch_version),
        "instanttensor": version("instanttensor"),
        "cutlass_dsl": version("nvidia-cutlass-dsl"),
        "vllm_commit": VLLM_COMMIT,
        "b12x_commit": B12X_COMMIT,
        "b12x_rocenante_commit": B12X_ROCENANTE_COMMIT,
        "exllamav3_commit": EXLLAMAV3_COMMIT,
    }


def check_sm121() -> dict[str, object]:
    import torch

    assert torch.cuda.is_available(), "CUDA is unavailable inside the image"
    capability = torch.cuda.get_device_capability()
    assert capability == (12, 1), capability

    # Exercise allocation, GEMM dispatch, stream capture, and replay rather
    # than treating device enumeration as proof that the CUDA stack is usable.
    left = torch.randn((128, 128), dtype=torch.float16, device="cuda")
    right = torch.randn((128, 128), dtype=torch.float16, device="cuda")
    static_out = torch.empty_like(left)
    torch.mm(left, right, out=static_out)
    torch.cuda.synchronize()
    capture_stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        torch.mm(left, right, out=static_out)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(static_out).all().item()

    return {
        "device": torch.cuda.get_device_name(),
        "capability": f"{capability[0]}.{capability[1]}",
        "cuda": torch.version.cuda,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    result = check_static_contract()
    if args.gpu:
        result.update(check_sm121())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
