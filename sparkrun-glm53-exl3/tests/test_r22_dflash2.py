#!/usr/bin/env python3
"""Static/source-contract checks for the R22 EXL3 image and MTP3 recipe."""

from __future__ import annotations

import importlib.util
import os
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = (ROOT / "Dockerfile.r22-dflash2").read_text(encoding="utf-8")
RECIPE = (ROOT / "recipes/glm53-exl3-dflash2-4x.yaml").read_text(encoding="utf-8")
BUILDER = (ROOT / "scripts/build-r22-dflash2-image.sh").read_text(encoding="utf-8")
PATCH_PATH = ROOT / "overlay/patch_r22_exl3.py"
EXLLAMA_ARM_PATCH_PATH = ROOT / "overlay/patch_exllamav3_aarch64.py"
SMOKE_PATCH_PATH = ROOT / "overlay/patch_smoke_r22_registration.py"
SMOKE_PATH = ROOT / "overlay/smoke_r22_image.py"

SPEC = importlib.util.spec_from_file_location("patch_r22_exl3", PATCH_PATH)
assert SPEC is not None and SPEC.loader is not None
overlay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(overlay)

ARM_SPEC = importlib.util.spec_from_file_location(
    "patch_exllamav3_aarch64", EXLLAMA_ARM_PATCH_PATH
)
assert ARM_SPEC is not None and ARM_SPEC.loader is not None
arm_patch = importlib.util.module_from_spec(ARM_SPEC)
ARM_SPEC.loader.exec_module(arm_patch)

SMOKE_SPEC = importlib.util.spec_from_file_location(
    "patch_smoke_r22_registration", SMOKE_PATCH_PATH
)
assert SMOKE_SPEC is not None and SMOKE_SPEC.loader is not None
smoke_patch = importlib.util.module_from_spec(SMOKE_SPEC)
SMOKE_SPEC.loader.exec_module(smoke_patch)


def test_immutable_arm64_source_composition() -> None:
    assert "ARG SPARKRING_FABRIC_IMAGE=" in DOCKERFILE
    assert "FROM ${SPARKRING_FABRIC_IMAGE} AS sparkring-fabric" in DOCKERFILE
    assert (
        "FROM vllm/vllm-openai:nightly-aarch64@sha256:"
        "a551e05307cd2e0092139d84db32af9c97e67d2eeeff072d21e429131d8c23f0"
        in DOCKERFILE
    )
    for identity in (
        "70b3c1c7f1c76fcf0847fcbb4a0b8b5583b78d19",
        "89481110674c08be1759a9222c525a0be14ad52a",
        "3c0a496caf9339f396b0be8da6910b1887920709",
        "c7345580eb4e4753420ebae812f5ec12a442c95a",
        "1e59a1fd09f782d302b1068b15c8a0bd66103894",
        "f322c804eec1c58a63bd4fe6e7901a95a678a575",
        "704aefd743b390af4bd0fb429d1906f9b964c7d8",
    ):
        assert identity in DOCKERFILE
    assert "CUTE_DSL_ARCH=sm_121a" in DOCKERFILE
    assert "TORCH_CUDA_ARCH_LIST=12.1a" in DOCKERFILE
    assert "CUDA_HOME=/usr/local/cuda" in DOCKERFILE
    assert "TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas" in DOCKERFILE
    assert "test -x /usr/local/cuda/bin/ptxas" in DOCKERFILE
    for package in (
        "cuda-nvrtc-${cuda_version_dash}",
        "cuda-nvrtc-dev-${cuda_version_dash}",
        "libcublas-dev-${cuda_version_dash}",
        "libcurand-dev-${cuda_version_dash}",
        "libcusolver-dev-${cuda_version_dash}",
        "libcusparse-dev-${cuda_version_dash}",
    ):
        assert package in DOCKERFILE
    assert "test -r /usr/local/cuda/include/cusparse.h" in DOCKERFILE
    assert "test -e /usr/local/cuda/lib64/libnvrtc.so" in DOCKERFILE
    for build_requirement in (
        '"setuptools==80.10.2"',
        '"setuptools-scm==10.2.3"',
        '"setuptools-rust==1.13.0"',
        '"wheel==0.48.0"',
    ):
        assert build_requirement in DOCKERFILE
    assert "COPY --from=sparkring-fabric /opt/sparkring/nccl" in DOCKERFILE
    assert 'test -n "${SPARKRING_FABRIC_IMAGE_ID}"' in DOCKERFILE
    assert "test -r /opt/sparkring/nccl/libnccl.so.2" in DOCKERFILE
    assert "test -r /usr/local/cuda/compat/libcuda.so.1" in DOCKERFILE
    assert "COPY overlay/smoke_r22_image.py" in DOCKERFILE
    assert "&& python3 /opt/compose/smoke_r22_image.py" in DOCKERFILE
    assert 'sparkring.fabric.image.id="${SPARKRING_FABRIC_IMAGE_ID}"' in DOCKERFILE
    assert "torch_version.release[:2] == (2, 13)" in DOCKERFILE
    assert "nvidia-cutlass-dsl\") == \"4.6.2\"" in DOCKERFILE
    assert "vllm-openai:jovian-judgement" not in DOCKERFILE


def test_exllamav3_build_has_aarch64_cpu_probe_compatibility() -> None:
    source = EXLLAMA_ARM_PATCH_PATH.read_text(encoding="utf-8")
    assert '"avx2_target.cpp"' in source
    assert '"avx512_target.cpp"' in source
    assert '"all_reduce_cpu_avx2.cpp"' in source
    assert '"all_reduce_cpu_avx512.cpp"' in source
    assert "#if defined(__aarch64__)" in source
    assert 'f"{signature} {{ return false; }}"' in source
    assert "ExLlamaV3 AVX2 CPU all-reduce is unavailable on AArch64" in source
    assert "ExLlamaV3 AVX-512 CPU all-reduce is unavailable on AArch64" in source
    assert "COPY overlay/patch_exllamav3_aarch64.py" in DOCKERFILE
    assert "patch_exllamav3_aarch64.py /opt/exllamav3-python --check" in DOCKERFILE

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source_dir = root / "exllamav3" / "exllamav3_ext"
        parallel_dir = source_dir / "parallel"
        parallel_dir.mkdir(parents=True)
        (source_dir / "avx2_target.cpp").write_text(
            'bool is_avx2_supported() { return __builtin_cpu_supports("avx2"); }\n',
            encoding="utf-8",
        )
        (source_dir / "avx512_target.cpp").write_text(
            'bool is_avx512_supported() { return __builtin_cpu_supports("avx512f"); }\n',
            encoding="utf-8",
        )
        for name in ("all_reduce_cpu_avx2.cpp", "all_reduce_cpu_avx512.cpp"):
            (parallel_dir / name).write_text(
                "#include <immintrin.h>\nvoid perform_cpu_reduce() {}\n",
                encoding="utf-8",
            )
        arm_patch.patch(root, check=False)
        arm_patch.patch(root, check=True)
        for path in source_dir.rglob("*.cpp"):
            patched = path.read_text(encoding="utf-8")
            assert patched.count(arm_patch.MARKER) == 1, path


def test_exl3_overlay_is_fail_closed_and_uses_r22_b12x_abi() -> None:
    source = PATCH_PATH.read_text(encoding="utf-8")
    assert overlay.R22_COMMIT == "70b3c1c7f1c76fcf0847fcbb4a0b8b5583b78d19"
    assert overlay.EXL3_BLOB == "384d1fb36be6887b251329cc862fc1d00be249b5"
    assert overlay.EXL3_CACHE_BLOB == "335b04a5291e0754affbb151cbb532e66e5579ad"
    assert "module.bind_mixed_trellis(" in source
    assert "module.run_bound_mixed_trellis(" in source
    assert "module.bind_mixed_trellis3(" in source
    assert "module.run_bound_mixed_trellis3(" in source
    assert 'kwargs["w13_layout"] = "trellis_t256_proj"' in source
    assert "expected exactly one source anchor" in source


def test_sparkrun_mounts_only_the_target_for_model_native_mtp() -> None:
    target = (
        "/home/juho/.cache/huggingface/hub/"
        "models--davidsyoung--GLM-5.3-EXL3-TR3-3.42bpw"
    )
    assert f"model: {target}" in RECIPE
    assert "models--incoai--GLM-5.3-DFlash2" not in RECIPE
    assert "draft_model:" not in RECIPE
    assert 'draft_root="{draft_model}"' not in RECIPE
    assert '\\"model\\":\\"$model_path\\",\\"method\\":\\"mtp\\"' in RECIPE


def test_four_spark_mtp3_runtime_contract() -> None:
    required = (
        "name: glm53-exl3-r22-mtp3-4x",
        "min_nodes: 4",
        "max_nodes: 4",
        "tensor_parallel: 4",
        "decode_context_parallel: 4",
        "gpu_memory_utilization: 0.895",
        "VLLM_USE_V2_MODEL_RUNNER: \"1\"",
        "VLLM_WORKER_MULTIPROC_METHOD: spawn",
        "PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True",
        "HF_HUB_OFFLINE: \"1\"",
        "TRANSFORMERS_OFFLINE: \"1\"",
        "INSTANTTENSOR_BUFFER_SIZE: \"536870912\"",
        "INSTANTTENSOR_CONCURRENCY: \"1\"",
        "INSTANTTENSOR_IO_DEPTH: \"3\"",
        "VLLM_EXL3_ONLINE_TRELLIS_BITS: \"6\"",
        "VLLM_B12X_MLA_CKV_GATHER: \"1\"",
        "VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE: \"1\"",
        "VLLM_DCP_GLOBAL_TOPK: \"1\"",
        "VLLM_DCP_SHARD_DRAFT: \"1\"",
        "VLLM_MTP_INSTRUMENT: \"1\"",
        "VLLM_USE_MEGA_AOT_ARTIFACT: \"1\"",
        "VLLM_ENABLE_PCIE_ALLREDUCE: \"0\"",
        "--block-size 2048",
        "--dtype bfloat16",
        "--quantization exl3",
        "--load-format instanttensor",
        "--model-loader-extra-config '{\"instanttensor_copy\":false}'",
        "--attention-backend B12X_MLA_SPARSE",
        "--disable-custom-all-reduce",
        "--no-enable-flashinfer-autotune",
        "--decode-context-parallel-size {decode_context_parallel}",
        '\\"method\\":\\"mtp\\"',
        '\\"num_speculative_tokens\\":3',
        '\\"draft_tensor_parallel_size\\":4',
        '\\"draft_sample_method\\":\\"greedy\\"',
        '"cudagraph_capture_sizes":[4,8,12,16,20,24,28,32]',
    )
    for value in required:
        assert value in RECIPE, value
    assert '\\"method\\":\\"dflash\\"' not in RECIPE
    assert '\\"num_speculative_tokens\\":7' not in RECIPE
    for inapplicable in (
        "PYTHONPATH",
        "CUDA_VISIBLE_DEVICES",
        "SAFETENSORS_FAST_GPU",
        "VLLM_PLUGINS",
        "VLLM_SSM_CONV_STATE_LAYOUT",
        "B12X_POLICY_MODE",
        "NCCL_NET_PLUGIN",
        "--trust-remote-code",
        "--block-size 16",
        "--kv-cache-memory-bytes",
        "--mamba-cache-mode",
        "--gdn-decode-kernel",
        "--linear-backend b12x",
        "--quantization modelopt_mixed",
        "--mm-encoder-tp-mode",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_xml",
    ):
        assert inapplicable not in RECIPE, inapplicable
    assert 'OMP_NUM_THREADS: "16"' not in RECIPE


def test_cluster_image_distribution_is_identity_checked() -> None:
    assert "if (( $# != 3 )); then" in BUILDER
    assert 'test "$(uname -m)" = "aarch64"' in BUILDER
    assert "--platform linux/arm64" in BUILDER
    assert 'docker image inspect "$fabric_base"' in BUILDER
    assert 'SPARKRING_FABRIC_IMAGE_ID=$fabric_id' in BUILDER
    assert 'test "$recorded_fabric_id" = "$fabric_id"' in BUILDER
    assert "docker save" in BUILDER and "docker load" in BUILDER
    assert "remote_id=" in BUILDER and 'test "$remote_id" = "$local_id"' in BUILDER
    assert BUILDER.count('test "$remote_platform" = "linux/arm64"') == 1
    assert BUILDER.count("/opt/compose/smoke_r22_image.py --gpu") == 2
    assert "r22-dflash2-sm121-v1" in BUILDER


def test_gpu_smoke_exercises_sm121_and_cuda_graph_replay() -> None:
    smoke = SMOKE_PATH.read_text(encoding="utf-8")
    assert "capability == (12, 1)" in smoke
    assert "torch.mm(left, right, out=static_out)" in smoke
    assert "torch.cuda.CUDAGraph()" in smoke
    assert "graph.replay()" in smoke
    assert 'Path("/opt/sparkring/nccl/libnccl.so.2").is_file()' in smoke


def test_r22_smoke_registration_patch_is_late_and_fail_closed() -> None:
    patch = SMOKE_PATCH_PATH.read_text(encoding="utf-8")
    assert 'exl3_config_cls = get_quantization_config("exl3")' in patch
    assert 'exl3_config_cls().get_name() == "exl3"' in patch
    copy = "COPY overlay/patch_smoke_r22_registration.py"
    install = "python3 -m pip install --no-cache-dir --no-build-isolation --no-deps"
    assert DOCKERFILE.index(copy) > DOCKERFILE.rindex(install)
    assert "patch_smoke_r22_registration.py" in DOCKERFILE
    assert "/opt/compose/smoke_r22_image.py --check" in DOCKERFILE

    with tempfile.TemporaryDirectory() as temporary:
        smoke = Path(temporary) / "smoke.py"
        smoke.write_text(smoke_patch.OLD, encoding="utf-8")
        smoke_patch.patch(smoke, check=False)
        smoke_patch.patch(smoke, check=True)
        assert smoke.read_text(encoding="utf-8") == smoke_patch.NEW


def test_exact_patched_tree_when_available() -> None:
    source = os.environ.get("GLM53_R22_PATCHED_SOURCE")
    if source:
        overlay.verify_patched(Path(source))


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"R22 EXL3 + MTP3 deployment OK ({len(tests)} tests)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
