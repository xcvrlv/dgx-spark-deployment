import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, value = stripped.partition("=")
        if not separator:
            raise AssertionError(f"invalid env line: {line}")
        values[key] = value
    return values


class DeploymentStaticTests(unittest.TestCase):
    def test_text_sources_have_linux_line_endings_and_no_trailing_space(self) -> None:
        source_suffixes = {".sh", ".py", ".env", ".md"}
        for path in ROOT.rglob("*"):
            if not path.is_file() or path.suffix not in source_suffixes:
                continue
            content = path.read_bytes()
            self.assertNotIn(b"\r", content, path)
            for line_number, line in enumerate(content.splitlines(), start=1):
                self.assertEqual(line, line.rstrip(), f"{path}:{line_number}")

    def test_inventory_has_four_unique_nodes_on_each_network(self) -> None:
        inventory = read_env(ROOT / "sparks.env")
        self.assertEqual(inventory["TP_NODE_COUNT"], "4")
        for suffix in ("MANAGEMENT_IP", "CX0_IP", "CX1_IP"):
            addresses = [inventory[f"SPARK_{node}_{suffix}"] for node in range(1, 5)]
            self.assertEqual(len(set(addresses)), 4)

    def test_recipe_pins_proven_non_ring_iteration_one_defaults(self) -> None:
        recipe = read_env(ROOT / "recipes/glm-5.3-flash-nvfp4.env")
        self.assertRegex(recipe["MODEL_REVISION"], r"^[0-9a-f]{40}$")
        self.assertEqual(
            recipe["MODEL_REVISION"], "8627752b10b78c2b0f2fc69790a94ec9f1ddaa26"
        )
        self.assertEqual(recipe["MODEL_ID"], "local-inference-lab/GLM-5.3-Flash-NVFP4")
        self.assertEqual(recipe["MTP_SPECULATIVE_TOKENS"], "0")
        self.assertEqual(recipe["MAX_MODEL_LEN"], "1048576")
        self.assertEqual(recipe["MAX_NUM_BATCHED_TOKENS"], "8192")
        self.assertEqual(recipe["KV_CACHE_DTYPE"], "fp8")
        self.assertEqual(recipe["KV_CACHE_MEMORY_BYTES"], "8589934592")
        self.assertEqual(recipe["QUANTIZATION"], "modelopt_mixed")
        self.assertEqual(recipe["LINEAR_BACKEND"], "flashinfer_b12x")
        self.assertEqual(recipe["MOE_BACKEND"], "flashinfer_cutlass")
        self.assertEqual(recipe["KDA_PREFILL_BACKEND"], "flashkda")
        self.assertEqual(recipe["BLOCK_SIZE"], "256")
        self.assertEqual(recipe["CLUSTER_RAIL"], "CX0")
        self.assertEqual(recipe["BUILD_IMAGE"], "1")
        self.assertEqual(recipe["PULL_IMAGE"], "0")
        self.assertEqual(recipe["ALLOW_UNVERIFIED_MODEL"], "0")
        self.assertEqual(recipe["RUN_SMOKE_TEST"], "1")

    def test_launcher_is_worker_first_and_tears_down_every_rank(self) -> None:
        launcher = (ROOT / "scripts/launch-glm53-flash.sh").read_text()
        worker_loop = re.search(r"for rank in ([^;]+); do\n\s+remote_node start", launcher)
        self.assertIsNotNone(worker_loop)
        self.assertEqual(worker_loop.group(1).split(), ["3", "2", "1"])
        self.assertLess(launcher.index("stop_all\n"), launcher.index("remote_node start 0"))
        self.assertIn("docker save", launcher)
        self.assertIn("smoke-test-glm53.sh", launcher)
        self.assertIn('this recipe requires TP_NODE_COUNT=4', launcher)
        self.assertIn('MODEL_MOUNT_HOST_PATH="${MODEL_MOUNT_HOST_PATH:-$MODEL_HOST_PATH}"', launcher)
        self.assertIn('MODEL_MOUNT_CONTAINER_PATH="${MODEL_MOUNT_CONTAINER_PATH:-$MODEL_CONTAINER_PATH}"', launcher)
        for action in (
            "build",
            "prepare",
            "preflight",
            "start",
            "stop",
            "status",
            "verify",
            "logs",
        ):
            self.assertIn(f"{action})", launcher)

    def test_node_launch_uses_native_tp4_and_sm121_network_guards(self) -> None:
        node = (ROOT / "scripts/glm53-node.sh").read_text()
        required = (
            "--tensor-parallel-size 4",
            "--distributed-executor-backend mp",
            "--nnodes 4",
            "NCCL_CUMEM_ENABLE=0",
            "NCCL_NVLS_ENABLE=0",
            "TORCH_CUDA_ARCH_LIST=12.1a",
            "FLASHINFER_CUDA_ARCH_LIST=12.1f",
            "--quantization",
            "--linear-backend",
            "--kda-prefill-backend",
            "--max-num-batched-tokens",
            "--kv-cache-memory-bytes",
            "--enable-chunked-prefill",
            "--enable-prefix-caching",
            "--async-scheduling",
        )
        for fragment in required:
            self.assertIn(fragment, node)
        self.assertIn('-v "$MODEL_MOUNT_HOST_PATH:$MODEL_MOUNT_CONTAINER_PATH:ro"', node)

    def test_ring_transport_settings_are_not_imported(self) -> None:
        deployment = "\n".join(
            (ROOT / path).read_text()
            for path in (
                "scripts/glm53-node.sh",
                "scripts/launch-glm53-flash.sh",
                "recipes/glm-5.3-flash-nvfp4.env",
            )
        )
        for ring_setting in (
            "NCCL_ALGO=Ring",
            "NCCL_SKIP_TREE_CONNECT",
            "SPARKRING",
            "PATCHED_NCCL_SO",
            "VLLM_NCCL_SO_PATH",
            "LD_PRELOAD",
        ):
            self.assertNotIn(ring_setting, deployment)

    def test_image_and_patches_are_pinned_and_guarded(self) -> None:
        dockerfile = (ROOT / "docker/Dockerfile.glm53-sm121").read_text()
        patch = (ROOT / "docker/patch_sm121.py").read_text()
        self.assertIn("@sha256:", dockerfile)
        self.assertIn("flashinfer-python==${FLASHINFER_VERSION}", dockerfile)
        self.assertIn("instanttensor==${INSTANTTENSOR_VERSION}", dockerfile)
        self.assertIn('count != 1', patch)
        self.assertIn("return major in (9, 10)", patch)
        self.assertIn("capability.major in (9, 12)", patch)

    def test_smoke_test_covers_finite_decode_and_tool_calling(self) -> None:
        smoke = (ROOT / "scripts/smoke-test-glm53.sh").read_text()
        self.assertIn('"logprobs":true', smoke)
        self.assertIn("math.isfinite", smoke)
        self.assertIn("EXPECTED_MAX_MODEL_LEN", smoke)
        self.assertIn('"tool_choice":"required"', smoke)
        self.assertIn('== "echo_word"', smoke)


if __name__ == "__main__":
    unittest.main()
