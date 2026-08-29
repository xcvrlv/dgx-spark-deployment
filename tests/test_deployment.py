import importlib.util
import re
import tempfile
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
        self.assertEqual(recipe["MOE_BACKEND"], "marlin")
        self.assertEqual(recipe["KDA_PREFILL_BACKEND"], "flashkda")
        self.assertEqual(recipe["BLOCK_SIZE"], "256")
        self.assertEqual(recipe["CLUSTER_RAIL"], "CX0")
        self.assertEqual(recipe["BUILD_IMAGE"], "1")
        self.assertEqual(recipe["PULL_IMAGE"], "0")
        self.assertEqual(recipe["ALLOW_UNVERIFIED_MODEL"], "0")
        self.assertEqual(recipe["RUN_SMOKE_TEST"], "1")
        self.assertEqual(recipe["CONTAINER_NOFILE"], "1048576")
        self.assertEqual(recipe["ENFORCE_EAGER"], "1")
        self.assertEqual(recipe["VERIFY_CUDA_GRAPHS"], "0")
        self.assertEqual(recipe["MTP_DRAFT_SAMPLE_METHOD"], "greedy")
        self.assertEqual(recipe["MTP_MOE_BACKEND"], "marlin")

    def test_performance_profiles_stage_graphs_before_mtp(self) -> None:
        graphs = read_env(ROOT / "profiles/glm53-cudagraph.env")
        mtp_one = read_env(ROOT / "profiles/glm53-mtp-1.env")
        mtp_three = read_env(ROOT / "profiles/glm53-mtp-3.env")
        for profile in (graphs, mtp_one, mtp_three):
            self.assertEqual(profile["ENFORCE_EAGER"], "0")
            self.assertEqual(profile["VERIFY_CUDA_GRAPHS"], "1")
            self.assertIn('"cudagraph_mode":"FULL_AND_PIECEWISE"', profile["COMPILATION_CONFIG"])
            self.assertEqual(profile["MAX_NUM_BATCHED_TOKENS"], "32768")
        self.assertEqual(graphs["MTP_SPECULATIVE_TOKENS"], "0")
        self.assertEqual(mtp_one["MTP_SPECULATIVE_TOKENS"], "1")
        self.assertEqual(mtp_three["MTP_SPECULATIVE_TOKENS"], "3")
        for profile in (mtp_one, mtp_three):
            self.assertEqual(profile["ASYNC_SCHEDULING"], "1")
            self.assertEqual(profile["MTP_DRAFT_TP_SIZE"], "4")
            self.assertEqual(profile["MTP_MOE_BACKEND"], "marlin")
            self.assertEqual(profile["MTP_USE_LOCAL_ARGMAX_REDUCTION"], "1")
            self.assertEqual(profile["MTP_DRAFT_SAMPLE_METHOD"], "greedy")

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
            "benchmark",
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
            '--ulimit "nofile=$CONTAINER_NOFILE:$CONTAINER_NOFILE"',
            "--quantization",
            "--linear-backend",
            "--kda-prefill-backend",
            "--max-num-batched-tokens",
            "--kv-cache-memory-bytes",
            "--enable-chunked-prefill",
            "--enable-prefix-caching",
            "--async-scheduling",
            "--compilation-config",
            "--speculative-config",
            'moe_backend\\\":\\\"$MTP_MOE_BACKEND',
            'grep -Ei \'Capturing CUDA graph|CUDA graph capture|Graph capturing finished\'',
        )
        for fragment in required:
            self.assertIn(fragment, node)
        self.assertIn('-v "$MODEL_MOUNT_HOST_PATH:$MODEL_MOUNT_CONTAINER_PATH:ro"', node)
        self.assertIn('model is not readable in the container at $MODEL_CONTAINER_PATH', node)
        self.assertIn(
            'docker run --rm --gpus all --entrypoint vllm "$IMAGE" serve --help=all',
            node,
        )
        self.assertIn('"SM121 persistent TopK guard is missing"', node)

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
            "VLLM_SPARK_TP4_MODE=custom",
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
        self.assertIn("multi_processor_count >= 78", patch)
        self.assertIn("def guard_persistent_topk", patch)
        self.assertIn("MTP draft MoE backend propagation", patch)
        self.assertIn("moe_backend=speculative_config.moe_backend", patch)
        self.assertIn("ModelOpt MXFP8 MTP prefix alias", patch)
        self.assertIn('if ".mtp_block." in prefix', patch)
        self.assertIn('"language_model.model." + suffix', patch)
        recipe = (ROOT / "recipes/glm-5.3-flash-nvfp4.env").read_text()
        self.assertIn("IMAGE=spark-vllm-glm53:sm121-v5", recipe)
        node = (ROOT / "scripts/glm53-node.sh").read_text()
        self.assertIn('resolved=q._resolve_quant_algo(p)', node)
        self.assertIn('assert resolved == "MXFP8"', node)

    def test_topk_guard_finds_the_call_instead_of_matching_branch_text(self) -> None:
        patch_source = (ROOT / "docker/patch_sm121.py").read_text()
        helpers = patch_source.split("\nvllm = package_root", maxsplit=1)[0]
        namespace: dict[str, object] = {}
        exec(helpers, namespace)
        fixture = """\
def decode(logits):
    use_fast_topk = select_k in (512, 1024, 2048)
    if use_fast_topk:
        torch.ops._C.persistent_topk(logits)
    else:
        fallback(logits)
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "indexer.py"
            path.write_text(fixture)
            namespace["guard_persistent_topk"](path)
            patched = path.read_text()
        self.assertIn("(use_fast_topk) and torch.cuda.get_device_properties", patched)
        self.assertIn("multi_processor_count >= 78", patched)
        compile(patched, "indexer.py", "exec")

    def test_smoke_test_covers_finite_decode_and_tool_calling(self) -> None:
        smoke = (ROOT / "scripts/smoke-test-glm53.sh").read_text()
        self.assertIn('"logprobs":true', smoke)
        self.assertIn("math.isfinite", smoke)
        self.assertIn("EXPECTED_MAX_MODEL_LEN", smoke)
        self.assertIn('"tool_choice":"required"', smoke)
        self.assertIn('== "echo_word"', smoke)

    def test_benchmark_uses_uncached_prefill_and_streaming_decode_math(self) -> None:
        benchmark_path = ROOT / "scripts/benchmark-glm53.py"
        source = benchmark_path.read_text()
        self.assertIn('"stream_options": {"include_usage": True}', source)
        self.assertIn("(completion_tokens - 1) / decode_seconds", source)
        self.assertIn("uuid.uuid4()", source)

        spec = importlib.util.spec_from_file_location(
            "benchmark_glm53", benchmark_path
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        prompt_a = module.make_prefill_prompt(32)
        prompt_b = module.make_prefill_prompt(32)
        self.assertNotEqual(prompt_a, prompt_b)
        self.assertGreaterEqual(len(prompt_a.split()), 32)
        metrics = module.metric_delta(
            {
                "vllm:spec_decode_num_accepted_tokens_total": 10.0,
                "vllm:spec_decode_num_draft_tokens_total": 20.0,
            },
            {
                "vllm:spec_decode_num_accepted_tokens_total": 40.0,
                "vllm:spec_decode_num_draft_tokens_total": 70.0,
            },
        )
        self.assertEqual(metrics["calculated_draft_acceptance_rate"], 0.6)

    def test_full_glm52_exl3_recipe_pins_r34_tp4_contract(self) -> None:
        recipe = read_env(ROOT / "recipes/glm-5.2-exl3-r7.env")
        self.assertEqual(
            recipe["MODEL_ID"],
            "brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78",
        )
        self.assertEqual(
            recipe["MODEL_REVISION"],
            "9ab9579774cc432df91567a36f6e9e863e0d4c9f",
        )
        self.assertEqual(recipe["QUANTIZATION"], "exl3")
        self.assertEqual(recipe["ATTENTION_BACKEND"], "B12X_MLA_SPARSE")
        self.assertEqual(recipe["MOE_BACKEND"], "b12x")
        self.assertEqual(recipe["ONLINE_QUANT"], "exl3-b6")
        self.assertIn('"shared_experts"', recipe["ONLINE_QUANT_CONFIG"])
        self.assertEqual(recipe["KV_CACHE_DTYPE"], "nvfp4_ds_mla")
        self.assertEqual(recipe["LOAD_FORMAT"], "instanttensor")
        self.assertEqual(recipe["INSTANTTENSOR_COPY"], "0")
        self.assertEqual(recipe["B12X_PCIE_DMA"], "0")
        self.assertEqual(recipe["DISABLE_CUSTOM_ALL_REDUCE"], "1")
        self.assertEqual(recipe["MTP_SPECULATIVE_TOKENS"], "0")
        self.assertEqual(recipe["ENFORCE_EAGER"], "1")
        self.assertEqual(recipe["MAX_MODEL_LEN"], "65536")

    def test_exl3_arm64_image_reconstructs_immutable_upstream_trees(self) -> None:
        dockerfile = (ROOT / "docker/Dockerfile.glm52-exl3-sm121").read_text()
        required = (
            "glm53-flash-arm64-cu130@sha256:",
            "7302862b8fcfdc7c06a411a61e1f0fb072258880",
            "e2666d9a65f41fc376607531453cbd57c4c71016",
            "b0f8c85c7b96497e0148a18230f43d18854ae04a",
            "7cecbb2c4819636ae7f05f8b116f2c45ee2cff7b",
            "cd3ce190f0f1917402cdfd5773724267cc9a63f8",
            "704aefd743b390af4bd0fb429d1906f9b964c7d8",
            "49b4010afc1cae0441e71fe0b0bffc24fa05e932",
            "git -C /opt/vllm-r34 apply --index",
            "git -C /opt/b12x-r34 apply --index",
            "TORCH_CUDA_ARCH_LIST=12.1a",
            "CUTE_DSL_ARCH=sm_121a",
            "FLASHINFER_CUDA_ARCH_LIST=12.1f",
            "python3 setup.py build_ext --inplace",
            "VLLM_EXL3_EXT_PATH=/opt/exllamav3",
            'local-inference.vllm.integration.tree="b0f8c85',
            'local-inference.b12x.integration.tree="cd3ce19',
        )
        for fragment in required:
            self.assertIn(fragment, dockerfile)

    def test_exl3_node_uses_roce_tp4_without_single_host_pcie_collectives(self) -> None:
        node = (ROOT / "scripts/glm52-exl3-node.sh").read_text()
        required = (
            "--tensor-parallel-size 4",
            '--decode-context-parallel-size "$DECODE_CONTEXT_PARALLEL_SIZE"',
            "--distributed-executor-backend mp",
            "--nnodes 4",
            '--node-rank "$rank"',
            '--master-addr "$head_ip"',
            "--attention-backend",
            "--quantization-config",
            "--disable-custom-all-reduce",
            "VLLM_USE_B12X_PCIE_DMA=0",
            "VLLM_ENABLE_PCIE_ALLREDUCE=0",
            "B12X_PCIE_DMA_FP8=0",
            "NCCL_NET=IB",
            "CUTE_DSL_ARCH=sm_121a",
            "VLLM_EXL3_ONLINE_CACHE_DIR=/cache/exl3-online",
            "VLLM_EXL3_EXT_PATH=/opt/exllamav3",
            'warning: config has no recognized EXL marker',
            'tail.get("tp") not in (None, 4)',
            '"local-inference.vllm.integration.tree"',
            '"local-inference.b12x.integration.tree"',
        )
        for fragment in required:
            self.assertIn(fragment, node)

    def test_exl3_profiles_stage_graphs_before_mtp3(self) -> None:
        graphs = read_env(ROOT / "profiles/glm52-exl3-cudagraph.env")
        mtp_one = read_env(ROOT / "profiles/glm52-exl3-mtp-1.env")
        mtp_three = read_env(ROOT / "profiles/glm52-exl3-mtp-3.env")
        self.assertEqual(graphs["MTP_SPECULATIVE_TOKENS"], "0")
        self.assertEqual(mtp_one["MTP_SPECULATIVE_TOKENS"], "1")
        self.assertEqual(mtp_three["MTP_SPECULATIVE_TOKENS"], "3")
        for profile in (graphs, mtp_one, mtp_three):
            self.assertEqual(profile["ENFORCE_EAGER"], "0")
            self.assertEqual(profile["VERIFY_CUDA_GRAPHS"], "1")
            self.assertIn('"cudagraph_mode":"FULL_AND_PIECEWISE"', profile["COMPILATION_CONFIG"])
            self.assertIn('"fuse_allreduce_rms":false', profile["COMPILATION_CONFIG"])
        for profile in (mtp_one, mtp_three):
            self.assertEqual(profile["MTP_USE_LOCAL_ARGMAX_REDUCTION"], "1")
            self.assertEqual(profile["ASYNC_SCHEDULING"], "0")

    def test_exl3_wrapper_reuses_guarded_fleet_orchestration(self) -> None:
        wrapper = (ROOT / "scripts/launch-glm52-exl3.sh").read_text()
        self.assertIn("recipes/glm-5.2-exl3-r7.env", wrapper)
        self.assertIn("scripts/glm52-exl3-node.sh", wrapper)
        self.assertIn("Dockerfile.glm52-exl3-sm121", wrapper)
        self.assertIn('exec bash "$root_dir/scripts/launch-glm53-flash.sh"', wrapper)

    def test_sparkring_switch_recipe_matches_reference_serving_contract(self) -> None:
        recipe = read_env(ROOT / "recipes/glm-5.2-exl3-sparkring-switch.env")
        expected = {
            "SPARKRING_UPSTREAM_COMMIT": "510556275ed3b77fc56a14367d319417072eeb8c",
            "MODEL_PROVISIONING": "local",
            "PREPARE_IMAGE": "0",
            "MODEL_ID": "davidsyoung/GLM-5.3-EXL3-TR3-3.42bpw",
            "Q40_CHECKPOINT_REVISION": "fa6cf0511c0c0d0477874ee2c8570652e1d63f66",
            "MODEL_HOST_PATH": "/home/juho/.cache/huggingface/hub/models--davidsyoung--GLM-5.3-EXL3-TR3-3.42bpw",
            "MODEL_CONTAINER_PATH": "auto",
            "ALLOW_UNVERIFIED_MODEL": "1",
            "DECODE_CONTEXT_PARALLEL_SIZE": "4",
            "DCP_COMM_BACKEND": "ag_rs",
            "DCP_KV_CACHE_INTERLEAVE_SIZE": "1",
            "MTP_SPECULATIVE_TOKENS": "4",
            "MTP_DRAFT_TP_SIZE": "4",
            "MTP_MOE_BACKEND": "b12x",
            "MAX_MODEL_LEN": "1048576",
            "MAX_NUM_SEQS": "16",
            "MAX_NUM_BATCHED_TOKENS": "4096",
            "BLOCK_SIZE": "64",
            "KV_CACHE_DTYPE": "nvfp4_ds_mla",
            "KV_CACHE_MEMORY_BYTES": "9250000000",
            "KV_FP8_ROPE": "1",
            "VLLM_NVFP4_MLA_DYNAMIC_SCALE": "1",
            "VLLM_USE_B12X_DCP_A2A": "1",
            "VLLM_B12X_MLA_CKV_GATHER": "1",
            "VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS": "1048576",
            "VLLM_EXL3_PREFILL_CAPACITY": "4096",
            "VLLM_SPARK_MAX_QUERY_ROWS": "40",
            "MAX_CUDAGRAPH_CAPTURE_SIZE": "40",
            "Q40_ENABLED": "1",
            "NETWORK_TOPOLOGY": "switch-star-dual-rail",
            "NCCL_CROSS_NIC": "1",
        }
        for key, value in expected.items():
            self.assertEqual(recipe[key], value, key)
        self.assertIn('"cudagraph_capture_sizes":[1,2,3,4,5', recipe["COMPILATION_CONFIG"])
        self.assertIn('38,39,40]', recipe["COMPILATION_CONFIG"])
        self.assertEqual(recipe["NCCL_ALGO"], "")
        self.assertEqual(recipe["NCCL_IB_HCA"], "")
        self.assertEqual(recipe["MODEL_CONFIG_SHA256"], "")
        self.assertEqual(recipe["MODEL_INDEX_SHA256"], "")
        self.assertEqual(recipe["MODEL_SHARD_COUNT"], "")
        self.assertEqual(recipe["MODEL_INDEX_TOTAL_SIZE"], "")
        self.assertNotIn("NCCL_SKIP_TREE_CONNECT", recipe)

    def test_sparkring_switch_launcher_uses_upstream_runtime_without_sircl(self) -> None:
        builder = (ROOT / "scripts/build-glm52-sparkring-runtime.sh").read_text()
        node = (ROOT / "scripts/glm52-exl3-node.sh").read_text()
        launcher = (ROOT / "scripts/launch-glm53-flash.sh").read_text()
        wrapper = (ROOT / "scripts/launch-glm52-sparkring-switch.sh").read_text()
        for fragment in (
            "runtime/exl3-r7/build-image.sh",
            "prepare_q40_overlay_inputs.py",
            "q40_exact_state_overlay.py",
            "q40_exact_state_attestation_overlay.py",
            "--image-id \"$image_id\"",
            '--checkpoint-revision "$q40_checkpoint_revision"',
        ):
            self.assertIn(fragment, builder)
        for fragment in (
            'if [[ "$MODEL_PROVISIONING" == "local" ]]',
            "local model ready at $effective_model_host_path (offline; no files downloaded)",
            'effective_model_container_path="$MODEL_MOUNT_CONTAINER_PATH/snapshots/$revision"',
            'IFS=\',\' read -r -a fabric_ips <<<"$FABRIC_IPS"',
            'hca="$(resolve_hca_for_iface "$iface")"',
            "dual-rail discovery expected two distinct RDMA HCAs",
            '--dcp-comm-backend "$DCP_COMM_BACKEND"',
            '--dcp-kv-cache-interleave-size "$DCP_KV_CACHE_INTERLEAVE_SIZE"',
            '--block-size "$BLOCK_SIZE"',
            '--kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES"',
            "VLLM_NVFP4_MLA_DYNAMIC_SCALE",
            "VLLM_B12X_MLA_CKV_GATHER",
            "SPARK_Q40_EXACT_STATE_ATTEST_PATH",
            "SPARK_Q40_EXACT_STATE_CHECKPOINT=$Q40_CHECKPOINT_REVISION",
            'NCCL_CROSS_NIC=$NCCL_CROSS_NIC',
        ):
            self.assertIn(fragment, node)
        self.assertNotIn("VLLM_SPARK_TP4_MODE=custom", node)
        self.assertNotIn("NCCL_SKIP_TREE_CONNECT", node)
        self.assertNotIn("download_exl3_r7.py", builder)
        self.assertIn('if [[ "$PREPARE_IMAGE" == "1" ]]', launcher)
        self.assertIn('"FABRIC_IFACE=${FABRIC_IFACE:-}" "FABRIC_IPS=$fabric_ips"', launcher)
        self.assertIn("PREPARE_IMAGE=0: using the existing runtime image and local model files", launcher)
        self.assertIn("glm-5.2-exl3-sparkring-switch.env", wrapper)


if __name__ == "__main__":
    unittest.main()
