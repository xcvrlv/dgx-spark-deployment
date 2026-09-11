#!/usr/bin/env python3
"""Build the single shared V4.1 command; executed inside each node's container."""
import argparse
import json
import os


def command(config, rank):
    if len(config["nodes"]) != 4 or config["decode_context_parallel"] != 1:
        raise ValueError("This initial deployment requires four nodes and DCP1")
    if config["num_speculative_tokens"] != 5:
        raise ValueError("Use the checkpoint's qualified five-token DSpark setup")
    # Fixed five-token DSpark: six verification rows per request. Adaptive
    # verification overrides FULL_DECODE_ONLY to FULL_AND_PIECEWISE upstream.
    capacity = config["max_num_seqs"] * 6
    sizes = list(range(6, capacity + 1, 6))
    graph_mode = config.get("graph_mode", "FULL_DECODE_ONLY")
    if graph_mode not in ("FULL_DECODE_ONLY", "FULL_AND_PIECEWISE"):
        raise ValueError("Unsupported graph mode")
    if graph_mode == "FULL_AND_PIECEWISE":
        # Include exact draft and verification batch widths plus prefill buckets.
        sizes = sorted(set(sizes + list(range(5, config["max_num_seqs"] * 5 + 1, 5))
                           + config.get("prefill_graph_sizes", [128,256,512,1024,2048,4096])))
        if sizes[-1] > config["max_num_batched_tokens"]:
            raise ValueError("Prefill graph capacity exceeds scheduler budget")
    spec = {"method": "dspark", "num_speculative_tokens": 5,
            "draft_tensor_parallel_size": 4,
            "attention_backend": "B12X_MLA_SPARSE_DSV41",
            "draft_sample_method": "greedy", "rejection_sample_method": "standard",
            "enable_adaptive_verification": False}
    compile_config = {"mode": 0, "cudagraph_mode": graph_mode,
                      "cudagraph_capture_sizes": sizes, "custom_ops": ["all"],
                      "max_cudagraph_capture_size": sizes[-1],
                      "pass_config": {"fuse_allreduce_rms": False}}
    result = ["python3", "-m", "vllm.entrypoints.cli.main", "serve", "/model",
              "--served-model-name", config["served_model_name"], "--host", "0.0.0.0",
              "--port", str(config["port"]), "--dtype", "bfloat16",
              "--tensor-parallel-size", "4", "--pipeline-parallel-size", "1",
              "--enable-expert-parallel", "--decode-context-parallel-size", "1",
              "--distributed-executor-backend", "mp", "--nnodes", "4",
              "--node-rank", str(rank), "--master-addr", config["nodes"][0]["cx0"],
              "--master-port", str(config["master_port"]),
              "--load-format", "safetensors", "--safetensors-load-strategy", "lazy",
              "--block-size", "256", "--enable-prefix-caching", "--enable-chunked-prefill",
              "--gpu-memory-utilization", str(config["gpu_memory_utilization"]),
              "--max-model-len", str(config["max_model_len"]),
              "--max-num-seqs", str(config["max_num_seqs"]),
              "--max-num-batched-tokens", str(config["max_num_batched_tokens"]),
              "--generation-config", "vllm", "--limit-mm-per-prompt", '{"image":2}',
              "--engram-config", '{"cpu_offload":false,"table_memory":"disk"}',
              "--linear-backend", "b12x", "--moe-backend", "b12x",
              "--attention-backend", "B12X_MLA_SPARSE_DSV41",
              "--speculative-config", json.dumps(spec),
              "--compilation-config", json.dumps(compile_config),
              "--jit-monitor-mode", "error", "--reasoning-parser", "deepseek_v41",
              "--tool-call-parser", "deepseek_v41", "--enable-auto-tool-choice"]
    if config.get("kv_cache_memory_bytes") is not None:
        if config["kv_cache_memory_bytes"] <= 0:
            raise ValueError("KV cache budget must be positive")
        result += ["--kv-cache-memory-bytes", str(config["kv_cache_memory_bytes"])]
    if config.get("profile_trace"):
        result += ["--profiler-config",json.dumps({"profiler":"torch",
            "torch_profiler_dir":"/cache/ds41-traces", "ignore_frontend":True,
            "torch_profiler_with_stack":False, "torch_profiler_record_shapes":True,
            "torch_profiler_with_memory":False, "torch_profiler_use_gzip":True,
            "max_iterations":4})]
    if rank:
        result.append("--headless")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/opt/ds41/cluster.json")
    parser.add_argument("--rank", type=int, required=True, choices=range(4))
    args = parser.parse_args()
    with open(args.config) as stream:
        config = json.load(stream)
    config = json.loads(os.environ.get("DS41_CONFIG_JSON", json.dumps(config)))
    cmd = command(config, args.rank)
    print(f"DS41: TP4/EP4, DCP1, SSD FP4 Engram, fixed DSpark5, {config.get('graph_mode', 'FULL_DECODE_ONLY')} CUDA graphs", flush=True)
    os.execvp(cmd[0], cmd)
