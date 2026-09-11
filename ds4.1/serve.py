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
    # Adaptive verification needs all short depths; match the Spark launch's
    # denser step-4 buckets beyond depth=6, with max capacity 16*(5+1)=96.
    capacity = config["max_num_seqs"] * 6
    sizes = sorted(set(list(range(1, 7)) + list(range(6, capacity + 1, 4)) + [capacity]))
    spec = {"method": "dspark", "num_speculative_tokens": 5,
            "draft_tensor_parallel_size": 4,
            "attention_backend": "B12X_MLA_SPARSE_DSV41",
            "draft_sample_method": "greedy", "rejection_sample_method": "standard",
            "enable_adaptive_verification": True}
    compile_config = {"cudagraph_mode": "FULL_DECODE_ONLY",
                      "cudagraph_capture_sizes": sizes, "custom_ops": ["all"]}
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
    print("DS41: TP4/EP4, DCP1, SSD FP4 Engram, DSpark5, breakable decode CUDA graphs", flush=True)
    os.execvp(cmd[0], cmd)
