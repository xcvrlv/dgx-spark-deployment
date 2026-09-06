#!/usr/bin/env python3
"""v14 exact numerical, graph-replay and dependency gates. Timing is diagnostic."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def load_extension():
    # Recipe PYTHONPATH is absent in a plain `docker run --entrypoint python3`.
    # Use the serving loader, which also honors an explicitly configured ABI shim.
    os.environ.setdefault("VLLM_EXL3_EXT_PATH", "/opt/exllamav3")
    from vllm.model_executor.layers.quantization.exl3 import _load_exl3_ext
    ext = _load_exl3_ext()
    assert callable(ext.had_r_128)
    return ext


def ms(fn):
    import torch
    for _ in range(3):
        fn()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(20):
        fn()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) / 20


def argmax_gpu():
    import torch
    from vllm.model_executor.layers import gb10_argmax as impl
    torch.manual_seed(5314)
    cases, timings = 0, []
    for rows in (1, 4, 8):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            # Simulate TP4 rank order, including padding and non-power-of-two vocab.
            width, valid_last = 38400, 38373
            shards = [torch.randn(rows, width, device="cuda", dtype=dtype) for _ in range(4)]
            shards[-1][:, valid_last:] = 100  # Padding must not win.
            def candidate():
                pairs = [impl.local_pair(x, valid_last if i == 3 else width, i * width)
                         for i, x in enumerate(shards)]
                return impl.global_tokens(torch.cat(pairs, dim=-1), 4)
            def reference():
                return torch.cat([*shards[:3], shards[3][:, :valid_last]], dim=-1).argmax(-1)
            for case in ("random", "ties", "nan", "negative_inf"):
                if case == "ties":
                    for x in shards:
                        x[:, 11] = 200
                        x[:, 1031] = 200
                elif case == "nan":
                    shards[0][:, 1030] = float("nan")
                    shards[1][:, 5] = float("nan")
                elif case == "negative_inf":
                    for x in shards:
                        x.fill_(-float("inf"))
                torch.testing.assert_close(candidate(), reference(), rtol=0, atol=0)
                cases += 1
            # Capture reduction stages, then mutate their inputs on every replay.
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                candidate()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = candidate()
            for _ in range(3):
                for x in shards:
                    x.normal_()
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(output, reference(), rtol=0, atol=0)
            x = shards[0]
            timings.append(dict(rows=rows, dtype=str(dtype),
                torch_local_max_ms=ms(lambda: x.max(dim=-1)),
                split_local_pair_ms=ms(lambda: impl.local_pair(x, width, 0))))
    return dict(argmax_cases=cases, argmax_timings=timings,
                argmax_collective="TP4 simulated locally; real cluster collective not exercised")


def rotations_gpu():
    import torch
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from b12x._lib.utils import current_cuda_stream
    from probe_r22_v14_rotations import RotationProbe
    torch.manual_seed(5414)
    timings = []
    for rows in (65, 129, 512, 4096):
        h, topk = 6144, 8
        for dtype in (torch.float16, torch.bfloat16):
            x = torch.randn(rows * h, device="cuda", dtype=dtype) * .25
            sg, su = [torch.randn(h, device="cuda", dtype=torch.float16) * .5 for _ in range(2)]
            routes = torch.arange(rows * topk, device="cuda", dtype=torch.int32)
            routes[::13] = rows * topk  # Packed padding sentinel.
            experts = torch.zeros(rows, device="cuda", dtype=torch.int32)
            experts[::7] = -1  # Non-local/missing routes must remain unconsumed.
            experts[1::7] = 1
            count = torch.tensor([routes.numel()], device="cuda", dtype=torch.int32)
            mapping = torch.arange(2, device="cuda", dtype=torch.int32)
            outputs = [torch.empty(rows * topk * h, device="cuda", dtype=torch.float16) for _ in range(4)]
            live = routes[(routes < rows * topk) & (experts.repeat_interleave(8) >= 0)].long()
            runs = []
            for shared, (g, u) in zip((False, True), (outputs[:2], outputs[2:])):
                args = [from_dlpack(t, assumed_align=16) for t in
                        (x, g, u, sg, su, routes, experts, count, mapping)]
                compiled = cute.compile(RotationProbe(rows, h, topk, shared), *args, current_cuda_stream())
                def run(fn=compiled, args=args):
                    fn(*args, current_cuda_stream())
                runs.append(run)
            def compare():
                for before, after in zip(outputs[:2], outputs[2:]):
                    torch.testing.assert_close(before.view(-1, h)[live], after.view(-1, h)[live], rtol=0, atol=0)
            for run in runs:
                run()
            torch.cuda.synchronize()
            compare()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                runs[1]()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                runs[1]()
            for _ in range(3):
                x.normal_(0, .25)
                runs[0]()
                graph.replay()
                torch.cuda.synchronize()
                compare()
            timings.append(dict(rows=rows, dtype=str(dtype), legacy_ms=ms(runs[0]), shared_ms=ms(runs[1])))
    return dict(shared_rotation_timings=timings)


def mixed_gpu():
    """Exercise the real two/three-tier cooperative grid and its phase barriers."""
    import torch
    from b12x.moe._shared.kernels.w4a16 import mixed_trellis as api
    from b12x.moe._shared.kernels.w4a16.host import max_packed_route_slots
    from b12x.moe._shared.kernels.w4a16.prepare import prepare_trellis256_moe_weights
    device = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    original = os.environ.get("VLLM_GB10_SHARED_INPUT_ROTATION")
    results = []
    try:
        for counts in ((4, 4), (3, 3, 2)):
            rows, hidden, intermediate, topk, block = 128, 6144, 512, 8, 32
            tiles = (128, 128, 32, 512)
            shared = [torch.ones(1, hidden, device=device, dtype=torch.float16) for _ in range(3)]
            # Distinct gate/up scales ensure the optimized branch does not
            # accidentally treat the two projections as identical.
            shared[1].mul_(.75)
            tiers = [prepare_trellis256_moe_weights(
                hidden_size=hidden, intermediate_size=intermediate, num_experts=count,
                activation="silu", fc1_tile_n=tiles[1], fc2_tile_n=tiles[3], device=device,
                seed=5314 + i, params_dtype=torch.float16, w13_layout="trellis_t256_proj",
                trellis_bits=3 + i, codebook="mcg", gate_suh=shared[0], up_suh=shared[1],
                down_svh=shared[2], intermediate_rotations=torch.ones(count, 3 * intermediate,
                    device=device, dtype=torch.float16), tile_config=tiles)
                for i, count in enumerate(counts)]
            rotations = api.MixedTrellisRotations(
                intermediate=torch.cat([t.intermediate_rotations for t in tiers]),
                gate_suh=shared[0], up_suh=shared[1], down_svh=shared[2])
            projection = tuple(i for i, count in enumerate(counts) for _ in range(count))
            maps = api.build_projection_tiered_maps(projection, projection, projection,
                                                    tier_slots=counts, device=device)
            x = torch.randn(rows, hidden, device=device, dtype=torch.bfloat16) * 1e-3
            ids = torch.arange(topk, device=device, dtype=torch.int32).expand(rows, -1).contiguous()
            weights = torch.rand(rows, topk, device=device)
            weights /= weights.sum(-1, keepdim=True)
            suffix = "3" if len(counts) == 3 else ""
            compile_fn = getattr(api, "compile_mixed_trellis" + suffix)
            buffers_fn = getattr(api, "make_mixed_trellis" + suffix + "_buffers")
            bind_fn = getattr(api, "bind_mixed_trellis" + suffix)
            run_fn = getattr(api, "run_bound_mixed_trellis" + suffix)
            runs, compiled = [], []
            for enabled in (False, True):
                os.environ["VLLM_GB10_SHARED_INPUT_ROTATION"] = str(int(enabled))
                launch = compile_fn(size_m=rows, hidden_size=hidden, intermediate_size=intermediate,
                    **{f"tier{i}_num_experts": count for i, count in enumerate(counts)},
                    top_k=topk, max_m_blocks=(max_packed_route_slots(rows * topk, block, sum(counts)) + block - 1) // block,
                    sms=props.multi_processor_count, max_shared_mem=props.shared_memory_per_block_optin,
                    force_tile_config=tiles, moe_block_size=block, broadcast_suh=True, broadcast_svh=True)
                buffers = buffers_fn(launch, device=device, sms=props.multi_processor_count)
                binding = bind_fn(*tiers, *maps, rotations, launch)
                compiled.append(launch.compiled)
                runs.append(lambda binding=binding, buffers=buffers: run_fn(x, weights, ids, binding, buffers))
            assert compiled[0] is not compiled[1], "shared-input variants reused a stale compiled kernel"
            expected, actual = runs[0]().clone(), runs[1]()
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                runs[1]()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured = runs[1]()
            for _ in range(3):
                x.normal_(0, 1e-3)
                ids.copy_(torch.rand(rows, topk, device=device).argsort(-1).int())
                expected = runs[0]().clone()
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(captured, expected, rtol=0, atol=0)
            results.append(dict(tiers=len(counts), legacy_ms=ms(runs[0]), shared_ms=ms(runs[1])))
    finally:
        if original is None:
            os.environ.pop("VLLM_GB10_SHARED_INPUT_ROTATION", None)
        else:
            os.environ["VLLM_GB10_SHARED_INPUT_ROTATION"] = original
    return dict(mixed_k_timings=results)


def distributed_gpu():
    """Optional torchrun gate for the actual TP4/RoCEnante packet layout."""
    import torch
    import torch.distributed as dist
    from types import SimpleNamespace as NS
    from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
    from vllm.model_executor.layers import logits_processor as lp, gb10_argmax as impl
    os.environ["VLLM_ENABLE_ROCE_ALLREDUCE"] = "1"
    torch.cuda.set_device(0)
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 4, "requires four one-GPU hosts"
    cpu_group = dist.new_group(backend="gloo")
    comm = CudaCommunicator(cpu_group, torch.device("cuda", 0), dist.group.WORLD,
                            unique_name="tp:v14-mtp-smoke")
    assert comm.use_roce_allreduce and not comm.b12x_ar_comm.disabled
    original_gather, original_enabled = lp.tensor_model_parallel_all_gather, impl.ENABLED
    try:
        lp.tensor_model_parallel_all_gather = lambda tensor, dim=-1: comm.all_gather(tensor, dim=dim)
        width = 38400
        head = NS(tp_size=world, shard_indices=NS(num_org_vocab_padding=0,
                      org_vocab_start_index=rank*width, org_vocab_end_index=(rank+1)*width))
        # Drive the installed LogitsProcessor method, including its padding,
        # fast/stock dispatch, real collective and final token selection.
        processor = NS(scale=1.0, soft_cap=None, _apply_head=lambda head, x, bias: x)
        for rows in (1, 4, 8):
            x = torch.zeros(rows, width, device="cuda", dtype=torch.float32)
            for fast in (False, True):
                impl.ENABLED = fast
                def run():
                    return lp.LogitsProcessor.get_top_tokens(processor, head, x)
                x.zero_()
                x[:, 11] = rank + 1
                for _ in range(3):
                    eager = run()
                torch.testing.assert_close(eager, torch.full_like(eager, 3*width+11), rtol=0, atol=0)
                graph = torch.cuda.CUDAGraph()
                with comm.b12x_ar_comm.capture():
                    with torch.cuda.graph(graph):
                        output = run()
                for winner in (2, 0, 1):
                    x.zero_()
                    x[:, 1031] = 100 if rank == winner else 1
                    graph.replay()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(output, torch.full_like(output, winner*width+1031), rtol=0, atol=0)
                comm.b12x_ar_comm.check_health()
        # Aligned argmax packets must never request the padded-gather arena.
        assert comm.b12x_ar_comm._runtime._gather_buffers is None
        dist.barrier()
    finally:
        lp.tensor_model_parallel_all_gather, impl.ENABLED = original_gather, original_enabled
        comm.destroy()
        dist.destroy_process_group()
    return dict(mtp_tp4_roce_graph_replay="passed", rank=rank)


def main():
    import b12x
    import vllm
    from patch_r22_v14 import patch, VERSION
    from patch_r22_v13 import OUTPUTS as v13, KERNEL as changed_kernel
    from patch_r22_v11 import OUTPUT_HASHES as v11
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--distributed", action="store_true")
    args = parser.parse_args()
    b, v = Path(b12x.__file__).parent, Path(vllm.__file__).parent
    patch(b, v, check=True)
    # Preserve inherited contracts without asking v13's verifier to accept a
    # v14 kernel hash. Its numerical tests are still run below.
    for hashes in (v11, v13):
        for name, expected in hashes.items():
            if name == changed_kernel:
                continue
            root = v if name.startswith("model_executor/") else b
            source = (root / name).read_text(encoding="utf-8")
            assert hashlib.sha256(source.encode()).hexdigest() == expected, name
    ext = load_extension()
    result = dict(overlay=VERSION, exl3_extension=ext.__file__)
    if args.gpu:
        import torch
        assert torch.cuda.get_device_capability() == (12, 1)
        from smoke_r22_v13 import gpu as v13_gpu
        result.update(inherited_v13=v13_gpu())
        result.update(argmax_gpu())
        result.update(rotations_gpu())
        result.update(mixed_gpu())
    if args.distributed:
        result.update(distributed_gpu())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
