#!/usr/bin/env python3
"""v15 inherited exact checks plus activation, skinny-GEMM and proxy checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def sigmoid_gpu():
    import torch
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from b12x._lib.utils import current_cuda_stream
    from probe_r22_v15_activation import ActivationProbe
    from smoke_r22_v14 import ms
    x = torch.cat((torch.linspace(-105, 100, 65536, device="cuda"),
                   torch.tensor([float('inf'), -float('inf'), float('nan')], device="cuda")))
    results = [torch.empty_like(x), torch.empty_like(x)]
    runs = []
    for fast, out in zip((False, True), results):
        args = [from_dlpack(t, assumed_align=16) for t in (x, out)]
        compiled = cute.compile(ActivationProbe(x.numel(), fast), *args, current_cuda_stream())
        runs.append(lambda fn=compiled, args=args: fn(*args, current_cuda_stream()))
    for run in runs:
        run()
    torch.cuda.synchronize()
    torch.testing.assert_close(results[1], results[0], rtol=3e-7, atol=1e-7, equal_nan=True)
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.graph(graph, stream=stream):
        runs[1]()
    for _ in range(3):
        x.uniform_(-100, 100)
        runs[0]()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(results[1], results[0], rtol=3e-7, atol=1e-7)
    return dict(sigmoid_legacy_ms=ms(runs[0]), sigmoid_v15_ms=ms(runs[1]))


def skinny_gpu():
    import torch
    from vllm.models.deepseek_v32.nvidia import glm52_low_latency_gemm as impl
    from smoke_r22_v14 import ms
    torch.manual_seed(5315)
    results = []
    for n, k in ((2624, 6144), (2048, 2048), (6144, 12288)):
        w = torch.randn(n, k, device='cuda', dtype=torch.bfloat16) * .1
        plan = impl.build_glm52_plan(w, torch.bfloat16)
        assert plan is not None and set(plan) == {1, 2}
        assert all(backend == 'cute' for backend, _ in plan.values())
        for m in (1, 2):
            x = torch.randn(m, k, device='cuda', dtype=torch.bfloat16) * .1
            def run():
                return impl.run_glm52_plan(plan, x, w)
            def reference():
                return (x.float() @ w.float().t()).to(x.dtype)
            def compare(out):
                expected = reference()
                torch.testing.assert_close(out, expected, rtol=.02, atol=.02)
                relative = (out.float()-expected.float()).norm()/expected.float().norm().clamp_min(1e-12)
                assert relative.item() < .003, relative.item()
            compare(run())
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                run()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = run()
            for _ in range(3):
                x.normal_(0, .1)
                graph.replay()
                torch.cuda.synchronize()
                compare(output)
            results.append(dict(m=m, n=n, k=k, cublas_ms=ms(lambda: x@w.t()), skinny_ms=ms(run)))
        assert impl.run_glm52_plan(plan, torch.empty(4,k,device='cuda',dtype=w.dtype), w) is None
    return dict(skinny_timings=results)


def main():
    import b12x
    import vllm
    from patch_r22_v15 import patch, VERSION, KERNEL, PROXY
    from patch_r22_v14 import OUTPUTS as v14
    from patch_r22_v13 import OUTPUTS as v13
    from patch_r22_v11 import OUTPUT_HASHES as v11
    from smoke_r22_v14 import load_extension
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--distributed', action='store_true')
    args = parser.parse_args()
    b, v = Path(b12x.__file__).parent, Path(vllm.__file__).parent
    patch(b, v, check=True)
    for hashes in (v11, v13, v14):
        for name, expected in hashes.items():
            if name in (KERNEL, PROXY):
                continue
            root = v if name.startswith('model_executor/') else b
            assert hashlib.sha256((root/name).read_text(encoding='utf-8').encode()).hexdigest() == expected, name
    ext = load_extension()
    from b12x.comm.roce._proxy import load
    library = load()
    result = dict(overlay=VERSION, proxy=str(library._name), exl3_extension=ext.__file__)
    if args.gpu:
        import torch
        assert torch.cuda.get_device_capability() == (12, 1)
        from smoke_r22_v13 import gpu as v13_gpu
        from smoke_r22_v14 import argmax_gpu, rotations_gpu, mixed_gpu
        from probe_r22_v14_rotations import RotationProbe
        # The inherited standalone probe intentionally exercises the original
        # route-sized buffer contract. Full mixed_gpu exercises compact FC1.
        RotationProbe.gb10_compact_input = False
        result.update(inherited_v13=v13_gpu())
        result.update(argmax_gpu())
        result.update(rotations_gpu())
        result.update(mixed_gpu())
        result.update(sigmoid_gpu())
        result.update(skinny_gpu())
        result.update(mixed_activation_gpu())
    if args.distributed:
        from smoke_r22_v14 import distributed_gpu
        result.update(distributed_gpu())
        from smoke_r22_performance import check_distributed_roce
        result.update(check_distributed_roce())
    print(json.dumps(result, sort_keys=True))


def mixed_activation_gpu():
    """Exercise the real two/three-tier cooperative grid and its phase barriers."""
    import torch
    from smoke_r22_v14 import ms
    from b12x.moe._shared.kernels.w4a16 import mixed_trellis as api
    from b12x.moe._shared.kernels.w4a16.host import max_packed_route_slots
    from b12x.moe._shared.kernels.w4a16.prepare import prepare_trellis256_moe_weights
    device = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    original = os.environ.get("VLLM_GB10_SIGMOID")
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
                os.environ["VLLM_GB10_SIGMOID"] = str(int(enabled))
                launch = compile_fn(size_m=rows, hidden_size=hidden, intermediate_size=intermediate,
                    **{f"tier{i}_num_experts": count for i, count in enumerate(counts)},
                    top_k=topk, max_m_blocks=(max_packed_route_slots(rows * topk, block, sum(counts)) + block - 1) // block,
                    sms=props.multi_processor_count, max_shared_mem=props.shared_memory_per_block_optin,
                    force_tile_config=tiles, moe_block_size=block, broadcast_suh=True, broadcast_svh=True)
                buffers = buffers_fn(launch, device=device, sms=props.multi_processor_count)
                binding = bind_fn(*tiers, *maps, rotations, launch)
                compiled.append(launch.compiled)
                runs.append(lambda binding=binding, buffers=buffers: run_fn(x, weights, ids, binding, buffers))
            assert compiled[0] is not compiled[1], "sigmoid variants reused a stale compiled kernel"
            expected, actual = runs[0]().clone(), runs[1]()
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-5)
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
                torch.testing.assert_close(captured, expected, rtol=2e-3, atol=2e-5)
            results.append(dict(tiers=len(counts), legacy_ms=ms(runs[0]), shared_ms=ms(runs[1])))
    finally:
        if original is None:
            os.environ.pop("VLLM_GB10_SIGMOID", None)
        else:
            os.environ["VLLM_GB10_SIGMOID"] = original
    return dict(mixed_activation_timings=results)


if __name__ == "__main__":
    main()
