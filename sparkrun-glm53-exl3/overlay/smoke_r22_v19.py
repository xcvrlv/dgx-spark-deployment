#!/usr/bin/env python3
"""Cumulative image gates plus mixed-K M16/grouped-FC2 numerical and graph tests."""
import json
import os
from pathlib import Path
import sys


def mixed_schedule_gpu():
    import torch
    from b12x.moe._shared.kernels.w4a16 import mixed_trellis as api
    from b12x.moe._shared.kernels.w4a16.host import max_packed_route_slots
    from b12x.moe._shared.kernels.w4a16.prepare import prepare_trellis256_moe_weights
    from smoke_r22_v14 import ms
    device = torch.device('cuda', torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    assert torch.cuda.get_device_capability() == (12, 1), 'v19 requires a GB10 GPU'
    torch.manual_seed(5319)
    setting = 'VLLM_GB10_EXL3_FC2_GROUP'
    original = os.environ.get(setting)
    results = []
    try:
        for counts in ((8, 8), (6, 5, 5)):
            hidden, intermediate, topk = 6144, 512, 8
            tiles = (128, 128, 32, 512)
            shared = [torch.ones(1, hidden, device=device, dtype=torch.float16) for _ in range(3)]
            shared[1].mul_(.75)
            tiers = [prepare_trellis256_moe_weights(
                hidden_size=hidden, intermediate_size=intermediate, num_experts=count,
                activation='silu', fc1_tile_n=tiles[1], fc2_tile_n=tiles[3], device=device,
                seed=5319+i, params_dtype=torch.float16, w13_layout='trellis_t256_proj',
                trellis_bits=3+i, codebook='mcg', gate_suh=shared[0], up_suh=shared[1],
                down_svh=shared[2], intermediate_rotations=torch.ones(count, 3*intermediate,
                    device=device, dtype=torch.float16), tile_config=tiles)
                for i, count in enumerate(counts)]
            rotations = api.MixedTrellisRotations(
                intermediate=torch.cat([t.intermediate_rotations for t in tiers]),
                gate_suh=shared[0], up_suh=shared[1], down_svh=shared[2])
            projection = tuple(i for i, count in enumerate(counts) for _ in range(count))
            maps = api.build_projection_tiered_maps(projection, projection, projection,
                                                    tier_slots=counts, device=device)
            suffix = '3' if len(counts) == 3 else ''
            compile_fn = getattr(api, 'compile_mixed_trellis'+suffix)
            buffers_fn = getattr(api, 'make_mixed_trellis'+suffix+'_buffers')
            bind_fn = getattr(api, 'bind_mixed_trellis'+suffix)
            run_fn = getattr(api, 'run_bound_mixed_trellis'+suffix)
            # Non-multiples of M16/M32 exercise route padding. Random unique
            # top-8 over 16 experts leaves different tails for every expert.
            for rows in (129, 513):
                x = torch.randn(rows, hidden, device=device, dtype=torch.bfloat16)*1e-3
                ids = torch.rand(rows, sum(counts), device=device).argsort(-1)[:, :topk].int().contiguous()
                weights = torch.rand(rows, topk, device=device)
                weights /= weights.sum(-1, keepdim=True)
                runs, launches = {}, {}
                for block, group in ((32, 2), (16, 2), (16, 1), (32, 1), (32, 4)):
                    os.environ[setting] = str(group)
                    launch = compile_fn(size_m=rows, hidden_size=hidden, intermediate_size=intermediate,
                        **{f'tier{i}_num_experts': count for i, count in enumerate(counts)},
                        top_k=topk, max_m_blocks=(max_packed_route_slots(rows*topk, block, sum(counts))+block-1)//block,
                        sms=props.multi_processor_count, max_shared_mem=props.shared_memory_per_block_optin,
                        force_tile_config=tiles, moe_block_size=block, broadcast_suh=True, broadcast_svh=True)
                    assert launch.fc2_moe_block_size == 8
                    assert launch.fc2_schedule_route_block_factor == group
                    assert launch.shared_memory_bytes <= props.shared_memory_per_block_optin
                    buffers = buffers_fn(launch, device=device, sms=props.multi_processor_count)
                    binding = bind_fn(*tiers, *maps, rotations, launch)
                    launches[block, group] = launch
                    runs[block, group] = lambda binding=binding, buffers=buffers: run_fn(x, weights, ids, binding, buffers)
                assert len({id(v.compiled) for v in launches.values()}) == len(launches), 'stale kernel cache key'

                def compare(actual, expected, exact=False):
                    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
                    rms = expected.float().square().mean().sqrt().item()
                    assert rms > 0, 'degenerate all-zero reference'
                    if exact:
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    else:
                        # FC1 changes MMA row geometry; allow FP16 intermediate
                        # and BF16 output rounding, without a fixed absolute
                        # tolerance that would mask a small-output failure.
                        torch.testing.assert_close(actual, expected, rtol=.01, atol=rms*.001)
                        error = (actual.float()-expected.float()).square().mean().sqrt().item()/rms
                        assert error <= .005, f'relative RMS error {error}'

                expected = runs[32, 2]().clone()
                for key, run in runs.items():
                    compare(run(), expected, exact=key[0] == 32)
                # Capture every candidate, then change activations, weights and
                # expert ownership. The final replay leaves half the experts
                # empty, catching stale tails and cross-expert grouping errors.
                for key, run in runs.items():
                    if key == (32, 2):
                        continue
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        run()
                    stream.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        captured = run()
                    for replay in range(2):
                        x.normal_(0, 1e-3)
                        weights.uniform_()
                        weights /= weights.sum(-1, keepdim=True)
                        choices = sum(counts) if replay == 0 else topk
                        ids.copy_(torch.rand(rows, choices, device=device).argsort(-1)[:, :topk].int())
                        expected = runs[32, 2]().clone()
                        graph.replay()
                        torch.cuda.synchronize()
                        compare(captured, expected, exact=key[0] == 32)
                    del graph, captured
                ids.copy_(torch.rand(rows, sum(counts), device=device).argsort(-1)[:, :topk].int())
                timings = []
                for (block, group), run in runs.items():
                    launch = launches[block, group]
                    timings.append(dict(block_m=block, fc2_group=group, ms=ms(run),
                        blocks_per_sm=launch.blocks_per_sm, shared_bytes=launch.shared_memory_bytes))
                row = dict(tiers=len(counts), rows=rows, timings=timings)
                results.append(row)
                print(json.dumps(dict(v19_mixed_schedule=row)), flush=True)
                runs.clear()
                launches.clear()
    finally:
        if original is None:
            os.environ.pop(setting, None)
        else:
            os.environ[setting] = original
    return dict(v19_mixed_schedule='passed', v19_schedule_timings=results,
        v19_device=dict(sms=props.multi_processor_count,
            shared_memory_per_block_optin=props.shared_memory_per_block_optin,
            shared_memory_per_multiprocessor=props.shared_memory_per_multiprocessor))


def main():
    import b12x
    from patch_r22_v19 import patch, VERSION, OUTPUTS
    patch(Path(b12x.__file__).parent, check=True)
    from smoke_r22_v18 import main as inherited
    inherited(source_overrides=OUTPUTS)
    result = dict(mixed_schedule_overlay=VERSION)
    if '--gpu' in sys.argv:
        result.update(mixed_schedule_gpu())
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
