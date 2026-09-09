#!/usr/bin/env python3
"""Cumulative image gates plus archived R7 paired-FC2 numerical and graph tests."""
import json
import os
from pathlib import Path
import sys


def fc2_pair_gpu():
    import torch
    from b12x.moe._shared.kernels.w4a16 import mixed_trellis as api
    from b12x.moe._shared.kernels.w4a16.host import max_packed_route_slots
    from b12x.moe._shared.kernels.w4a16.prepare import prepare_trellis256_moe_weights
    from smoke_r22_v14 import ms
    device = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    assert torch.cuda.get_device_capability() == (12, 1), "v21 requires a GB10 GPU"
    torch.manual_seed(5321)
    setting = "VLLM_GB10_EXL3_FC2_M8_PAIR"
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
                activation="silu", fc1_tile_n=tiles[1], fc2_tile_n=tiles[3], device=device,
                seed=5320+i, params_dtype=torch.float16, w13_layout="trellis_t256_proj",
                trellis_bits=3+i, codebook="mcg", gate_suh=shared[0], up_suh=shared[1],
                down_svh=shared[2], intermediate_rotations=torch.ones(count, 3*intermediate,
                    device=device, dtype=torch.float16), tile_config=tiles)
                for i, count in enumerate(counts)]
            rotations = api.MixedTrellisRotations(
                intermediate=torch.cat([t.intermediate_rotations for t in tiers]),
                gate_suh=shared[0], up_suh=shared[1], down_svh=shared[2])
            projection = tuple(i for i, count in enumerate(counts) for _ in range(count))
            maps = api.build_projection_tiered_maps(projection, projection, projection,
                                                    tier_slots=counts, device=device)
            suffix = "3" if len(counts) == 3 else ""
            compile_fn = getattr(api, "compile_mixed_trellis"+suffix)
            buffers_fn = getattr(api, "make_mixed_trellis"+suffix+"_buffers")
            bind_fn = getattr(api, "bind_mixed_trellis"+suffix)
            run_fn = getattr(api, "run_bound_mixed_trellis"+suffix)
            # M8 decode never reaches the pair path (factor 1). Compile with
            # the flag off and on and require a cache hit: the flag must not
            # perturb any M8 decode binary.
            decode = {}
            for flag in ("0", "1"):
                os.environ[setting] = flag
                decode[flag] = compile_fn(size_m=1, hidden_size=hidden,
                    intermediate_size=intermediate,
                    **{f"tier{i}_num_experts": count for i, count in enumerate(counts)},
                    top_k=topk,
                    max_m_blocks=(max_packed_route_slots(topk, 8, sum(counts))+7)//8,
                    sms=props.multi_processor_count,
                    max_shared_mem=props.shared_memory_per_block_optin,
                    force_tile_config=tiles, moe_block_size=8,
                    broadcast_suh=True, broadcast_svh=True)
            assert decode["0"].compiled is decode["1"].compiled, "M8 decode binary changed"
            # Prefill arms: M32/M64 route blocks at the supported groupings.
            # The pair must engage on grouped FC2, bump the compilation
            # identity and grow the shared footprint (doubled A slab and
            # metadata regions), and produce numerically identical outputs.
            for rows, block in ((513, 32), (257, 64)):
                x = torch.randn(rows, hidden, device=device, dtype=torch.bfloat16)*1e-3
                ids = torch.rand(rows, sum(counts), device=device).argsort(-1)[:, :topk].int().contiguous()
                weights = torch.rand(rows, topk, device=device)
                weights /= weights.sum(-1, keepdim=True)
                runs, launches = {}, {}
                for flag in ("0", "1"):
                    os.environ[setting] = flag
                    launch = compile_fn(size_m=rows, hidden_size=hidden,
                        intermediate_size=intermediate,
                        **{f"tier{i}_num_experts": count for i, count in enumerate(counts)},
                        top_k=topk,
                        max_m_blocks=(max_packed_route_slots(rows*topk, block, sum(counts))+block-1)//block,
                        sms=props.multi_processor_count,
                        max_shared_mem=props.shared_memory_per_block_optin,
                        force_tile_config=tiles, moe_block_size=block,
                        broadcast_suh=True, broadcast_svh=True)
                    assert launch.fc2_moe_block_size == 8
                    assert launch.shared_memory_bytes <= props.shared_memory_per_block_optin
                    buffers = buffers_fn(launch, device=device, sms=props.multi_processor_count)
                    binding = bind_fn(*tiers, *maps, rotations, launch)
                    launches[flag] = launch
                    runs[flag] = lambda binding=binding, buffers=buffers: run_fn(x, weights, ids, binding, buffers)
                assert launches["0"].compiled is not launches["1"].compiled, "stale kernel cache key"
                assert launches["1"].shared_memory_bytes > launches["0"].shared_memory_bytes, \
                    "pair did not grow the paired footprint"

                def compare(actual, expected):
                    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
                    rms = expected.float().square().mean().sqrt().item()
                    assert rms > 0, "degenerate all-zero reference"
                    # The pair reuses one B stream for two accumulator sets;
                    # the arithmetic order per expert is unchanged, so the
                    # agreement is exact up to bf16 store rounding.
                    torch.testing.assert_close(actual, expected, rtol=.01, atol=rms*.001)
                    error = (actual.float()-expected.float()).square().mean().sqrt().item()/rms
                    assert error <= .005, f"relative RMS error {error}"

                expected = runs["0"]().clone()
                compare(runs["1"](), expected)
                tiles_routed = int(buffers.packed_route_count.item())
                # Capture the paired schedule, then change activations,
                # router weights and expert ownership across replays. The
                # final replay concentrates routes on few experts, leaving
                # most tier experts empty through the pair halves.
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    runs["1"]()
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    captured = runs["1"]()
                for replay in range(2):
                    x.normal_(0, 1e-3)
                    weights.uniform_()
                    weights /= weights.sum(-1, keepdim=True)
                    choices = sum(counts) if replay == 0 else topk
                    ids.copy_(torch.rand(rows, choices, device=device).argsort(-1)[:, :topk].int())
                    expected = runs["0"]().clone()
                    graph.replay()
                    torch.cuda.synchronize()
                    compare(captured, expected)
                del graph, captured
                ids.copy_(torch.rand(rows, sum(counts), device=device).argsort(-1)[:, :topk].int())
                timings = []
                for flag in ("0", "1"):
                    timings.append(dict(fc2_pair=flag, ms=ms(runs[flag]),
                        blocks_per_sm=launches[flag].blocks_per_sm,
                        shared_bytes=launches[flag].shared_memory_bytes))
                row = dict(tiers=len(counts), rows=rows, packed_blocks=tiles_routed,
                    grid=props.multi_processor_count, timings=timings)
                results.append(row)
                print(json.dumps(dict(v21_fc2_pair=row)), flush=True)
                runs.clear()
                launches.clear()
    finally:
        if original is None:
            os.environ.pop(setting, None)
        else:
            os.environ[setting] = original
    return dict(v21_fc2_pair="passed", v21_pair_timings=results)


def main():
    import b12x
    from patch_r22_v21 import patch, VERSION, OUTPUTS
    patch(Path(b12x.__file__).parent, check=True)
    # v21 replaces both files changed by v19/v20. Its exact final hashes
    # above verify those files; predecessor patch checks would reject them.
    # Preserve cumulative expectations, with the newest revision winning.
    from patch_r22_v19 import OUTPUTS as v19_outputs
    from patch_r22_v20 import OUTPUTS as v20_outputs
    from smoke_r22_v18 import main as inherited
    inherited(source_overrides={**v19_outputs, **v20_outputs, **OUTPUTS})
    result = dict(fc2_pair_overlay=VERSION)
    if "--gpu" in sys.argv:
        from smoke_r22_v19 import mixed_schedule_gpu
        result.update(mixed_schedule_gpu())
        print(json.dumps(result, sort_keys=True), flush=True)
        from smoke_r22_v20 import fc1_tailsplit_gpu
        result.update(fc1_tailsplit_gpu())
        print(json.dumps(result, sort_keys=True), flush=True)
        result.update(fc2_pair_gpu())
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
