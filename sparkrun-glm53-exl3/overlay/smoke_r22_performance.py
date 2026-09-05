#!/usr/bin/env python3
"""GPU correctness checks for the performance overlay; no model weights needed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def check_installed_overlay() -> dict[str, str]:
    import vllm
    from patch_r22_performance import VERSION, patch

    patch(Path(vllm.__file__).parent, check=True)
    return {"performance_overlay": VERSION}


def check_ckv_attention() -> dict[str, object]:
    import torch
    from vllm import _custom_ops as ops
    from vllm.utils.b12x import get_b12x_sparse_mla
    from vllm.v1.attention.backends.mla.b12x_mla_sparse import (
        _map_global_topk_to_gathered_ckv,
        _round_up_ckv_rank_tokens,
    )

    module = get_b12x_sparse_mla()
    device = torch.device("cuda", 0)
    torch.manual_seed(53010)
    world, rows, tokens, heads, width = 4, 32, 257, 64, 2048
    kv = torch.randn(tokens, 512, dtype=torch.bfloat16, device=device) * 0.1
    pe = torch.randn(tokens, 64, dtype=torch.bfloat16, device=device) * 0.1
    q = torch.randn(rows, heads, 576, dtype=torch.bfloat16, device=device) * 0.1
    scale = torch.ones(1, dtype=torch.float32, device=device)

    def pack(k, p):
        cache = torch.zeros(((k.shape[0] + 63) // 64, 64, 656), dtype=torch.uint8, device=device)
        ops.concat_and_cache_mla(k, p, cache, torch.arange(k.shape[0], device=device), "fp8_ds_mla", scale)
        return cache

    global_cache = pack(kv, pe)
    causal = torch.arange(tokens - rows + 1, tokens + 1, dtype=torch.int32, device=device)
    selected = torch.arange(width, dtype=torch.int32, device=device).expand(rows, -1).contiguous()
    selected.masked_fill_(selected >= causal[:, None], -1)
    lens = torch.tensor([len(range(rank, tokens, world)) for rank in range(world)], dtype=torch.int32, device=device)[:, None]
    starts = torch.zeros_like(lens)
    padded = _round_up_ckv_rank_tokens(int(lens.max().item()), page_size=64, dcp_world_size=world)
    gathered = torch.zeros(world * padded, 656, dtype=torch.uint8, device=device)
    local_caches = []
    for rank in range(world):
        cache = pack(kv[rank::world].contiguous(), pe[rank::world].contiguous())
        # Reversed physical pages exercise block-table addressing, not just a copy.
        cache = cache.flip(0).contiguous()
        table = torch.arange(cache.shape[0] - 1, -1, -1, dtype=torch.int32, device=device)[None]
        count = int(lens[rank].item())
        cu = torch.tensor([0, count], dtype=torch.int32, device=device)
        ops.cp_gather_cache(cache, gathered[rank * padded:rank * padded + count], table, cu, 1)
        expected = pack(kv[rank::world].contiguous(), pe[rank::world].contiguous())
        torch.testing.assert_close(gathered[rank * padded:rank * padded + count], expected.view(-1, 656)[:count], rtol=0, atol=0)
        local_caches.append(expected)

    mapped, counts = torch.empty_like(selected), torch.empty_like(causal)
    req_ids = torch.zeros(rows, dtype=torch.int32, device=device)
    _map_global_topk_to_gathered_ckv(req_ids, selected, starts, lens, mapped, counts,
                                  dcp_size=world, cp_kv_cache_interleave_size=1,
                                  padded_rank_tokens=padded)
    torch.testing.assert_close(counts, causal, rtol=0, atol=0)

    plans = {}
    def run(query, cache, indices, lengths):
        nheads = query.shape[1]
        if nheads not in plans:
            plans[nheads] = module.plan(module.Caps(
                device=device, num_q_heads=nheads, max_q_rows=rows,
                max_width=width, softmax_scale=576 ** -0.5,
                dtype=torch.bfloat16, kv_dtype=torch.uint8, head_dim=576,
                v_head_dim=512, mode="extend", max_batch=rows,
                max_chunks_per_row=32, page_size=64, return_lse=True,
                lse_scale="natural", model_type=int(module.ModelType.GLM_NSA),
                cache_record_bytes=656,
            ))
        plan = plans[nheads]
        scratch = torch.empty(int(plan.layout.nbytes), dtype=torch.uint8, device=device)
        binding = module.bind(plan, scratch=scratch, q=query, kv_cache=cache,
                              selected_indices=indices, cache_lengths=lengths,
                              selected_lengths=lengths)
        output, lse = module.run(binding)
        return output.clone(), lse.clone()

    local_q = q[:, :heads // world].contiguous()
    reference, _ = run(local_q, global_cache, selected, causal)
    actual, _ = run(local_q, gathered.view(-1, 64, 656), mapped, counts)
    torch.testing.assert_close(actual, reference, rtol=0.02, atol=0.002)

    # Also compare with four ordinary DCP shards and their LSE-weighted merge.
    partials, lses = [], []
    for rank, cache in enumerate(local_caches):
        index = torch.full_like(selected, -1)
        count = torch.zeros_like(causal)
        for row in range(rows):
            ids = selected[row]
            ids = ids[(ids >= 0) & (ids % world == rank)] // world
            index[row, :ids.numel()] = ids
            count[row] = ids.numel()
        out, lse = run(q, cache, index, count)
        partials.append(out.float())
        lses.append(lse.reshape(rows, heads).float())
    weights = torch.softmax(torch.stack(lses), dim=0)
    merged = (torch.stack(partials) * weights[..., None]).sum(0)
    torch.testing.assert_close(actual.float(), merged[:, :heads // world], rtol=0.02, atol=0.002)
    return {"ckv_attention": "passed", "ranks_simulated": world, "rows": rows,
            "unequal_shards": True, "nonsequential_pages": True}


def check_distributed_roce() -> dict[str, object]:
    """Run under torchrun with one GPU per host after the image is distributed."""
    import os
    import torch
    import torch.distributed as dist
    from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator

    os.environ["VLLM_ENABLE_ROCE_ALLREDUCE"] = "1"
    os.environ["VLLM_ROCE_DCP_ENABLE"] = "1"
    os.environ["VLLM_ROCE_DCP_RS_MAX_BYTES"] = "262144"
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 4, "qualification requires four one-GPU hosts"
    torch.cuda.set_device(0)
    cpu_group = dist.new_group(backend="gloo")
    comm = CudaCommunicator(cpu_group, torch.device("cuda", 0), dist.group.WORLD,
                            unique_name="dcp:performance-smoke")
    assert comm.use_roce_allreduce and comm.b12x_ar_comm is not None
    assert not comm.b12x_ar_comm.disabled, "RoCEnante fell back during qualification"

    for rows in (1, 4, 8, 32):
        x = torch.full((rows, 16, 576), rank + 1.0, dtype=torch.bfloat16, device="cuda")
        y = torch.full((rows, 64, 512), rank + 1.0, dtype=torch.bfloat16, device="cuda")
        y.add_(torch.arange(64, device="cuda", dtype=torch.bfloat16)[None, :, None])
        lse = torch.full((rows, 64), rank + 1.0, dtype=torch.float32, device="cuda")
        def step():
            return (comm.all_gather(x, dim=1), comm.reduce_scatter(y, dim=1),
                    comm.all_gather(lse, dim=0))
        expected_q = torch.cat([torch.full_like(x, r + 1.0) for r in range(world)], dim=1)
        expected_y = torch.full((rows, 16, 512), world * (world + 1) / 2,
                                dtype=torch.bfloat16, device="cuda")
        expected_y.add_(world * torch.arange(rank * 16, (rank + 1) * 16,
                        dtype=torch.bfloat16, device="cuda")[None, :, None])
        expected_lse = torch.cat([torch.full_like(lse, r + 1.0) for r in range(world)])
        for _ in range(3):
            gathered, reduced, gathered_lse = step()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with comm.b12x_ar_comm.capture():
            with torch.cuda.graph(graph):
                gathered, reduced, gathered_lse = step()
        # Change input values to detect stale capture-time results.
        x.add_(1)
        y.add_(1)
        lse.add_(1)
        expected_q.add_(1)
        expected_y.add_(world)
        expected_lse.add_(1)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        comm.b12x_ar_comm.check_health()
        torch.testing.assert_close(gathered, expected_q, rtol=0, atol=0)
        torch.testing.assert_close(reduced, expected_y, rtol=0, atol=0)
        torch.testing.assert_close(gathered_lse, expected_lse, rtol=0, atol=0)
    dist.barrier()
    comm.destroy()
    dist.destroy_process_group()
    return {"roce_dcp_graph_replay": "passed", "rank": rank}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    result = check_installed_overlay()
    if args.distributed:
        result.update(check_distributed_roce())
    elif args.gpu:
        result.update(check_ckv_attention())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
