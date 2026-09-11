#!/usr/bin/env python3
"""Tiny GPU/SSD oracle test: all nibble codes, TP edges, repeats, graph consumer.

This checks the adapter and io_uring access, not the full model or TP collectives.
"""
import tempfile
from pathlib import Path


def main():
    import torch
    from b12x.sequence import engram
    from b12x.sequence.engram._fp4_disk import expand_rows
    from b12x.sequence.engram._kernels import lookup_op
    assert torch.cuda.get_device_capability() == (12, 1), "Expected Spark SM121"
    device = torch.device("cuda:0")
    # Exhaust every packed byte, checking both nibble order and signed zero bits.
    packed = torch.arange(256, device=device).to(torch.uint8).reshape(2, 128)
    decoded = torch.empty((2, 256), dtype=torch.float8_e4m3fn, device=device)
    expand_rows(packed, decoded, 2)
    lut = torch.tensor([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6], device=device)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).long().reshape(2, 256)
    oracle = lut[codes].to(torch.float8_e4m3fn)
    assert torch.equal(decoded.view(torch.uint8), oracle.view(torch.uint8))

    geometry = engram.build_geometry(base_table_size=2, compressed_vocab_size=32)
    with tempfile.TemporaryDirectory(prefix="ds41-fp4-") as temp:
        root = Path(temp)
        for rank in range(4):
            plan = engram.plan(engram.Caps(device=device, max_tokens=3, max_seqs=3,
                                         max_requests=4, vocab_size=32, layer_id=1,
                                         tp_rank=rank, tp_size=4),
                               token_map=list(range(32)), geometry=geometry)
            packed_cpu = (torch.arange(plan.table_rows * 128).reshape(-1,128) % 256).to(torch.uint8)
            scales_cpu = torch.tensor([0,1,123,127,130,253,254,255], dtype=torch.uint8).expand(plan.table_rows,-1).contiguous()
            table = engram.DiskTable(plan, weight_format="fp4", queue_depth=4)
            for is_scale, payload, offset in [(False,packed_cpu,4093),(True,scales_cpu,19)]:
                path = root / f"{rank}-{is_scale}.bin"
                path.write_bytes(bytes(offset) + payload.numpy().tobytes())
                table.add_shard(0, str(path), offset, scale=is_scale)
            rows = [-1,0,plan.shard_start-1,plan.shard_start,plan.shard_end-1,
                    plan.shard_end,plan.table_rows-1,plan.table_rows] * 3
            ids = torch.tensor([rows]*3, device=device, dtype=torch.int64)
            count = torch.tensor([3],device=device,dtype=torch.int32)
            out = torch.empty((3,6144),device=device,dtype=torch.bfloat16)
            binding = engram.bind_lookup(plan,disk_table=table,hash_ids=ids,num_tokens=count,out=out)
            # The fused path aliases the 128-byte cache: no FP8 intermediate.
            assert table.weight.data_ptr() == table._cache.weight.data_ptr()
            assert table.weight.dtype == torch.uint8 and table.weight.shape == (3*24,128)
            baseline_weight = torch.empty((3*24,256),dtype=torch.float8_e4m3fn,device=device)
            baseline_out = torch.empty_like(out)
            result = torch.empty_like(out)
            # Graph reads stable device output after disk preparation.
            engram.run_lookup(binding,token_count=3)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                result.copy_(out)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                result.copy_(out)
            for active, prepared in [(3,3),(1,2),(0,0),(2,3)]:
                count.fill_(active)
                ids.copy_(ids.flip(1))
                engram.run_lookup(binding,token_count=prepared)
                # Compare directly with v1's expansion + original FP8 lookup.
                expand_rows(table.weight,baseline_weight,prepared*24)
                lookup_op(baseline_weight,table.scale_bytes,ids,count,baseline_out,
                          plan.table_rows,plan.shard_start,plan.shard_end,
                          compact_rows=True,prepared_tokens=prepared)
                torch.testing.assert_close(out,baseline_out,rtol=0,atol=0,equal_nan=True)
                graph.replay()
                expected = torch.zeros((3,24,256), dtype=torch.bfloat16)
                for token in range(min(active,prepared)):
                    for column,row in enumerate(ids[token].cpu().tolist()):
                        if plan.shard_start <= row < min(plan.shard_end,plan.table_rows):
                            codes_cpu = torch.stack((packed_cpu[row]&15,packed_cpu[row]>>4),dim=-1).long().flatten()
                            scale = torch.pow(2.,scales_cpu[row].float()-127).repeat_interleave(32)
                            scale[scales_cpu[row].repeat_interleave(32)==255] = float('nan')
                            expected[token,column] = lut.cpu()[codes_cpu] * scale
                # CPU oracle uses ordinary scales; extreme E8M0 values are
                # checked against the original GPU lookup above (same FTZ rules).
                torch.testing.assert_close(result.cpu().view_as(expected)[...,64:160],
                                           expected[...,64:160],rtol=0,atol=0)
                assert torch.count_nonzero(result[min(active,prepared):]) == 0
            assert table.stats()["weight_cache_bytes"] == 3*24*128
    print("DS41_FP4_DISK_GPU_PASS: fused/v1 parity, E8M0 extremes, SSD reads, TP4 edges, fresh graph replay")


if __name__ == "__main__":
    main()
