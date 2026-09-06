#!/usr/bin/env python3
"""v12 numerical checks and an isolated CKV metadata timing comparison."""
import argparse
import json
from pathlib import Path


def check_gpu():
    import torch
    from vllm.v1.attention.backends.mla import b12x_mla_sparse as backend
    from smoke_r22_performance import check_ckv_attention

    assert torch.cuda.get_device_capability() == (12, 1)
    torch.manual_seed(5312)
    cases = 0
    timing = {}
    for interleave in (1, 16):
        for rows_per_req in (1, 16, 128, 2048):
            # Two requests, independent rank-local packed starting offsets,
            # unequal shards, invalid holes and unsorted causal selections.
            rows, width, world = rows_per_req * 2, 2048, 4
            seq = torch.tensor([rows_per_req + 37, rows_per_req + 193], device="cuda", dtype=torch.int32)
            qsl = torch.tensor([0, rows_per_req, rows], device="cuda", dtype=torch.int32)
            req = torch.arange(2, device="cuda", dtype=torch.int32).repeat_interleave(rows_per_req)
            lens_cpu = [[sum((t // interleave) % world == r for t in range(n))
                         for n in (rows_per_req + 37, rows_per_req + 193)] for r in range(world)]
            lens = torch.tensor(lens_cpu, device="cuda", dtype=torch.int32)
            starts = torch.zeros_like(lens)
            starts[:, 1] = lens[:, 0]
            padded = int(lens.sum(1).max().item()) + 16
            causal_ref = seq[req.long()] - qsl[req.long() + 1] + torch.arange(rows, device="cuda", dtype=torch.int32) + 1
            ids = torch.arange(width, device="cuda", dtype=torch.int32)[None].expand(rows, -1).clone()
            ids.masked_fill_(ids >= causal_ref[:, None], -1)
            ids[:, 3::13] = -1
            ids = ids[:, torch.randperm(width, device="cuda")].contiguous()
            out, counts, causal = torch.empty_like(ids), torch.empty_like(req), torch.empty_like(req)
            old_out, old_counts = torch.empty_like(out), torch.empty_like(counts)

            def fused():
                backend._v12_prepare_ckv_metadata(req, ids, starts, lens, seq, qsl,
                    out, counts, causal, dcp_size=world, interleave=interleave, padded_tokens=padded)

            def legacy():
                backend._map_global_topk_to_gathered_ckv(req, ids, starts, lens, old_out, old_counts,
                    dcp_size=world, cp_kv_cache_interleave_size=interleave, padded_rank_tokens=padded)
                lengths = backend._global_causal_lens_for_ckv_gather(seq, qsl, req, rows).contiguous()
                torch.minimum(old_counts, lengths, out=old_counts)
                backend._mask_page_table_after_nsa_len(old_out, old_counts)

            fused()
            legacy()
            torch.cuda.synchronize()
            torch.testing.assert_close(causal, causal_ref, rtol=0, atol=0)
            torch.testing.assert_close(counts, old_counts, rtol=0, atol=0)
            # Legacy cross-CTA atomics do not define the order of selected ids.
            torch.testing.assert_close(out.sort(1).values, old_out.sort(1).values, rtol=0, atol=0)
            graph = torch.cuda.CUDAGraph()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                fused()
            stream.synchronize()
            with torch.cuda.graph(graph, stream=stream):
                fused()
            # A replay must overwrite the full old row, including its tail.
            ids.fill_(-1)
            torch.cuda.synchronize()
            graph.replay()
            torch.cuda.synchronize()
            assert bool((out == -1).all()) and bool((counts == 0).all())
            torch.testing.assert_close(causal, causal_ref, rtol=0, atol=0)
            if rows_per_req == 2048:
                ids.copy_(torch.arange(width, device="cuda", dtype=torch.int32)[None].expand(rows, -1))
                ids.masked_fill_(ids >= causal_ref[:, None], -1)
                def ms(fn):
                    for _ in range(3):
                        fn()
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record()
                    for _ in range(20):
                        fn()
                    end.record()
                    end.synchronize()
                    return begin.elapsed_time(end) / 20
                timing[str(interleave)] = {"legacy_ms": ms(legacy), "v12_ms": ms(fused)}
            cases += 1

    # Reuse the real FP8-cache/full-attention comparison with the new mapper.
    # Its sorted request rows let us reconstruct query boundaries from req_ids.
    original_mapper = backend._map_global_topk_to_gathered_ckv
    def mapper(req, ids, starts, lens, out, counts, *, dcp_size,
               cp_kv_cache_interleave_size, padded_rank_tokens):
        nreq = lens.shape[1]
        lengths = lens.sum(0).to(torch.int32)
        qsl = torch.cat((torch.zeros(1, device=req.device, dtype=torch.int32),
                         torch.bincount(req.long(), minlength=nreq).cumsum(0).to(torch.int32)))
        causal = torch.empty_like(counts)
        backend._v12_prepare_ckv_metadata(req, ids, starts, lens, lengths, qsl,
            out, counts, causal, dcp_size=dcp_size, interleave=cp_kv_cache_interleave_size,
            padded_tokens=padded_rank_tokens)
    try:
        backend._map_global_topk_to_gathered_ckv = mapper
        attention = check_ckv_attention()
    finally:
        backend._map_global_topk_to_gathered_ckv = original_mapper
    # Borrow eligibility: packed q works; scratch aliasing/unaligned views do not.
    q = torch.randn(4, 64, 576, device="cuda", dtype=torch.bfloat16)
    scratch = torch.empty(1024, device="cuda", dtype=torch.uint8)
    assert backend._v12_can_borrow_query(q, scratch, True, 4, 64, 576)
    assert not backend._v12_can_borrow_query(q, q.view(torch.uint8), True, 4, 64, 576)
    assert not backend._v12_can_borrow_query(q, scratch, False, 4, 64, 576)
    return {"metadata_cases": cases, "metadata_4096_rows": timing, "attention": attention}


def main():
    import vllm
    from patch_r22_v12 import patch, VERSION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--distributed", action="store_true")
    args = parser.parse_args()
    patch(Path(vllm.__file__).parent, check=True)
    result = {"overlay": VERSION}
    if args.gpu:
        result.update(check_gpu())
    if args.distributed:
        from smoke_r22_performance import check_distributed_roce
        result.update(check_distributed_roce())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
