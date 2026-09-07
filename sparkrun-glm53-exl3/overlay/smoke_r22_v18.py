"""v18 metadata and partition-equivalence checks, plus all inherited GPU tests."""
import json
from pathlib import Path
import sys


def metadata_gpu():
    import torch
    from vllm.v1.attention.backends.mla.gb10_indexer_prefill import build_paged_chunk, _causal_metadata
    from vllm.v1.attention.backends.mla.indexer import build_prefill_chunk_metadata
    cases = 0
    for world in (1, 4):
        for interleave in (1, 16):
            for rank in range(world):
                # A preceding request, partial final chunk and long cached prefix.
                # CPU context is an upper bound; GPU metadata is authoritative.
                for total, upper in ((263, 263), (131069, 131072)):
                    qcpu = torch.tensor([0, 4, 267], dtype=torch.int32)
                    qsl = qcpu.cuda()
                    seq = torch.tensor([19, total], dtype=torch.int32, device='cuda')
                    seqcpu = torch.tensor([19, upper], dtype=torch.int32)
                    table = torch.arange(4096, dtype=torch.int32, device='cuda').view(2, -1)
                    args = (1, 2, qsl, qcpu, seq, seq, seqcpu, table, 1)
                    kwargs = dict(query_slice=slice(127, 263), dcp_rank=rank,
                                  dcp_world_size=world, cp_kv_cache_interleave_size=interleave)
                    old = build_prefill_chunk_metadata(*args, **kwargs)
                    new = build_paged_chunk(*args, **kwargs)
                    for field in ('cu_seqlen_ks', 'cu_seqlen_ke', 'cu_seq_lens', 'local_cu_seq_lens'):
                        torch.testing.assert_close(getattr(new, field), getattr(old, field), rtol=0, atol=0)
                    live_pages = (old.local_total_seq_lens + 63) // 64
                    torch.testing.assert_close(new.block_table[:, :live_pages], old.block_table[:, :live_pages], rtol=0, atol=0)
                    assert bool(torch.all(new.block_table[:, live_pages:] == -1))
                    assert new.token_start == old.token_start and new.token_end == old.token_end
                    assert new.token_to_seq.numel() == 0
                    assert new.local_total_seq_lens >= old.local_total_seq_lens
                    assert new.b12x_seq_lens.data_ptr() == new.cu_seqlen_ke.data_ptr()
                    # A separate CPU oracle enumerates global token ownership.
                    counts = ((torch.arange(total) // interleave) % world == rank).cumsum(0)
                    expected = counts[total - 263 + 127:total].to(torch.int32)
                    torch.testing.assert_close(new.b12x_seq_lens.cpu(), expected, rtol=0, atol=0)
                    cases += 1
    # Device lengths remain live under replay; no baked-in CPU length.
    graph = torch.cuda.CUDAGraph()
    def kernel():
        width = new.block_table.shape[1]
        _causal_metadata[((max(136, width) + 255) // 256,)](qsl, seq, new.cu_seqlen_ks, new.cu_seqlen_ke,
            new.cu_seq_lens, new.local_cu_seq_lens, table[1], new.block_table, width,
            1, 127, 136, rank, world, interleave, 256)
    with torch.cuda.graph(graph):
        kernel()
    seq[1] -= 17
    graph.replay()
    torch.cuda.synchronize()
    expected = build_prefill_chunk_metadata(*args, **kwargs)
    torch.testing.assert_close(new.b12x_seq_lens, expected.cu_seqlen_ke - expected.cu_seqlen_ks, rtol=0, atol=0)
    return dict(v18_metadata_cases=cases, v18_live_causal_replay='passed')


def partition_gpu():
    import torch
    from b12x.attention import dsa_indexer as mod
    from b12x.attention.dsa_indexer.reference import pack_index_k_cache_reference, paged_decode_logits_reference
    from smoke_r22_v14 import ms
    torch.manual_seed(5318)
    rows, heads, context, topk = 65, 4, 65536, 2048
    # Integer dot products make score ties exact and expose partition-dependent
    # membership changes without an ambiguous floating-point threshold.
    q = torch.randint(-1, 2, (rows, heads, 128), device='cuda').to(torch.float8_e4m3fn)
    weights = torch.ones((rows, heads), dtype=torch.float32, device='cuda')
    cache = pack_index_k_cache_reference(torch.randint(-1, 2, (context, 128), device='cuda').float())
    table = torch.arange(context // 64, dtype=torch.int32, device='cuda').view(1, -1).expand(rows, -1)
    lens = torch.arange(context - rows + 1, context + 1, dtype=torch.int32, device='cuda')
    active = torch.tensor([context], dtype=torch.int32, device='cuda')
    plan = mod.plan(mod.Caps(device=q.device, num_q_heads=heads, max_q_rows=rows,
        max_page_table_width=table.shape[1], topk=topk, mode='prefill', max_batch=1))
    scratch = tuple(torch.empty(shape, dtype=dtype, device='cuda') for shape, dtype in plan.shapes_and_dtypes())
    outputs = []
    runs = []
    for chunk_size in (16, rows):
        indices = torch.empty((rows, topk), dtype=torch.int32, device='cuda')
        scores = torch.empty((rows, topk), dtype=torch.float32, device='cuda')
        bindings = [mod.bind(plan, scratch=scratch, q_fp8=q[start:start + chunk_size],
            query_weights=weights[start:start + chunk_size], index_k_cache=cache,
            page_table=table[start:start + chunk_size], cache_lengths=lens[start:start + chunk_size],
            active_width=active, output_indices=indices[start:start + chunk_size],
            output_scores=scores[start:start + chunk_size]) for start in range(0, rows, chunk_size)]
        def run(bindings=bindings):
            for binding in bindings:
                mod.run(binding)
        run()
        runs.append(run)
        outputs.append((indices, scores))
    torch.cuda.synchronize()
    # Compare sorted (ID, score) pairs, not unspecified output column order.
    old_ids, new_ids = (torch.sort(out[0], dim=1) for out in outputs)
    torch.testing.assert_close(old_ids.values, new_ids.values, rtol=0, atol=0)
    torch.testing.assert_close(outputs[0][1].gather(1, old_ids.indices),
                               outputs[1][1].gather(1, new_ids.indices), rtol=1e-4, atol=1e-4)
    logits = paged_decode_logits_reference(q_fp8=q, weights=weights, index_k_cache=cache,
        real_page_table=table, query_row_to_batch=torch.arange(rows, device='cuda', dtype=torch.int32),
        seqlens_per_query=lens)
    ids, scores = outputs[1]
    assert bool(torch.all((ids >= 0) & (ids < lens[:, None])))
    assert bool(torch.all(new_ids.values[:, 1:] != new_ids.values[:, :-1]))
    torch.testing.assert_close(scores, logits.gather(1, ids.long()), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(torch.sort(scores, dim=1).values,
        torch.sort(torch.topk(logits, topk, dim=1).values, dim=1).values, rtol=1e-4, atol=1e-4)
    return dict(v18_paged_partition='passed', v18_partitioned_ms=ms(runs[0]), v18_coalesced_ms=ms(runs[1]))


def main():
    import vllm
    from patch_r22_v18 import patch, VERSION, OUTPUTS
    patch(Path(vllm.__file__).parent, check=True)
    from smoke_r22_v17 import main as inherited
    inherited(source_overrides=OUTPUTS)
    result = dict(continuation_overlay=VERSION)
    if '--gpu' in sys.argv:
        result.update(metadata_gpu())
        result.update(partition_gpu())
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
