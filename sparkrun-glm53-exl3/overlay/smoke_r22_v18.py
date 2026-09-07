"""v18 metadata and partition-equivalence checks, plus all inherited GPU tests."""
import json
from pathlib import Path
import sys


def assert_topk_result(indices, scores, logits, lengths):
    """Validate membership by score, allowing different IDs at an exact tie.

    B12X's local tiled selector uses atomics to take the last tied candidates.
    It does not promise the stable-ID ordering of the separate DCP reducer.
    Inputs are CPU arrays so this oracle is also exercised by offline tests.
    """
    import numpy as np
    ids, values, reference, lens = map(np.asarray, (indices, scores, logits, lengths))
    assert ids.ndim == 2 and values.shape == ids.shape
    assert reference.ndim == 2 and reference.shape[0] == ids.shape[0]
    assert lens.shape == (ids.shape[0],)
    assert np.issubdtype(ids.dtype, np.integer)
    assert np.all((ids >= 0) & (ids < lens[:, None]) & (ids < reference.shape[1]))
    ordered = np.sort(ids, axis=1)
    assert np.all(ordered[:, 1:] != ordered[:, :-1]), 'duplicate top-k IDs'
    assert np.isfinite(values).all(), 'non-finite top-k scores'
    expected_at_ids = np.take_along_axis(reference, ids, axis=1)
    np.testing.assert_allclose(values, expected_at_ids, rtol=1e-4, atol=1e-4)
    topk = ids.shape[1]
    expected = np.partition(reference, reference.shape[1] - topk, axis=1)[:, -topk:]
    np.testing.assert_allclose(np.sort(values, axis=1), np.sort(expected, axis=1),
                               rtol=1e-4, atol=1e-4)


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
    # Deliberately tie-heavy. Local top-k may choose different equally scored
    # IDs, so validate both selections against the full score oracle.
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
    logits = paged_decode_logits_reference(q_fp8=q, weights=weights, index_k_cache=cache,
        real_page_table=table, query_row_to_batch=torch.arange(rows, device='cuda', dtype=torch.int32),
        seqlens_per_query=lens)
    reference, lengths = logits.cpu().numpy(), lens.cpu().numpy()
    for name, (ids, scores) in zip(('partitioned', 'coalesced'), outputs):
        try:
            assert_topk_result(ids.cpu().numpy(), scores.cpu().numpy(), reference, lengths)
        except AssertionError as exc:
            raise AssertionError(f'v18 {name} top-k failed the independent oracle') from exc
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
        print(json.dumps(result, sort_keys=True), flush=True)
        result.update(partition_gpu())
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
