"""Run in the built image on each Spark before model serving (no weights needed)."""
import json
import torch
from sglang.kernels.ops.attention.dsv4.topk import (
    plan_topk_v2, topk_transform_paged_v2, topk_transform_ragged_v2,
)


def check(k):
    width, page_size = 32768, 64
    lens = torch.tensor([width, 117, 0], device='cuda', dtype=torch.int32)
    scores = torch.arange(width, device='cuda', dtype=torch.float32).repeat(3, 1)
    pages = torch.arange(width // page_size, device='cuda', dtype=torch.int32).flip(0).repeat(3, 1)
    out = torch.empty((3, k), device='cuda', dtype=torch.int32)
    raw = torch.empty_like(out)
    ragged = torch.empty_like(out)
    offsets = torch.zeros(3, device='cuda', dtype=torch.int32)
    plan = plan_topk_v2(lens)

    def run():
        topk_transform_paged_v2(scores, lens, pages, out, page_size, plan, raw)
        topk_transform_ragged_v2(scores, lens, out_offsets=offsets, out_indices=ragged)

    def verify():
        for row, length in enumerate(lens.tolist()):
            n = min(k, length)
            expected = scores[row, :length].topk(n).indices.to(torch.int32).sort().values
            for actual in (raw[row], ragged[row]):
                chosen = actual[actual >= 0].sort().values
                torch.testing.assert_close(chosen, expected, rtol=0, atol=0)
            expected_pages = pages[row, expected.long() // page_size] * page_size + expected % page_size
            torch.testing.assert_close(out[row][out[row] >= 0].sort().values,
                                       expected_pages.sort().values, rtol=0, atol=0)

    for _ in range(3): run()
    torch.cuda.synchronize()
    verify()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): run()
    scores.neg_()  # Replay must select different tokens, still exactly k.
    graph.replay()
    torch.cuda.synchronize()
    verify()
    return {'index_topk': k, 'paged_raw_and_ragged': 'pass', 'cuda_graph_replay': 'pass'}


if __name__ == '__main__':
    print(json.dumps({'gpu': torch.cuda.get_device_name(), 'checks': [check(512), check(2048)]}, indent=2))
