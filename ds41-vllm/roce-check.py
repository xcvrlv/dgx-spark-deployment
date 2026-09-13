"""Four-node fail-closed RoCEnante/NCCL comparison; invoked by fleet.py fabric."""
import json
import os
from datetime import timedelta
import torch
import torch.distributed as dist
from b12x.comm import roce
from vllm.distributed.device_communicators.b12x_roce_all_reduce import B12xRoceAllReduce

torch.cuda.set_device(0)
assert torch.cuda.get_device_capability() == (12, 1)
assert int(os.environ['WORLD_SIZE']) == 4
assert roce.is_supported(torch.device('cuda', 0)), 'RoCEnante unsupported; no skip permitted'
dist.init_process_group('gloo', timeout=timedelta(seconds=180))
nccl = dist.new_group(backend='nccl', timeout=timedelta(seconds=180))
rank = dist.get_rank()
adapter = B12xRoceAllReduce(dist.group.WORLD, nccl, torch.device('cuda', 0))
assert not adapter.disabled, 'vLLM disabled RoCEnante'
rt = adapter._runtime
assert tuple(rt.hca_names) == tuple(os.environ['B12X_ROCE_HCA'].split(',')), rt.hca_names
rt.prepare((torch.bfloat16, torch.float32, torch.float16))
before = rt.stats()['bytes_posted_per_hca']
for dtype in (torch.bfloat16, torch.float16, torch.float32):
    for nbytes in (16, 4096, 262144, 2 * 1024 * 1024):
        x = torch.full((nbytes // torch.empty((), dtype=dtype).element_size(),), rank + 1, dtype=dtype, device='cuda')
        expected = x.clone()
        dist.all_reduce(expected, group=nccl)
        assert adapter.should_custom_ar(x)
        actual = adapter.custom_all_reduce(x)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        gathered = rt.all_gather(x, dim=0)
        parts = [torch.empty_like(x) for _ in range(4)]
        dist.all_gather(parts, x, group=nccl)
        torch.testing.assert_close(gathered, torch.cat(parts), rtol=0, atol=0)
        adapter.check_health()
after = rt.stats()['bytes_posted_per_hca']
assert len(after) == 2 and all(a > b for a, b in zip(after, before)), (before, after)
assert not adapter.should_custom_ar(torch.empty(1024 * 1024 + 8, dtype=torch.bfloat16, device='cuda'))
# Capture mixed collectives, then change input on every replay (no stale-output pass).
x = torch.full((2048,), rank + 1, dtype=torch.bfloat16, device='cuda')
reduced, gathered = torch.empty_like(x), torch.empty(8192, dtype=x.dtype, device='cuda')
stream = torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream):
    rt.all_reduce(x, out=reduced)
    rt.all_gather(x, dim=0, out=gathered)
torch.cuda.current_stream().wait_stream(stream)
torch.cuda.synchronize()
dist.barrier()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, stream=stream), rt.capture(stream=stream):
    rt.all_reduce(x, out=reduced)
    rt.all_gather(x, dim=0, out=gathered)
for i in range(5):
    x.fill_(rank + i + 1)
    expected = x.clone()
    dist.all_reduce(expected, group=nccl)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(reduced, expected, rtol=0, atol=0)
    for peer in range(4):
        assert torch.all(gathered[peer * 2048:(peer + 1) * 2048] == peer + i + 1)
    adapter.check_health()
dist.barrier()
print(json.dumps({'rank': rank, 'status': 'PASS', 'stats': rt.stats()}), flush=True)
rt.close()
dist.destroy_process_group()
