"""Four-node fail-closed RoCEnante/NCCL comparison; invoked by fleet.py fabric."""
import json
import os
from datetime import timedelta
import torch
import torch.distributed as dist
from b12x.comm import roce
from vllm.distributed.device_communicators.b12x_roce_all_reduce import B12xRoceAllReduce


def main():
    # Spawned compiler workers re-import this module; the __main__ guard keeps
    # CUDA initialization and the comparison out of them (CUDA is hidden from
    # compiler children, so their re-import must not run the check).
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
    # Use the same preparation provider as serving, not removed runtime.prepare().
    from b12x.preparation import PreparationSession
    from vllm.utils.b12x import B12xWorkload
    workload = B12xWorkload(stage='weights', token_counts=(1,), fixed_token_counts=(1,),
                            output_dtype=torch.bfloat16, max_tokens=16, max_seqs=16,
                            max_model_len=1048576, eager_only=True)
    units = adapter.get_b12x_preparation_units(adapter, workload)
    assert units, 'RoCE preparation provider returned no units'
    # Compiler processes compete with GPU allocations on GB10 unified memory.
    session = PreparationSession(device=torch.device('cuda', 0), autotune=False,
                                 compile_workers=int(os.environ.get('B12X_COMPILE_WORKERS', '4')))
    # The RoCE request declares a collective, so priming needs the same
    # world-coordinated rounds as serving: the coordinator authorizes it only
    # when every participant rank reported ready, and every rank stays in the
    # exchange until all jobs are done (a done job keeps advancing safely, so
    # rank drift in the compile/drain steps cannot strand the exchange).
    from types import SimpleNamespace
    from vllm.v1.worker.b12x_startup import B12xPreparationCoordinator

    def exchange():
        # Karmic coordinates preparation over a store-backed control channel.
        # Use torchrun's existing default store, scoped by the coordinator.
        store = dist.distributed_c10d._get_default_store()
        def all_gather_obj(payload):
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, payload, group=dist.group.WORLD)
            return gathered
        return SimpleNamespace(ranks=tuple(range(dist.get_world_size())),
                               tcp_store_group=SimpleNamespace(store=store,
                                                               all_gather_obj=all_gather_obj))

    coordinator = B12xPreparationCoordinator(
        session,
        [(tuple(request for unit in units for request in unit.requests), False)],
        global_rank=rank, world_group=exchange(),
    )
    outcome = coordinator.status()
    while not outcome['done']:
        outcome = coordinator.advance()
        assert outcome['error'] is None, f'preparation failed: {outcome["error"]}'
    plan = adapter._prepared_plan()
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
            gathered = rt.all_gather(x, dim=0, plan=plan)
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
        rt.all_reduce(x, out=reduced, plan=plan)
        rt.all_gather(x, dim=0, out=gathered, plan=plan)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream), rt.capture(stream=stream):
        rt.all_reduce(x, out=reduced, plan=plan)
        rt.all_gather(x, dim=0, out=gathered, plan=plan)
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
    session.close()
    rt.close()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
