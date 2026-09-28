"""Run inside the performance image on an idle Spark; no checkpoint needed."""
import gc

import pytest
import torch


@pytest.mark.parametrize('dtype', [torch.int32, torch.int64, torch.float32])
@pytest.mark.parametrize('rows', [None, 3, 0])
def test_host_snapshot_survives_delayed_copies_and_allocator_reuse(monkeypatch, dtype, rows):
    from vllm.v1.utils import CpuGpuBuffer

    monkeypatch.setenv('DS41_H2D_STAGING', '1')
    buffer = CpuGpuBuffer(8, 16, dtype=dtype, device=torch.device('cuda'))
    assert buffer.cpu.is_pinned(), 'Qualification requires the serving pinned-memory path'
    expected = torch.arange(128, dtype=dtype).reshape(8, 16)
    buffer.cpu.copy_(expected)
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        # Keep both DMA reads queued while the scheduler rewrites the host rows.
        torch.cuda._sleep(100_000_000)
        first = buffer.copy_to_gpu(rows).clone()
        buffer.cpu.add_(1000)
        second = buffer.copy_to_gpu(rows).clone()
        buffer.cpu.add_(1000)
        # Freed staging must not be recycled before its DMA event completes.
        for _ in range(32):
            torch.empty_like(buffer.cpu, pin_memory=True).fill_(-100)
    stream.synchronize()
    sliced = expected if rows is None else expected[:rows]
    torch.testing.assert_close(first.cpu(), sliced, rtol=0, atol=0)
    torch.testing.assert_close(second.cpu(), sliced + 1000, rtol=0, atol=0)


def test_native_gate_releases_work_before_cyclic_finalizers_resume(monkeypatch):
    from b12x.preparation._measurement import _StreamGate

    monkeypatch.setenv('DS41_DEFER_AUTOTUNE_GC', '1')
    stream = torch.cuda.Stream()
    output = torch.zeros(1, device='cuda')
    torch.cuda.synchronize()
    gate = _StreamGate()
    enabled, thresholds = gc.isenabled(), gc.get_threshold()
    finalized = []

    class Cycle:
        def __init__(self):
            self.cycle = self

        def __del__(self):
            # A CUDA module destructor can wait for queued work in the same way.
            released = gate.flag.value == gate.sequence
            stream.synchronize()
            finalized.append(released)

    try:
        gc.enable()
        gc.collect()
        gc.set_threshold(5, 5, 5)
        with torch.cuda.stream(stream), gate.hold(stream):
            assert not gc.isenabled()
            output.add_(7)
            cycle = Cycle()
            del cycle
            garbage = [[i] for i in range(200)]
            assert not finalized
            del garbage
        assert gc.isenabled()
        gc.collect()
        assert finalized == [True]
        stream.synchronize()
        assert output.item() == 7
    finally:
        gate.close()
        gc.set_threshold(*thresholds)
        gc.enable() if enabled else gc.disable()


@pytest.mark.parametrize('slots', [1, 4, 31, 64])
@pytest.mark.parametrize('coalesce', ['0', '1'])
def test_native_moe_barrier_arena_clears_on_every_graph_replay(monkeypatch, slots, coalesce):
    from b12x.moe.fused_moe.ds41_barriers import clear_barriers

    monkeypatch.setenv('DS41_MOE_COALESCE_BARRIERS', coalesce)
    start = 4
    epoch_start = ((start+slots+3)//4)*4
    end = epoch_start+slots
    arena = torch.full((end+4,), 73, dtype=torch.int32, device='cuda')
    count, epoch = arena[start:start+slots], arena[epoch_start:end]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            clear_barriers(count, epoch)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            clear_barriers(count, epoch)
        for value in (19, -4, 103):
            arena.fill_(value)
            graph.replay()
            stream.synchronize()
            assert not count.count_nonzero().item()
            assert not epoch.count_nonzero().item()
            assert arena[:start].tolist() == [value]*start
            assert arena[end:].tolist() == [value]*4
            padding = 0 if coalesce == '1' else value
            assert arena[start+slots:epoch_start].tolist() == [padding]*(epoch_start-start-slots)
