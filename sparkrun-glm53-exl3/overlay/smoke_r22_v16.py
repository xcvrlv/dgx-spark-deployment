#!/usr/bin/env python3
"""v16 pinned-source, CUDA lifetime and optional four-node transport checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS


def ckv_gpu():
    """Exercise actual cache gather with reversed pages and active/capacity gaps."""
    import torch
    from vllm.v1.attention.backends.mla import b12x_mla_sparse as impl
    original = impl.get_dcp_group
    world, page, record, capacity = 4, 64, 656, 256
    cases = 0
    try:
        for counts in ((0, 1, 63, 65), (129, 128, 127, 126)):
            padded = ((max(counts) + page - 1) // page) * page
            expected = torch.zeros(world * padded, record, device='cuda', dtype=torch.uint8)
            caches = []
            for rank, count in enumerate(counts):
                cache = torch.randint(0, 256, (capacity//page, page, record), device='cuda', dtype=torch.uint8)
                caches.append(cache.flip(0).contiguous())
                expected[rank*padded:rank*padded+count].copy_(cache.view(-1,record)[:count])
            for rank, count in enumerate(counts):
                table = torch.arange(capacity//page-1, -1, -1, device='cuda', dtype=torch.int32)[None]
                meta = NS(num_actual_tokens=32, num_reqs=1, block_table=table,
                          dcp_local_cu_seq_lens=torch.tensor([0,count],device='cuda',dtype=torch.int32),
                          dcp_local_total_tokens=count, dcp_padded_total_tokens=padded)
                for inplace in (False, True):
                    gathered = torch.full((world*capacity,record), 239, device='cuda',dtype=torch.uint8)
                    local = torch.empty((0 if inplace else capacity, record),device='cuda',dtype=torch.uint8)
                    driver = NS(_v16_ckv_inplace=inplace, _ckv_local_capacity=capacity,
                                _cache_record_bytes=record, _kernel_page_size=page, dcp_world_size=world,
                                uses_full_ckv_dcp=lambda *_: True)
                    def gather(out, inp):
                        if inplace:
                            assert inp.data_ptr() == out.data_ptr() + rank*padded*record
                        torch.testing.assert_close(inp, expected[rank*padded:(rank+1)*padded].flatten(), rtol=0,atol=0)
                        out.copy_(expected.flatten())
                    impl.get_dcp_group = lambda: NS(world_size=world,rank_in_group=rank,
                        device_communicator=NS(pynccl_comm=NS(disabled=False,all_gather=gather)))
                    out = impl.B12xMLASparseImpl._gather_full_ckv(driver,caches[rank],meta,local,gathered)
                    torch.testing.assert_close(out.view(-1,record)[:world*padded],expected,rtol=0,atol=0)
                    assert bool((gathered[world*padded:] == 239).all())
                    cases += 1
    finally:
        impl.get_dcp_group = original
    return dict(ckv_inplace_cases=cases)


def indexer_reference_ids(indices, scores, world, interleave):
    """CPU oracle for this fixture's identical local candidates on every rank.

    Select by descending score, then ascending global ID. Output order from
    the GPU's atomic append is unspecified, so return canonical ID order.
    """
    import numpy as np
    indices, scores = np.asarray(indices), np.asarray(scores)
    valid = indices >= 0
    safe = np.maximum(indices, 0).astype(np.int64)
    global_ids = np.concatenate([
        np.where(valid, (safe // interleave) * (world * interleave)
                 + rank * interleave + safe % interleave, -1)
        for rank in range(world)], axis=1)
    candidate_scores = np.tile(np.where(valid, scores, -np.inf), (1, world))
    # Invalid candidates always follow valid ones, including score=-inf.
    order = np.lexsort((global_ids, -candidate_scores, global_ids < 0), axis=1)
    selected = np.take_along_axis(global_ids, order[:, :indices.shape[1]], axis=1)
    return np.sort(selected, axis=1).astype(np.int32)


def assert_indexer_multiset(actual, reference, context):
    import numpy as np
    # Sorting retains multiplicity and padding; a set comparison would hide
    # duplicate IDs or lost candidates. Keep every row independent and exact.
    np.testing.assert_array_equal(np.sort(actual, axis=1), reference, err_msg=context)


def indexer_gpu(group=None):
    """Real Triton packing + CuTe stable merge, workspace reuse and replay."""
    import torch
    from vllm.v1.attention.backends.mla import b12x_indexer as impl
    from vllm.v1.worker.workspace import WorkspaceManager
    from vllm.distributed.device_communicators import gb10_dcp
    from smoke_r22_v14 import ms
    original = impl.get_dcp_group, impl.current_workspace_manager, impl._V16_MERGE_ROWS, gb10_dcp.ENABLED
    rank, world = (group.rank_in_group, group.world_size) if group else (0, 4)
    manager = WorkspaceManager(torch.device('cuda',0), num_lanes=2)
    manager.reserve_all(*impl._v16_merge_specs(256,2048,world))
    manager.lock()
    workspace_ptrs = [x.data_ptr() for x in manager._current_workspaces]
    # Single GPU simulation supplies identical scores from four ranks and
    # converts rank-zero IDs to each rank's global IDs. Real test uses RoCE.
    def simulated_gather(packed, dim):
        assert dim == 1
        peers = []
        for peer in range(world):
            x = packed.clone()
            x[...,1] = torch.where(x[...,1] >= 0, x[...,1]+peer*interleave, x[...,1])
            peers.append(x)
        return torch.cat(peers, dim=1)
    results = []
    try:
        impl.current_workspace_manager = lambda: manager
        impl.get_dcp_group = lambda: group or NS(all_gather=simulated_gather)
        gb10_dcp.ENABLED = True
        for topk, rows, interleave in ((512,1,1),(1024,8,16),(2048,257,1),(2048,513,16)):
            # Unique local indices, repeated scores, invalid tails and sparse holes.
            ids = torch.arange(topk,device='cuda',dtype=torch.int32)[None].expand(rows,-1).clone()
            ids[:,13::29] = -1
            ids[-1,:] = -1
            scores = (torch.arange(topk,device='cuda',dtype=torch.float32)%17)[None].expand(rows,-1).contiguous()
            expected, actual = torch.empty_like(ids), torch.empty_like(ids)
            def run(target, optimized):
                impl._V16_MERGE_ROWS = 256 if optimized else 0
                target.copy_(ids)
                impl._merge_dcp_topk(target,scores,rank,world,interleave)
            def compare(phase):
                reference = indexer_reference_ids(ids.cpu().numpy(), scores.cpu().numpy(), world, interleave)
                context = f'{phase}: rows={rows}, topk={topk}, interleave={interleave}, rank={rank}'
                assert_indexer_multiset(expected.cpu().numpy(), reference, 'legacy '+context)
                assert_indexer_multiset(actual.cpu().numpy(), reference, 'v16 '+context)
            run(expected,False)
            run(actual,True)
            torch.cuda.synchronize()
            compare('eager')
            graph = torch.cuda.CUDAGraph()
            if group:
                with group.device_communicator.b12x_ar_comm.capture():
                    with torch.cuda.graph(graph): run(actual,True)
            else:
                with torch.cuda.graph(graph): run(actual,True)
            for _ in range(3):
                scores.add_(torch.arange(topk,device='cuda',dtype=torch.float32)[None]%3)
                run(expected,False)
                graph.replay()
                torch.cuda.synchronize()
                compare('graph replay')
            if group is None:
                results.append(dict(rows=rows,topk=topk,legacy_ms=ms(lambda:run(expected,False)),
                                    pooled_ms=ms(lambda:run(actual,True))))
        assert workspace_ptrs == [x.data_ptr() for x in manager._current_workspaces]
    finally:
        impl.get_dcp_group, impl.current_workspace_manager, impl._V16_MERGE_ROWS, gb10_dcp.ENABLED = original
    return dict(indexer_merge_cases=4, indexer_timings=results, merge_workspace_bytes=256*2048*8*5)


def distributed_gpu():
    import torch
    import torch.distributed as dist
    from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
    from vllm.distributed.device_communicators import gb10_dcp
    from vllm.v1.attention.backends.mla import b12x_mla_sparse as attention
    os.environ['VLLM_ENABLE_ROCE_ALLREDUCE'] = '1'
    os.environ['VLLM_ROCE_DCP_ENABLE'] = '1'
    dist.init_process_group('nccl')
    torch.cuda.set_device(0)
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 4
    cpu = dist.new_group(backend='gloo')
    comm = CudaCommunicator(cpu,torch.device('cuda',0),dist.group.WORLD,unique_name='dcp:v16-smoke')
    assert comm.use_roce_allreduce and comm.b12x_ar_comm is not None and not comm.b12x_ar_comm.disabled
    group = NS(world_size=world,rank_in_group=rank,device_communicator=comm,
               all_gather=lambda x,dim=0:comm.all_gather(x,dim))
    old = gb10_dcp.ENABLED
    gb10_dcp.ENABLED = True
    try:
        runtime = comm.b12x_ar_comm._runtime
        # Alternating sizes exercises both slots, padding and sequence reuse.
        for size in (16, 64*656, 4096*656, 16, 8192*656):
            output = torch.empty(world*size,device='cuda',dtype=torch.uint8)
            inp = output[rank*size:(rank+1)*size]
            expected = torch.cat([torch.full((size,),r+1,device='cuda',dtype=torch.uint8) for r in range(world)])
            def step():
                inp.fill_(rank+1)
                attention._dcp_all_gather_current_stream(group,inp,output)
            for _ in range(3): step()
            torch.cuda.synchronize()
            torch.testing.assert_close(output,expected,rtol=0,atol=0)
            with comm.b12x_ar_comm.capture():
                with torch.cuda.graph(graph := torch.cuda.CUDAGraph()): step()
            for _ in range(5): graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(output,expected,rtol=0,atol=0)
            runtime.check_health()
        result = indexer_gpu(group)
        assert runtime._gather_buffers is None, 'aligned DCP unexpectedly allocated padded scratch'
        runtime.check_health()
        dist.barrier()
        return dict(result, roce_ckv_inplace_replay='passed',rank=rank)
    finally:
        gb10_dcp.ENABLED = old
        comm.destroy()
        dist.destroy_process_group()


def main():
    import b12x
    import vllm
    from patch_r22_v16 import patch, VERSION, OUTPUTS, PROXY, RUNTIME
    from patch_r22_v15 import OUTPUTS as v15
    from patch_r22_v14 import OUTPUTS as v14
    from patch_r22_v13 import OUTPUTS as v13
    from patch_r22_v11 import OUTPUT_HASHES as v11
    from smoke_r22_v14 import load_extension
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',action='store_true')
    parser.add_argument('--distributed',action='store_true')
    args = parser.parse_args()
    b,v = Path(b12x.__file__).parent,Path(vllm.__file__).parent
    patch(b,v,check=True)
    # Latest hash wins; old smoke main() functions reject intentional overlays.
    hashes = {**v11, **v13, **v14, **v15, **OUTPUTS}
    for name,expected in hashes.items():
        root = b if name.startswith(('moe/','comm/')) else v
        assert hashlib.sha256((root/name).read_text(encoding='utf-8').encode()).hexdigest() == expected,name
    from b12x.comm.roce._proxy import load
    result = dict(overlay=VERSION,proxy=str(load()._name),exl3_extension=load_extension().__file__)
    if args.gpu:
        import torch
        torch.backends.cuda.matmul.allow_tf32 = False
        from smoke_r22_v12 import check_gpu
        from smoke_r22_v13 import gpu as v13_gpu
        from smoke_r22_v14 import argmax_gpu,rotations_gpu,mixed_gpu
        from smoke_r22_v15 import sigmoid_gpu,skinny_gpu,mixed_activation_gpu
        from probe_r22_v14_rotations import RotationProbe
        RotationProbe.gb10_compact_input = False
        result.update(inherited_v12=check_gpu(),inherited_v13=v13_gpu())
        for test in (argmax_gpu,rotations_gpu,mixed_gpu,sigmoid_gpu,skinny_gpu,mixed_activation_gpu,ckv_gpu,indexer_gpu):
            result.update(test())
    if args.distributed:
        result.update(distributed_gpu())
        from smoke_r22_v14 import distributed_gpu as mtp_gpu
        result.update(mtp_gpu())
    print(json.dumps(result,sort_keys=True))


if __name__ == '__main__':
    main()
