"""Verify v17, then run the complete inherited v16 smoke entrypoint."""
from pathlib import Path
import json
import os
import sys


def reclaim_gpu():
    import torch
    from vllm.v1.worker.gb10_startup_memory import reclaim_startup_memory
    x = torch.ones(256,device='cuda')
    out = torch.empty_like(x)
    pointers = x.data_ptr(), out.data_ptr()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.add(x,1,out=out)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):
        torch.add(x,1,out=out)
    original = os.environ.get('VLLM_GB10_STARTUP_RECLAIM')
    try:
        os.environ['VLLM_GB10_STARTUP_RECLAIM'] = '1'
        report = reclaim_startup_memory('smoke-live-graph',x.device)
        assert report is not None
        x.fill_(2)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out,torch.full_like(out,3),rtol=0,atol=0)
        assert pointers == (x.data_ptr(),out.data_ptr())
    finally:
        if original is None:
            os.environ.pop('VLLM_GB10_STARTUP_RECLAIM',None)
        else:
            os.environ['VLLM_GB10_STARTUP_RECLAIM'] = original
    return {'live_graph_after_reclamation':'passed','memory':report}


def main(source_overrides=None):
    import vllm
    from patch_r22_v17 import patch, VERSION
    patch(Path(vllm.__file__).parent,check=True)
    print(json.dumps({'startup_overlay':VERSION}),flush=True)
    from smoke_r22_v16 import main as inherited
    if source_overrides is None:
        inherited()
    else:
        inherited(source_overrides=source_overrides)
    if '--gpu' in sys.argv:
        print(json.dumps(reclaim_gpu(),sort_keys=True),flush=True)


if __name__ == '__main__':
    main()
