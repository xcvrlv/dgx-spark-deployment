#!/usr/bin/env python3
"""Read real FP4 checkpoint rows through the native reader; no model load.

Reports warm-device O_DIRECT random reads, not an inference speed prediction.
Verifies sampled packed weight AND scale bytes against ordinary file reads.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import struct
import time


def source(model, index, key):
    path = model / index[key]
    with path.open('rb') as stream:
        length = struct.unpack('<Q',stream.read(8))[0]
        assert length < 16*2**20
        tensor = json.loads(stream.read(length))[key]
    return path, 8+length+tensor['data_offsets'][0], tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rank',type=int,choices=range(4),default=0)
    parser.add_argument('--model',type=Path,default=Path('/model'))
    parser.add_argument('--blocks',nargs='+',type=int,choices=[512,4096],default=[4096,512])
    parser.add_argument('--tokens',nargs='+',type=int,default=[1,6,48,512,4096])
    parser.add_argument('--repeats',type=int,default=5)
    args=parser.parse_args()
    assert args.repeats > 0 and min(args.tokens)>0 and max(args.tokens)<=4096
    import torch
    from b12x.sequence._shared.disk_table import DiskRowCache
    index=json.loads((args.model/'model.safetensors.index.json').read_text())['weight_map']
    os.environ['DS41_DISK_LOG_EVERY']='0'
    for block in args.blocks:
        os.environ['DS41_DISK_BLOCK_BYTES']=str(block)
        for layer in [1,14]:
            prefix=f'layers.{layer}.engram.embed'
            weight,woff,wt=source(args.model,index,prefix+'.weight')
            scale,soff,st=source(args.model,index,prefix+'.scale')
            rows=wt['shape'][0]
            assert wt['dtype']=='U8' and wt['shape']==[rows,128]
            assert st['dtype'] in ('U8','F8_E8M0') and st['shape']==[rows,8]
            width=(rows+3)//4
            start,end=args.rank*width,min((args.rank+1)*width,rows)
            cache=DiskRowCache(device='cuda:0',max_lookups=max(args.tokens)*24,
                table_rows=rows,shard_rows=rows,shard_start=start,shard_end=end,
                weight_row_bytes=128,scale_row_bytes=8,queue_depth=64)
            cache.add_shard(0,str(weight),woff)
            cache.add_shard(0,str(scale),soff,scale=True)
            cache.freeze()
            generator=torch.Generator().manual_seed(4100+layer)
            for nt in args.tokens:
                measurements=[]
                for repeat in range(args.repeats+1):
                    host_ids=torch.randint(rows,(nt*24,),generator=generator,dtype=torch.int64)
                    # Always exercise local ownership and verify a real local row.
                    host_ids[0]=start
                    ids=host_ids.cuda()
                    torch.cuda.synchronize()
                    begun=time.perf_counter()
                    with cache.transaction():
                        cache.read_rows(ids,len(host_ids))
                    torch.cuda.synchronize()
                    elapsed=time.perf_counter()-begun
                    stats=cache.stats()
                    if repeat:
                        measurements.append(dict(wall_ms=elapsed*1000,**stats))
                    for i,row in enumerate(host_ids[:16].tolist()):
                        for path,offset,size,buffer in [(weight,woff,128,cache.weight_host),
                                                       (scale,soff,8,cache.scale_host)]:
                            if start<=row<end:
                                with path.open('rb',buffering=0) as stream:
                                    stream.seek(offset+row*size)
                                    expected=stream.read(size)
                            else:
                                expected=bytes(size)
                            assert bytes(buffer[i].numpy())==expected, (layer,block,row,size)
                result=dict(rank=args.rank,layer=layer,block_bytes=block,tokens=nt,
                    median_wall_ms=statistics.median(x['wall_ms'] for x in measurements),
                    median_native_ms=1000*statistics.median(x['execution_seconds'] for x in measurements),
                    read_bytes=statistics.median(x['read_bytes'] for x in measurements),
                    requested_bytes=statistics.median(x['requested_bytes'] for x in measurements),
                    read_calls=statistics.median(x['read_calls'] for x in measurements),
                    checked_bytes=True)
                result['amplification']=result['read_bytes']/max(1,result['requested_bytes'])
                print('DS41_DISK_BENCH '+json.dumps(result),flush=True)
            del cache
            gc.collect()
    print('DS41_DISK_BENCH_PASS')


if __name__=='__main__':
    main()
