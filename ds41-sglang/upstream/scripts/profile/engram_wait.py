import gzip,json,sys,statistics
for path in sys.argv[1:]:
    with gzip.open(path,'rt') as f: data=json.load(f)
    kern=sorted([e for e in data['traceEvents'] if e.get('ph')=='X' and e.get('cat')=='kernel'],key=lambda e:e['ts'])
    eng=[];other=[]
    for i,k in enumerate(kern):
        if 'nccl' not in k['name'].lower(): continue
        # look back up to 6 kernels for an engram gather
        prev=[kern[j]['name'] for j in range(max(0,i-6),i)]
        (eng if any('_engram_gather' in n for n in prev) else other).append(k['dur'])
    tot=sum(eng)+sum(other)
    print(f"{path}: engram-following NCCL: n={len(eng)} total={sum(eng)/1000:.1f} ms mean={statistics.mean(eng) if eng else 0:.0f}us median={statistics.median(eng) if eng else 0:.0f}us max={max(eng) if eng else 0:.0f}us | other NCCL n={len(other)} total={sum(other)/1000:.1f} ms | engram share {100*sum(eng)/tot:.0f}%")
