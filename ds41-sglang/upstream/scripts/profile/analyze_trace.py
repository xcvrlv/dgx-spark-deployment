import gzip, json, sys, collections, re
path=sys.argv[1]
opener = gzip.open if path.endswith('.gz') else open
with opener(path,'rt') as f: data=json.load(f)
ev=[e for e in data['traceEvents'] if e.get('ph')=='X']
print("events:",len(ev))
cats=collections.Counter(e.get('cat') for e in ev); print("cats:",dict(cats))
gpu=[e for e in ev if e.get('cat') in ('kernel','gpu_memcpy','gpu_memset','gpu_user_annotation')]
kern=[e for e in ev if e.get('cat') in ('kernel','gpu_memcpy','gpu_memset')]
kern.sort(key=lambda e:e['ts'])
cpu=[e for e in ev if e.get('cat') in ('cpu_op','cuda_runtime','user_annotation','python_function','cuda_driver')]
t0=kern[0]['ts']; t1=max(e['ts']+e['dur'] for e in kern)
print(f"GPU span: {(t1-t0)/1000:.1f} ms")
# step boundaries: use cudaGraphLaunch runtime calls or user annotations
launches=[e for e in ev if e.get('cat')=='cuda_runtime' and 'GraphLaunch' in e['name']]
print("cudaGraphLaunch count:",len(launches))
ann=[e for e in ev if e.get('cat') in ('user_annotation','gpu_user_annotation')]
annc=collections.Counter(e['name'] for e in ann)
print("annotations:",annc.most_common(25))
def cat_of(n):
    l=n.lower()
    if 'nccl' in l: return 'nccl'
    if 'memcpy' in l or 'memset' in l: return 'memcpy/set'
    if 'engram' in l or 'hash' in l: return 'engram'
    if any(k in l for k in ('moe','expert','grouped','group_gemm','fp4','mxfp4','trtllm','topk','router','routing','silu','swiglu')): return 'moe'
    if any(k in l for k in ('attn','attention','mla','flash','indexer','index','sparse','paged','decode_kernel','rope','rotary','kv_cache','set_kv')): return 'attention'
    if any(k in l for k in ('gemm','matmul','cutlass','fp8','blockwise','scaled_mm','wgmma','bmm','sm90','sm100','sm120','cublas','gemv')): return 'gemm'
    if any(k in l for k in ('norm','rms')): return 'norm'
    if any(k in l for k in ('elementwise','vectorized','copy_','fill','cat','index_select','gather','scatter','arange','reduce','softmax','sort','cumsum','where','argmax','sampl','triton_')): return 'elementwise/other'
    return 'other'
# union busy time
busy=0; cur_s=None; cur_e=None
for e in kern:
    s,en=e['ts'],e['ts']+e['dur']
    if cur_e is None or s>cur_e:
        if cur_e is not None: busy+=cur_e-cur_s
        cur_s,cur_e=s,en
    else: cur_e=max(cur_e,en)
busy+=cur_e-cur_s
print(f"GPU busy (union): {busy/1000:.1f} ms = {100*busy/(t1-t0):.1f}% of span")
bycat=collections.defaultdict(float); cnt=collections.Counter()
byname=collections.defaultdict(float); cntn=collections.Counter()
for e in kern:
    c=cat_of(e['name']); bycat[c]+=e['dur']; cnt[c]+=1; byname[e['name']]+=e['dur']; cntn[e['name']]+=1
print("\n== GPU time by category (sum of kernel durations, ms; count) ==")
for c,v in sorted(bycat.items(),key=lambda kv:-kv[1]): print(f"  {c:18s} {v/1000:9.1f} ms  n={cnt[c]}")
print("\n== top 40 kernels by total time ==")
for n,v in sorted(byname.items(),key=lambda kv:-kv[1])[:40]: print(f"  {v/1000:8.1f} ms n={cntn[n]:5d} avg={v/cntn[n]:8.1f}us  {n[:110]}")
# gaps
gaps=[]; prev_e=kern[0]['ts']+kern[0]['dur']; prev_n=kern[0]['name']
for e in kern[1:]:
    s=e['ts']
    if s-prev_e>300: gaps.append((s-prev_e,prev_e,prev_n,e['name']))
    prev_e=max(prev_e,s+e['dur']); prev_n=e['name'] if s+e['dur']>=prev_e else prev_n
print(f"\n== GPU idle gaps >0.3ms: n={len(gaps)} total={sum(g[0] for g in gaps)/1000:.1f} ms ==")
gb=collections.defaultdict(lambda:[0,0.0])
for g in gaps:
    k=(g[2][:60],g[3][:60]); gb[k][0]+=1; gb[k][1]+=g[0]
for k,v in sorted(gb.items(),key=lambda kv:-kv[1][1])[:15]: print(f"  {v[1]/1000:8.1f} ms n={v[0]:4d} after [{k[0]}] before [{k[1]}]")
# NCCL kernel stats
nk=[e for e in kern if 'nccl' in e['name'].lower()]
if nk:
    ds=sorted(e['dur'] for e in nk)
    print(f"\n== NCCL kernels: n={len(nk)} total={sum(ds)/1000:.1f} ms median={ds[len(ds)//2]:.0f}us p90={ds[int(len(ds)*.9)]:.0f}us max={ds[-1]:.0f}us")
    print("  names:",collections.Counter(e['name'][:80] for e in nk).most_common(6))
# CPU-side top ops
cpuop=collections.defaultdict(float); cpucnt=collections.Counter()
for e in cpu:
    cpuop[(e['cat'],e['name'])]+=e['dur']; cpucnt[(e['cat'],e['name'])]+=1
print("\n== top 25 CPU-side events by total time ==")
for k,v in sorted(cpuop.items(),key=lambda kv:-kv[1])[:25]: print(f"  {v/1000:8.1f} ms n={cpucnt[k]:5d}  {k[0]}:{k[1][:90]}")
