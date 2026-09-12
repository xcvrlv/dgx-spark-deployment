import gzip, json, sys, collections, statistics
path=sys.argv[1]; dump_step=int(sys.argv[2]) if len(sys.argv)>2 else -1
with gzip.open(path,'rt') as f: data=json.load(f)
ev=[e for e in data['traceEvents'] if e.get('ph')=='X']
kern=sorted([e for e in ev if e.get('cat') in ('kernel','gpu_memcpy','gpu_memset')],key=lambda e:e['ts'])
launches=sorted([e for e in ev if e.get('cat')=='cuda_runtime' and 'GraphLaunch' in e['name']],key=lambda e:e['ts'])
# steps: pair consecutive graph launches (2 per decode step). Use the GPU kernels between launch[i] start and launch[i+2] start.
starts=[l['ts'] for l in launches[0::2]]
ends=starts[1:]+[max(e['ts']+e['dur'] for e in kern)]
def cat_of(n):
    l=n.lower()
    if 'nccl' in l: return 'nccl'
    if 'memcpy' in l or 'memset' in l: return 'memcpy'
    if 'w8a8_block_fp8' in l: return 'fp8_triton_gemm'
    if 'groupproblemshape' in l: return 'moe_grouped_gemm'
    if 'cutlass_80_wmma' in l or 'nvjet' in l or 'gemm' in l and 'tiny' not in l: return 'bf16_gemm'
    if 'tiny_n_gemm' in l: return 'router_tiny_gemm'
    if 'hc_' in l or 'mhc' in l: return 'hyperconn'
    if 'sparse_mla' in l or 'mqa_logits' in l or 'topk' in l or 'page_' in l or 'rope' in l or 'k_norm' in l: return 'attention'
    if 'engram' in l or 'hash' in l: return 'engram'
    if 'tensorrt_llm' in l or 'moe' in l or 'expert' in l or 'router' in l or 'silu' in l or 'quantize_with_block' in l: return 'moe_aux'
    if 'quant' in l: return 'quant'
    if 'norm' in l: return 'norm'
    return 'other'
per=[]
for i,(s,e) in enumerate(zip(starts,ends)):
    ks=[k for k in kern if s<=k['ts']<e]
    if not ks: continue
    d=collections.defaultdict(float); c=collections.Counter()
    for k in ks: d[cat_of(k['name'])]+=k['dur']; c[cat_of(k['name'])]+=1
    busy=0;cs=None;ce=None
    for k in ks:
        a,b=k['ts'],k['ts']+k['dur']
        if ce is None or a>ce:
            if ce is not None: busy+=ce-cs
            cs,ce=a,b
        else: ce=max(ce,b)
    busy+=ce-cs
    per.append(dict(i=i,wall=e-s,busy=busy,cats=d,cnt=c,n=len(ks)))
per=per[1:-1]  # drop first (may include prefill tail) and last (truncated)
print(f"decode steps analysed: {len(per)}")
walls=[p['wall']/1000 for p in per]; print(f"step wall ms: mean={statistics.mean(walls):.1f} median={statistics.median(walls):.1f} min={min(walls):.1f} max={max(walls):.1f}")
print(f"GPU busy per step: mean={statistics.mean(p['busy']/1000 for p in per):.1f} ms")
allc=collections.defaultdict(list)
for p in per:
    for k in set().union(*[q['cats'].keys() for q in per]): allc[k].append(p['cats'].get(k,0)/1000)
print("\n== per-step GPU time by category (ms): mean / median / max ; kernels per step ==")
tot=0
for k,v in sorted(allc.items(),key=lambda kv:-statistics.mean(kv[1])):
    n=statistics.mean(p['cnt'].get(k,0) for p in per); tot+=statistics.mean(v)
    print(f"  {k:18s} {statistics.mean(v):7.1f} / {statistics.median(v):7.1f} / {max(v):7.1f}   n={n:.0f}")
print(f"  {'SUM':18s} {tot:7.1f}")
# NCCL per-step
nc=[p['cats'].get('nccl',0)/1000 for p in per]
print(f"\nNCCL per step ms: mean={statistics.mean(nc):.1f} median={statistics.median(nc):.1f} min={min(nc):.1f} max={max(nc):.1f}")
nk=[k for k in kern if 'nccl' in k['name'].lower() and k['ts']>=starts[1]]
ds=sorted(k['dur'] for k in nk)
import bisect
print("NCCL kernel duration percentiles us: " + " ".join(f"p{q}={ds[min(len(ds)-1,int(len(ds)*q/100))]:.0f}" for q in (10,25,50,75,90,95,99)))
print(f"NCCL kernels <150us: {sum(1 for x in ds if x<150)/len(ds)*100:.0f}%  time share of >300us kernels: {sum(x for x in ds if x>300)/sum(ds)*100:.0f}%")
if dump_step>=0:
    p=per[dump_step]; s=starts[p['i']]; e=ends[p['i']]
    ks=[k for k in kern if s<=k['ts']<e]
    print(f"\n== kernel sequence for step {p['i']} (collapsed runs), wall={p['wall']/1000:.1f}ms ==")
    runs=[]
    for k in ks:
        nm=k['name'][:70]
        if runs and runs[-1][0]==nm: runs[-1][1]+=1; runs[-1][2]+=k['dur']
        else: runs.append([nm,1,k['dur'],k['ts']-s])
    for nm,n,d,off in runs[:400]:
        print(f"  +{off/1000:7.2f}ms  {d:8.0f}us x{n:<3d} {nm}")
