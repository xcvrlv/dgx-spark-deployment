"""Source preflight and executable CPU models of actual v16 dispatch/lifetimes."""
import ast
import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = Path(os.getenv('GLM53_V16_BASELINE', ROOT.parent/'tmp/exl3-v16/base'))
spec = importlib.util.spec_from_file_location('v16patch',ROOT/'overlay/patch_r22_v16.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


class Tensor:
    def __init__(self,a): self.a=np.asarray(a)
    @property
    def shape(self): return self.a.shape
    def numel(self): return self.a.size
    def stride(self,n): return self.a.strides[n]//self.a.itemsize
    def __getitem__(self,k): return Tensor(self.a[k])
    def flatten(self,start=0): return Tensor(self.a.reshape(*self.shape[:start],-1))
    def view(self,*shape): return Tensor(self.a.reshape(shape))
    def copy_(self,x): np.copyto(self.a,x.a);return self
    def zero_(self): self.a.fill(0);return self
    @property
    def device(self): return 'cpu'


def functions(source,names,ns):
    for node in ast.walk(ast.parse(source)):
        if isinstance(node,ast.FunctionDef) and node.name in names:
            node.decorator_list=[]
            exec('from __future__ import annotations\n'+ast.unparse(node),ns)
    return ns


class Tests(unittest.TestCase):
    def source(self,name,owner='vllm'):
        if not BASE.is_dir(): self.skipTest('set GLM53_V16_BASELINE to the exact v15 composition')
        return (BASE/owner/name).read_text(encoding='utf-8')

    def test_pinned_preflight_is_idempotent_and_atomic(self):
        self.source(p.ATTENTION)
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)/'source';shutil.copytree(BASE,dest)
            b,v=dest/'b12x',dest/'vllm'
            p.patch(b,v);p.patch(b,v);p.patch(b,v,check=True)
            (v/p.ATTENTION).write_text(self.source(p.ATTENTION),encoding='utf-8')
            (b/p.PROXY).write_text('// drift\n',encoding='utf-8')
            before={f:f.read_bytes() for f in dest.rglob('*') if f.is_file()}
            with self.assertRaises(RuntimeError): p.patch(b,v)
            self.assertEqual(before,{f:f.read_bytes() for f in dest.rglob('*') if f.is_file()})

    def test_sigmoid_has_one_final_return_and_preserves_fallback(self):
        source = p.kernel(self.source(p.KERNEL, 'b12x'))
        cls = next(n for n in ast.parse(source).body
                   if isinstance(n, ast.ClassDef) and n.name == 'W4A16FusedMoeKernel')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_sigmoid_f32')
        returns = [n for n in ast.walk(method) if isinstance(n, ast.Return)]
        self.assertEqual(returns, [method.body[-1]])
        # Python permits a new name in both arms; CuTe staged regions require
        # that name to exist before the region so it can carry the result out.
        branch = next(i for i, n in enumerate(method.body)
                      if isinstance(n, ast.If) and 'gb10_sigmoid' in ast.unparse(n.test))
        initializers = [n for n in method.body[:branch] if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == 'result' for t in n.targets)]
        self.assertEqual(len(initializers), 1)
        self.assertEqual(ast.unparse(initializers[0].value), 'cutlass.Float32(0.0)')
        reciprocal_calls, exponential_modes = [], []
        def reciprocal(value):
            reciprocal_calls.append(value)
            return np.float32(1) / value
        def exponential(value, *, fastmath):
            exponential_modes.append(fastmath)
            return np.exp(value)
        ns = functions(ast.unparse(method), {'_sigmoid_f32'}, {
            'cutlass': NS(Float32=np.float32, const_expr=bool),
            'cute': NS(math=NS(exp=exponential)), 'gb10_reciprocal': reciprocal})
        for fast_math in (False, True):
            for enabled in (False, True):
                for value in (-np.inf, -105., -90., -87., -20., 0., 20., np.inf, np.nan):
                    reciprocal_calls.clear()
                    with np.errstate(over='ignore', invalid='ignore'):
                        x = np.float32(value)
                        denominator = np.float32(1) + np.exp(-x)
                        expected = np.float32(1) / denominator
                        actual = ns['_sigmoid_f32'](NS(fast_math=fast_math, gb10_sigmoid=enabled), x)
                    np.testing.assert_equal(actual, expected)
                    eligible = enabled and denominator >= 1 and denominator < np.float32(2.**126)
                    self.assertEqual(len(reciprocal_calls), int(eligible))
                    self.assertEqual(exponential_modes[-1], fast_math)

    def test_ckv_inplace_actual_method_uses_active_rank_offset(self):
        group=[None];calls=[]
        def cache_gather(*,src_cache,dst,block_table,cu_seq_lens,batch_size):
            # Reverse physical pages and concatenate requests according to the
            # real gather contract, leaving padding to the method under test.
            for req in range(batch_size):
                start,end=map(int,cu_seq_lens.a[req:req+2])
                logical=src_cache.a[block_table.a[req]].reshape(-1,16)
                dst.a[start:end]=logical[:end-start]
        expected=[None]
        def allgather(g,inp,out):
            rank,padded,world=g.rank_in_group,g.padded,4
            np.testing.assert_equal(inp.a,expected[0][rank*padded:(rank+1)*padded].reshape(-1))
            if g.inplace:
                self.assertEqual(inp.a.ctypes.data-out.a.ctypes.data,rank*padded*16)
            out.a[:]=expected[0].reshape(-1);calls.append(rank)
        ns=functions(p.attention(self.source(p.ATTENTION)),{'_gather_full_ckv','_workspace_specs'},
            {'get_dcp_group':lambda:group[0], '_is_glm_next_ckv_source_layout':lambda *a,**k:True,
             '_dcp_all_gather_current_stream':allgather,'ops':NS(cp_gather_cache=cache_gather),
             'torch':NS(bfloat16='bf16',uint8='u8')})
        for counts in ((0,1,3,5),(9,8,7,6)):
            padded=(max(counts)+3)//4*4;capacity=16
            canonical=[np.arange(capacity*16,dtype=np.uint8).reshape(capacity,16)+rank for rank in range(4)]
            expected[0]=np.concatenate([np.pad(a[:n],((0,padded-n),(0,0))) for a,n in zip(canonical,counts)])
            for rank,count in enumerate(counts):
                for inplace in (False,True):
                    driver=NS(_v16_ckv_inplace=inplace,_ckv_local_capacity=capacity,_cache_record_bytes=16,
                              _kernel_page_size=4,dcp_world_size=4,uses_full_ckv_dcp=lambda *a:True,
                              _max_tokens=32,_q_head_dim=576,_scratch_nbytes=256)
                    group[0]=NS(rank_in_group=rank,padded=padded,inplace=inplace)
                    meta=NS(num_actual_tokens=32,num_reqs=1,block_table=Tensor([[3,2,1,0]]),
                            dcp_local_cu_seq_lens=Tensor([0,count]),dcp_local_total_tokens=count,dcp_padded_total_tokens=padded)
                    local=Tensor(np.empty((0 if inplace else capacity,16),np.uint8))
                    gathered=Tensor(np.full((4*capacity,16),251,np.uint8))
                    cache=Tensor(canonical[rank].reshape(4,4,16)[::-1].copy())
                    out=ns['_gather_full_ckv'](driver,cache,meta,local,gathered)
                    np.testing.assert_equal(out.a.reshape(-1,16)[:4*padded],expected[0])
                    self.assertTrue((gathered.a[4*padded:]==251).all())
                    specs=ns['_workspace_specs'](driver,None,input_num_heads=32,include_ckv=True)
                    self.assertEqual(specs[2][0],(0 if inplace else capacity,16))
        self.assertEqual(len(calls),16)

    def test_merge_reuses_bounded_arena_and_preserves_chunk_order(self):
        specs_seen=[];alloc=[];direct=[True];interleave=[1]
        def empty(shape,**kw): alloc.append(shape);return Tensor(np.empty(shape,np.float32))
        def workspace(*specs):
            specs_seen.append(specs)
            n=sum(np.prod(shape) for shape,dtype in specs)
            arena=np.empty(n,np.float32);out=[];offset=0
            for shape,dtype in specs:
                count=int(np.prod(shape));out.append(Tensor(arena[offset:offset+count].reshape(shape)));offset+=count
            return out
        def gather(inp,dim):
            x=inp.a.reshape(inp.shape[0],-1,2)
            peers=[]
            for rank in range(4):
                a=x.copy();a[...,1]=np.where(a[...,1]>=0,a[...,1]+rank*interleave[0],a[...,1]);peers.append(a)
            return Tensor(np.concatenate(peers,axis=1).reshape(inp.shape[0],-1) if len(inp.shape)==2 else np.concatenate(peers,axis=1))
        def into(g,inp,out,dim):
            if not direct[0]: return False
            out.copy_(gather(inp,dim));return True
        class Pack:
            def __getitem__(self,grid):
                def run(ids,scores,out,is_,ss,ps,pc,rank,world,il,k,block,**kw):
                    idx=ids.a;valid=idx>=0;safe=np.maximum(idx,0)
                    out.a[...,0]=np.where(valid,scores.a,-np.inf)
                    out.a[...,1]=np.where(valid,(safe//il)*(world*il)+rank*il+safe%il,-1)
                return run
        def reducer(g,topk,out):
            order=np.argsort(-g.a[...,0],axis=1,kind='stable')[:,:topk]
            out.a[:]=np.take_along_axis(g.a[...,1],order,axis=1).astype(np.int32)
        ns=functions(p.indexer(self.source(p.INDEXER)),{'_merge_dcp_topk','_v16_merge_specs'},
            {'torch':NS(float32='f32',empty=empty),'triton':NS(cdiv=lambda a,b:(a+b-1)//b),
             '_pack_dcp_candidates_kernel':Pack(),'current_workspace_manager':lambda:NS(get_simultaneous=workspace),
             'get_dcp_group':lambda:NS(all_gather=gather),'try_roce_gather_into':into,'_V16_MERGE_ROWS':0})
        module='vllm.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl'
        with patch.dict(sys.modules,{module:NS(stable_topk_from_gathered_candidates_cutedsl=reducer)}):
            for rows,k,il in ((1,512,1),(257,1024,16),(513,2048,1)):
                interleave[0]=il
                ids=np.broadcast_to(np.arange(k,dtype=np.int32),(rows,k)).copy();ids[:,13::29]=-1
                scores=Tensor(np.broadcast_to(np.arange(k,dtype=np.float32)%17,(rows,k)).copy())
                expected=Tensor(ids.copy());ns['_V16_MERGE_ROWS']=0
                ns['_merge_dcp_topk'](expected,scores,0,4,il)
                for use_direct in (False,True):
                    direct[0]=use_direct;actual=Tensor(ids.copy());alloc.clear();ns['_V16_MERGE_ROWS']=256
                    ns['_merge_dcp_topk'](actual,scores,0,4,il)
                    np.testing.assert_equal(actual.a,expected.a)
                    self.assertEqual(alloc,[])
                    self.assertLessEqual(sum(np.prod(s)*4 for s,d in specs_seen[-1]),20*1024**2)

    def test_dispatch_is_rank_invariant_and_transport_errors_propagate(self):
        source=(ROOT/'overlay/gb10_dcp.py').read_text(encoding='utf-8')
        calls=[]
        runtime=NS(all_gather=lambda inp,**kw:calls.append((inp,kw)))
        adapter=NS(disabled=False,backend_name='B12X_ROCENANTE',should_all_gather=lambda *a:True,_runtime=runtime)
        group=NS(device_communicator=NS(use_roce_allreduce=True,b12x_ar_comm=adapter))
        ns=functions(source,{'try_roce_gather_into'},{'ENABLED':True,'logger':NS(info_once=lambda *a:None)})
        run=ns['try_roce_gather_into']
        inp,out=object(),object()
        self.assertTrue(run(group,inp,out));self.assertIs(calls[0][1]['out'],out)
        adapter.should_all_gather=lambda *a:False
        self.assertFalse(run(group,inp,out));self.assertEqual(len(calls),1)
        adapter.should_all_gather=lambda *a:True
        def failed(*a,**kw): raise RuntimeError('transport failed')
        runtime.all_gather=failed
        with self.assertRaisesRegex(RuntimeError,'transport failed'):run(group,inp,out)
        adapter.disabled=True;self.assertFalse(run(group,inp,out))
        self.assertFalse(run(NS(),inp,out))
        adapter.disabled=False;ns['ENABLED']=False;self.assertFalse(run(group,inp,out))

    def test_payload_initialization_only_zeroes_protocol_state(self):
        source=p.runtime(self.source(p.RUNTIME,'b12x'))
        node=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.If) and 'B12X_ROCE_LAZY_PAYLOAD_INIT' in ast.unparse(n.test))
        for enabled in (False,True):
            driver=NS(_layout=NS(total_bytes=2048,flag_off=1024,send_off=1280,ctrl_off=1920))
            ns={'self':driver,'os':NS(getenv=lambda *a:'1' if enabled else '0'),
                'torch':NS(uint8='u8',empty=lambda n,**kw:Tensor(np.full(n,219,np.uint8)),zeros=lambda n,**kw:Tensor(np.zeros(n,np.uint8)))}
            exec(ast.unparse(node),ns)
            region=driver._region.a
            self.assertTrue((region[1024:1280]==0).all() and (region[1920:]==0).all())
            self.assertTrue((region[:1024]==(219 if enabled else 0)).all())
            self.assertTrue((region[1280:1920]==(219 if enabled else 0)).all())

    def test_image_and_build_contract(self):
        docker=(ROOT/'Dockerfile.r22-dflash2-v16').read_text(encoding='utf-8')
        recipe=(ROOT/'recipes/glm53-exl3-v16-4x.yaml').read_text(encoding='utf-8')
        for name,value in [('VLLM_GB10_DCP_GATHER_INTO','1'),('VLLM_GB10_CKV_INPLACE','1'),
                           ('VLLM_GB10_DCP_MERGE_ROWS','256'),('B12X_ROCE_LAZY_PAYLOAD_INIT','1'),('B12X_ROCE_SKIP_EMPTY_CQ','1')]:
            self.assertIn(name+'='+value,docker);self.assertIn(name+': "'+value+'"',recipe)
        self.assertIn('0.87',recipe)
        build=(ROOT/'scripts/build-r22-dflash2-image.sh').read_text(encoding='utf-8')
        self.assertEqual(build.count('${GLM53_R22_SMOKE_SCRIPT:-smoke_r22_v16.py}'),2)
        self.assertLess(build.index('GLM53_R22_V16_SMOKE'),build.index('GLM53_R22_V12_SMOKE'))

    def test_inherited_timing_helpers_are_imported(self):
        tree = ast.parse((ROOT/'overlay/smoke_r22_v15.py').read_text(encoding='utf-8'))
        checked = []
        for function in tree.body:
            if not isinstance(function, ast.FunctionDef):
                continue
            calls = [n for n in ast.walk(function) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Name) and n.func.id == 'ms']
            if calls:
                self.assertTrue(any(isinstance(n, ast.ImportFrom) and n.module == 'smoke_r22_v14'
                                    and any(a.name == 'ms' for a in n.names)
                                    for n in function.body), function.name)
                checked.append(function.name)
        self.assertIn('mixed_activation_gpu', checked)
        docker = (ROOT/'Dockerfile.r22-dflash2-v16').read_text(encoding='utf-8')
        self.assertIn('COPY overlay/smoke_r22_v15.py /opt/compose/smoke_r22_v15.py', docker)

    def test_indexer_oracle_checks_selection_not_atomic_append_order(self):
        spec = importlib.util.spec_from_file_location('smoke16', ROOT/'overlay/smoke_r22_v16.py')
        smoke = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(smoke)
        ids = np.array([[0,1,2,-1],[-1,-1,-1,-1]], dtype=np.int32)
        scores = np.array([[5,5,5,99],[99,99,99,99]], dtype=np.float32)
        reference = smoke.indexer_reference_ids(ids,scores,2,1)
        np.testing.assert_array_equal(reference, [[0,1,2,3],[-1,-1,-1,-1]])
        smoke.assert_indexer_multiset(reference[:,::-1], reference, 'permutation')
        for column, wrong_id in ((0,1),(3,4),(0,-1)):
            bad = reference.copy(); bad[0,column] = wrong_id
            with self.assertRaises(AssertionError):
                smoke.assert_indexer_multiset(bad, reference, 'wrong selection')
        with self.assertRaises(AssertionError):
            smoke.assert_indexer_multiset(reference[::-1], reference, 'wrong row')
        np.testing.assert_array_equal(smoke.indexer_reference_ids(
            [[0,16,17,-1]], [[1,5,5,99]],2,16), [[32,33,48,49]])


if __name__ == '__main__': unittest.main()
