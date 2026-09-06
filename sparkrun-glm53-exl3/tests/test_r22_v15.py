"""Pinned-source and CPU semantic tests; CUDA/RDMA are image build/run gates."""
import ast
from dataclasses import dataclass
import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace as NS
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = Path(os.getenv('GLM53_V15_BASELINE', ROOT.parent/'tmp/exl3-v15/base'))
spec = importlib.util.spec_from_file_location('v15patch', ROOT/'overlay/patch_r22_v15.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


class Tests(unittest.TestCase):
    def test_pinned_overlay_reapplication_and_preflight(self):
        if not BASE.is_dir():
            self.skipTest('set GLM53_V15_BASELINE to exact v14 source')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'source'
            shutil.copytree(BASE,root)
            b,v=root/'b12x',root/'vllm'
            p.patch(b,v)
            p.patch(b,v)
            p.patch(b,v,check=True)
            # A mismatch in the last existing source must prevent all writes.
            (b/p.KERNEL).write_bytes((BASE/'b12x'/p.KERNEL).read_bytes())
            (v/p.SKINNY).write_text('# drift\n',encoding='utf-8')
            before={f:f.read_bytes() for f in root.rglob('*') if f.is_file()}
            with self.assertRaises(RuntimeError): p.patch(b,v)
            self.assertEqual(before,{f:f.read_bytes() for f in root.rglob('*') if f.is_file()})

    def test_compact_rotation_writes_only_token_rows(self):
        if not BASE.is_dir(): self.skipTest('exact v14 source required')
        s=p.kernel((BASE/'b12x'/p.KERNEL).read_text(encoding='utf-8'))
        cls=next(n for n in ast.parse(s).body if isinstance(n,ast.ClassDef) and n.name=='W4A16FusedMoeKernel')
        node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_run_input_rotation_shared')
        class NoCast(ast.NodeTransformer):
            def visit_Call(self,node):
                node=self.generic_visit(node)
                return node.func.value if isinstance(node.func,ast.Attribute) and node.func.attr=='to' else node
        node=NoCast().visit(node);node.decorator_list=[]
        ns={'Int32':int,'cutlass':NS(Float16=float,Float32=float,const_expr=bool,range_constexpr=range)}
        exec('from __future__ import annotations\n'+ast.unparse(node),ns)
        m,h,k=5,256,8
        drv=NS(hidden_size=h,top_k=k,cta_threads=128,gb10_compact_input=True,
               _had128_quad=lambda a,b,c,d,lane:(a,b,c,d))
        rng=np.random.default_rng(15)
        x,sg,su=rng.normal(size=m*h),rng.normal(size=h),rng.normal(size=h)
        g,u=np.full(m*k*h,np.nan),np.full(m*k*h,np.nan)
        for cta in range(2):
            for tid in range(128):
                ns['_run_input_rotation_shared'](drv,x,g,u,sg,su,None,None,None,None,8,8,tid,cta,2,m)
        np.testing.assert_equal(g[:m*h].reshape(m,h),x.reshape(m,h)*sg)
        np.testing.assert_equal(u[:m*h].reshape(m,h),x.reshape(m,h)*su)
        self.assertTrue(np.isnan(g[m*h:]).all() and np.isnan(u[m*h:]).all())
        # Actual FC1 address expressions must map each logical top-k route
        # to the same source token, including all three route-reader branches.
        reads=[n.value for n in ast.walk(ast.parse(s)) if isinstance(n,ast.Assign)
               and any(isinstance(t,ast.Name) and t.id=='rd_row' for t in n.targets)
               and 'input_route_divisor' in ast.unparse(n.value)]
        self.assertEqual(len(reads),3)
        for expr in reads:
            code=compile(ast.Expression(expr),'<route-reader>','eval')
            for idx in range(m*k):
                self.assertEqual(eval(code,{'idx':idx,'Int32':int,'self':NS(input_route_divisor=k)}),idx//k)

    def test_newton_reciprocal_one_ulp_bound(self):
        # Model an SFU seed at either endpoint of its 2^-23 relative-error
        # envelope, then execute both FP32 FMA corrections with exact products.
        values=np.geomspace(1.,2.**126,65537).astype(np.float32)
        exact=(1./values.astype(np.float64)).astype(np.float32)
        for sign in (-1,1):
            seed=(exact.astype(np.float64)*(1+sign*2.**-23)).astype(np.float32)
            error=(1.-values.astype(np.float64)*seed.astype(np.float64)).astype(np.float32)
            result=(seed.astype(np.float64)+error.astype(np.float64)*seed.astype(np.float64)).astype(np.float32)
            ulps=np.abs(result.view(np.uint32).astype(np.int64)-exact.view(np.uint32).astype(np.int64))
            self.assertLessEqual(ulps.max(),1)

    def test_sm121_has_its_own_skinny_profile_and_fallback(self):
        if not BASE.is_dir(): self.skipTest('exact v14 source required')
        source=p.skinny((BASE/'vllm'/p.SKINNY).read_text(encoding='utf-8'))
        def config(m,b,o,**kw): return NS(num_rows=m,block_size=b,outputs_per_block=o,vector_width=8,**kw)
        old={(2624,6144):object(),(2048,2048):object(),(6144,12288):object()}
        capability=[(12,1)]
        env={'VLLM_GB10_SKINNY_GEMM':'1'}
        ns={'__name__':__name__,'dataclass':dataclass,'SkinnyGemmConfig':config,'GLM52_PROJECTIONS':old,
            'os':NS(getenv=lambda name,default:env.get(name,default)),
            'current_platform':NS(is_device_capability=lambda cc:tuple(cc)==capability[0])}
        for node in ast.parse(source).body:
            if isinstance(node,ast.ClassDef) and node.name=='GLM52ProjectionSpec' or isinstance(node,ast.FunctionDef) and node.name in ('_is_sm103','_gb10_enabled','_device_supported','_projection_spec'):
                exec('from __future__ import annotations\n'+ast.unparse(node),ns)
        for shape in old:
            spec=ns['_projection_spec'](shape)
            self.assertEqual(set(spec.build_plan()),{1,2})
            self.assertTrue(all(backend=='cute' for backend,_ in spec.build_plan().values()))
            for _,cfg in spec.cute_configs:
                self.assertEqual(shape[0]%cfg.outputs_per_block,0)
                self.assertEqual(shape[1]%(cfg.block_size*cfg.vector_width),0)
        self.assertIsNone(ns['_projection_spec']((123,456)))
        env['VLLM_GB10_SKINNY_GEMM']='0'
        self.assertFalse(ns['_device_supported']())
        capability[0]=(10,3)
        self.assertTrue(ns['_device_supported']())
        self.assertIs(ns['_projection_spec']((2624,6144)),old[(2624,6144)])

    def test_build_and_rollback_contract(self):
        recipe=(ROOT/'recipes/glm53-exl3-v15-4x.yaml').read_text(encoding='utf-8')
        self.assertIn('gpu_memory_utilization: 0.87',recipe)
        for flag in ('VLLM_GB10_COMPACT_INPUT','VLLM_GB10_SIGMOID','VLLM_GB10_SKINNY_GEMM',
                     'B12X_ROCE_INLINE_PAYLOAD','B12X_ROCE_BALANCED_FANOUT'):
            self.assertIn(flag+': "1"',recipe)
        script=(ROOT/'scripts/build-r22-dflash2-image.sh').read_text(encoding='utf-8')
        self.assertEqual(script.count('/opt/compose/smoke_r22_v15.py --gpu'),2)
        docker=(ROOT/'Dockerfile.r22-dflash2-v15').read_text(encoding='utf-8')
        self.assertIn('RUN python3 /opt/compose/smoke_r22_v15.py',docker)
        self.assertEqual(hashlib.sha256((ROOT/'overlay/gb10_activation.py').read_bytes()).hexdigest(),p.OUTPUTS[p.ACTIVATION])


if __name__=='__main__': unittest.main()
