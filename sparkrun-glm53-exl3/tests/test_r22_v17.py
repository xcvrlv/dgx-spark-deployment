import ast
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BASE = Path(os.getenv('GLM53_V17_BASELINE', ROOT.parent/'tmp/exl3-v17/raw'))


def load(name):
    spec = importlib.util.spec_from_file_location(name,ROOT/'overlay'/f'{name}.py')
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


p = load('patch_r22_v17')
m = load('gb10_startup_memory')


class Tests(unittest.TestCase):
    def test_preflight_and_idempotence(self):
        if not BASE.is_dir(): self.skipTest('set GLM53_V17_BASELINE')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in p.INPUTS:
                out=root/name;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(BASE/name,out)
            p.patch(root);p.patch(root);p.patch(root,check=True)
            shutil.copyfile(BASE/p.WORKER,root/p.WORKER)
            (root/p.POOL).write_text('# unknown',encoding='utf-8')
            before={f:f.read_bytes() for f in root.rglob('*') if f.is_file()}
            with self.assertRaises(RuntimeError):p.patch(root)
            self.assertEqual(before,{f:f.read_bytes() for f in root.rglob('*') if f.is_file()})

    def test_cleanup_orders_synchronization_and_never_runs_in_capture(self):
        events=[]
        cuda=NS(get_device_capability=lambda d:(12,1),is_current_stream_capturing=lambda:False,
                synchronize=lambda d:events.append('sync'),empty_cache=lambda:events.append('empty'))
        logger=NS(info=lambda *a:events.append('log'))
        with patch.dict(os.environ,{'VLLM_GB10_STARTUP_RECLAIM':'1'}), \
             patch.dict(sys.modules,{'torch':NS(cuda=cuda),'vllm.logger':NS(init_logger=lambda n:logger)}), \
             patch.object(m,'snapshot',side_effect=lambda *a:events.append('snapshot') or {}), \
             patch.object(m.gc,'collect',side_effect=lambda:events.append('gc')), \
             patch.object(m,'trim_heap',side_effect=lambda:events.append('trim') or 1):
            m.reclaim_startup_memory('test',0)
            self.assertEqual(events,['sync','snapshot','gc','empty','trim','snapshot','log'])
            events.clear();cuda.is_current_stream_capturing=lambda:True
            with self.assertRaises(RuntimeError):m.reclaim_startup_memory('capture',0)
            self.assertEqual(events,[])
            os.environ['VLLM_GB10_STARTUP_RECLAIM']='0'
            self.assertIsNone(m.reclaim_startup_memory('off',0));self.assertEqual(events,[])
            os.environ['VLLM_GB10_STARTUP_RECLAIM']='1';cuda.get_device_capability=lambda d:(10,0)
            self.assertIsNone(m.reclaim_startup_memory('other',0));self.assertEqual(events,[])

    def test_proc_fields_and_optional_glibc(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'meminfo';path.write_text('MemAvailable: 123 kB\nVmRSS: bad\nOther: 456 kB\n',encoding='utf-8')
            self.assertEqual(m.proc_kib(path,('MemAvailable','VmRSS')),{'MemAvailable':123})
            self.assertEqual(m.proc_kib(path.parent/'missing',('MemAvailable',)),{})
        with patch.object(m.ctypes,'CDLL',return_value=NS()):
            self.assertIsNone(m.trim_heap())

    def test_prefix_update_bounds_lazy_hash_reads_and_preserves_entries(self):
        if not BASE.is_dir(): self.skipTest('set GLM53_V17_BASELINE')
        source=p.pool((BASE/p.POOL).read_text(encoding='utf-8'))
        method=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name=='cache_full_blocks')
        ns={'resolve_block_hashes':lambda hashes,*a:hashes,'make_block_hash_with_group_id':lambda h,g:(h,g)}
        exec('from __future__ import annotations\n'+ast.unparse(method),ns)
        class Hashes:
            def __getitem__(self,key):
                self.last=key
                if isinstance(key,slice):
                    assert key.stop == 5, 'copied future prompt hashes'
                    return list(range(key.start,key.stop))
                return key
        for masked in (False,True):
            inserts=[];hashes=Hashes()
            blocks=[NS(is_null=(i==3),block_hash=None) for i in range(7)]
            owner=NS(hash_block_size=64,enable_kv_cache_events=False,
                     _insert_block_hash=lambda h,b,**kw:inserts.append((h,kw['num_tokens'])))
            ns['cache_full_blocks'](owner,NS(block_hashes=hashes),blocks,2,5,256,7,
                                    block_mask=[True,True,not masked])
            self.assertEqual(inserts,[((2,7),768)] if masked else [((2,7),768),((4,7),1280)])

    def test_startup_hooks_are_outside_inference_and_recipe_keeps_safe_default(self):
        if not BASE.is_dir(): self.skipTest('set GLM53_V17_BASELINE')
        tree=ast.parse(p.worker((BASE/p.WORKER).read_text(encoding='utf-8')))
        owners=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.FunctionDef):
                for call in ast.walk(node):
                    if isinstance(call,ast.Call) and isinstance(call.func,ast.Name) and call.func.id=='reclaim_startup_memory':owners.append(node.name)
        self.assertEqual(owners,['load_model','initialize_from_config','compile_or_warm_up_model','compile_or_warm_up_model'])
        recipe=(ROOT/'recipes/glm53-exl3-v17-4x.yaml').read_text(encoding='utf-8')
        self.assertIn('gpu_memory_utilization: 0.87',recipe)
        self.assertIn('--enable-prefix-caching',recipe)


if __name__ == '__main__':unittest.main()
