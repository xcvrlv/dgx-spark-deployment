import ast
from contextlib import nullcontext
from enum import Enum
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('roce_dtype_patch',ROOT/'patches/roce_dtype.py')
p=importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
SOURCE=Path(os.environ.get('DS41_B12X_SOURCE',ROOT.parent/'tmp/jj-audit/local-inference-lab-b12x-3a8b879/b12x'))

class Dtype(Enum):
    float16=1
    bfloat16=2
    float32=3
    def __str__(self): return 'torch.'+self.name

@unittest.skipUnless(SOURCE.is_dir(),'set DS41_B12X_SOURCE to pinned b12x package')
class RoceDtypeTests(unittest.TestCase):
    def test_actual_compile_function_preserves_dtype_keys_and_passes_string_names(self):
        original=(SOURCE/p.RELATIVE).read_text()
        patched=original.replace(p.OLD.decode(),p.NEW.decode())
        def run(source, dtype_values):
            scope={'__package__':'probe_roce','torch':NS(**{d.name:d for d in Dtype},cuda=NS(device=lambda _:nullcontext())),
                   'RoceQuery':lambda **kw:NS(**kw)}
            functions=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in ('_dtypes','compile_roce')]
            exec('from __future__ import annotations\n'+'\n'.join(ast.unparse(n) for n in functions),scope)
            calls=[]
            def launcher(name,*args):
                if name not in ('float16','bfloat16','float32'):
                    raise ValueError('unsupported dtype name')
                calls.append((name,args))
                return name+'-launcher'
            package=ModuleType('probe_roce'); package.__path__=[]
            one=ModuleType('probe_roce._oneshot_cute'); one.get_launcher=launcher
            gather=ModuleType('probe_roce._allgather_cute'); gather.get_launcher=lambda *args:'gather-launcher'
            with patch.dict(sys.modules,{package.__name__:package,one.__name__:one,gather.__name__:gather}):
                result=scope['compile_roce']({'world_size':4,'rank':0,'call':{'dtypes':dtype_values},
                    'setup':{'threads':256,'slots':2,'flag_stride':128,'hca_count':2}},0)
            return result,calls
        with self.assertRaisesRegex(ValueError,'unsupported dtype'):
            run(original,['float16'])
        for values in (list(Dtype),[d.name for d in Dtype]):
            result,calls=run(patched,values)
            self.assertEqual(set(result),{*Dtype,'gather'})
            for d in Dtype:
                self.assertEqual(result[d],d.name+'-launcher')
            self.assertEqual([name for name,_ in calls],[d.name for d in Dtype])
            self.assertTrue(all(args==(4,0,256,2,128,2,0) for _,args in calls))

    def test_hash_guard_and_revert(self):
        with tempfile.TemporaryDirectory() as d:
            target=Path(d)/p.RELATIVE; target.parent.mkdir(parents=True)
            original=(SOURCE/p.RELATIVE).read_bytes(); target.write_bytes(original)
            p.patch(d); p.patch(d); p.patch(d,check=True)
            p.patch(d,revert=True)
            self.assertEqual(target.read_bytes(),original)
            target.write_bytes(original+b'\n')
            with self.assertRaisesRegex(RuntimeError,'Unexpected upstream'):
                p.patch(d)
