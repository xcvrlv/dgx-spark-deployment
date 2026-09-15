import ast
from collections.abc import Mapping
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
from typing import Any
import unittest
from unittest.mock import patch as mock_patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('roce_programs_patch', ROOT/'patches/roce_programs.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
SOURCE = Path(os.environ.get('DS41_B12X_SOURCE', ROOT.parent/'tmp/jj-audit/local-inference-lab-b12x-92cd380/b12x'))

@unittest.skipUnless(SOURCE.is_dir(), 'set DS41_B12X_SOURCE to pinned b12x package')
class RoceProgramTests(unittest.TestCase):
    def test_real_launcher_factories_retain_compiler_identity(self):
        # Execute actual launcher factories with a fake GPU compiler, and real
        # upstream metadata functions. No native compilation is claimed here.
        tree = ast.parse((SOURCE/'_lib/compile_plan.py').read_text())
        funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('program_keys', 'attach_programs')]
        ns = dict(Any=Any, ProgramKey=tuple, Mapping=Mapping)
        exec(compile(ast.Module(body=funcs, type_ignores=[]), '<metadata>', 'exec'), ns)
        module = ModuleType('b12x._lib.compile_plan')
        module.attach_programs = ns['attach_programs']
        for filename in p.SOURCES:
            text = (SOURCE/'comm/roce'/filename).read_text()
            for repaired in (False, True):
                tree = ast.parse(text.replace(p.OLD.decode(), p.NEW.decode()) if repaired else text)
                fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'get_launcher')
                fn.decorator_list = []
                # Factory annotations aren't involved in launch semantics.
                fn.returns = None
                for arg in fn.args.args: arg.annotation = None
                raw = NS(__b12x_programs__=(('cute', filename),))
                env = dict(_process_key=lambda *a: a, _PREPARED_LAUNCHERS=set(),
                    _RoceOneshotLaunch=lambda *a: None, _RoceAllGatherLaunch=lambda *a: None,
                    raise_if_kernel_resolution_frozen=lambda *a, **k: None,
                    b12x_compile=lambda *a, **k: raw, _dummy=lambda *a: None,
                    cutlass=NS(Uint32=0), current_cuda_stream=lambda: 0,
                    KernelCompileSpec=NS(from_key=lambda *a: None))
                exec(compile(ast.Module(body=[fn], type_ignores=[]), filename, 'exec'), env)
                args = (4, 0, 256, 8, 16, 2, 0)
                if filename == '_oneshot_cute.py': args = ('float16',) + args
                with mock_patch.dict(sys.modules, {'b12x._lib.compile_plan': module}):
                    launcher = env['get_launcher'](*args)
                if repaired:
                    self.assertEqual(ns['program_keys'](launcher), raw.__b12x_programs__)
                    self.assertIs(launcher.__b12x_dependencies__[0], raw)
                else:
                    self.assertFalse(hasattr(launcher, '__b12x_programs__'))

    def test_describe_compilation_walks_full_compile_roce_carriers(self):
        # describe_compilation walks the dict compile_roce returns: dtype keys
        # plus the gather entry. Real patched launcher factories, a deferred
        # compiler stand-in, and real upstream metadata functions. No native
        # compilation is claimed here.
        tree = ast.parse((SOURCE/'_lib/compile_plan.py').read_text())
        funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('program_keys', 'attach_programs')]
        ns = dict(Any=Any, ProgramKey=tuple, Mapping=Mapping)
        exec(compile(ast.Module(body=funcs, type_ignores=[]), '<metadata>', 'exec'), ns)
        module = ModuleType('b12x._lib.compile_plan')
        module.attach_programs = ns['attach_programs']
        import torch
        observed = set()

        def compiler(*a, **k):
            program = ('cute', f'key-{len(observed)}')
            observed.add(program)
            return NS(__b12x_programs__=(program,))

        launchers = {}
        for filename in p.SOURCES:
            text = (SOURCE/'comm/roce'/filename).read_text()
            tree = ast.parse(text.replace(p.OLD.decode(), p.NEW.decode()))
            fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'get_launcher')
            fn.decorator_list = []
            fn.returns = None
            for arg in fn.args.args: arg.annotation = None
            env = dict(_process_key=lambda *a: a, _PREPARED_LAUNCHERS=set(),
                _RoceOneshotLaunch=lambda *a: None, _RoceAllGatherLaunch=lambda *a: None,
                raise_if_kernel_resolution_frozen=lambda *a, **k: None,
                b12x_compile=compiler, _dummy=lambda *a: None,
                cutlass=NS(Uint32=0), current_cuda_stream=lambda: 0,
                KernelCompileSpec=NS(from_key=lambda *a: None))
            exec(compile(ast.Module(body=[fn], type_ignores=[]), filename, 'exec'), env)
            launchers[filename] = env['get_launcher']
        common = (4, 0, 2, 8, 16, 2, 0)
        with mock_patch.dict(sys.modules, {'b12x._lib.compile_plan': module}):
            programs = {dtype: launchers['_oneshot_cute.py'](str(dtype).removeprefix("torch."), *common)
                        for dtype in (torch.float16, torch.bfloat16, torch.float32)}
            programs['gather'] = launchers['_allgather_cute.py'](*common)
            returned = ns['program_keys'](programs)
        self.assertTrue(returned, 'compile factory returned no carriers')
        self.assertLessEqual(observed, set(returned), 'compile factory discarded required programs')
        self.assertEqual(len(returned), 4)

    def test_guards_reapplication_and_independent_rollback(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            dest = root/'comm/roce'
            dest.mkdir(parents=True)
            for name in p.SOURCES:
                (dest/name).write_bytes((SOURCE/'comm/roce'/name).read_bytes())
            p.patch(root)
            p.patch(root)
            p.patch(root, check=True)
            p.patch(root, revert=True)
            for name in p.SOURCES:
                self.assertEqual((dest/name).read_bytes(), (SOURCE/'comm/roce'/name).read_bytes())
            target = dest/'_allgather_cute.py'
            target.write_bytes(target.read_bytes()+b'# source drift\n')
            with self.assertRaises(RuntimeError): p.patch(root)
            self.assertEqual((dest/'_oneshot_cute.py').read_bytes(), (SOURCE/'comm/roce/_oneshot_cute.py').read_bytes())
