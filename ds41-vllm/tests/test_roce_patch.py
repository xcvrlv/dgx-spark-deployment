import ast
import importlib.util
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('roce_patch', ROOT / 'patches/roce.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
SOURCE = Path(os.environ.get('DS41_B12X_SOURCE', ROOT.parent / 'tmp/jj-audit/local-inference-lab-b12x-92cd380/b12x'))


class PatchTests(unittest.TestCase):
    def test_selective_initialization_preserves_payload_and_clears_protocol(self):
        original = '            self._region = torch.zeros(\n                self._layout.total_bytes, dtype=torch.uint8, pin_memory=True\n            )\n'
        transformed = p.runtime(original)
        node = ast.parse('def init(self):\n' + transformed).body[0].body[0]
        class Memory:
            def __init__(self, n, fill=219, data=None):
                self.data = memoryview(bytearray([fill]) * n) if data is None else data
            def __getitem__(self, key):
                return Memory(0, data=self.data[key])
            def zero_(self):
                self.data[:] = bytes(len(self.data))
        for enabled in (False, True):
            owner = NS(_layout=NS(total_bytes=2048, flag_off=1024, send_off=1280, ctrl_off=1920))
            scope = {'self': owner, 'os': NS(getenv=lambda *a: '1' if enabled else '0'),
                     'torch': NS(uint8='u8', empty=lambda n, **kw: Memory(n), zeros=lambda n, **kw: Memory(n, 0))}
            exec(ast.unparse(node), scope)
            data = owner._region.data
            self.assertEqual(bytes(data[1024:1280]), bytes(256))
            self.assertEqual(bytes(data[1920:]), bytes(128))
            fill = 219 if enabled else 0
            self.assertEqual(bytes(data[:1024]), bytes([fill]) * 1024)
            self.assertEqual(bytes(data[1280:1920]), bytes([fill]) * 640)

    @unittest.skipUnless(SOURCE.is_dir(), 'set DS41_B12X_SOURCE to the pinned b12x package source')
    def test_real_upstream_apply_reapply_and_drift_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in p.INPUTS:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(SOURCE / name, target)
            p.patch(root)
            p.patch(root, check=True)
            p.patch(root)
            target = root / next(iter(p.INPUTS))
            target.write_text(target.read_text() + '\n// unexpected drift\n')
            with self.assertRaisesRegex(RuntimeError, 'Unexpected upstream'):
                p.patch(root)
