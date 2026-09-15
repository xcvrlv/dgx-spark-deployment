import ast
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('engram_disk_patch', ROOT/'patches/engram_disk.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
SOURCE = Path(os.environ.get('DS41_B12X_SOURCE', ROOT.parent/'tmp/jj-audit/local-inference-lab-b12x-3a8b879/b12x'))


@unittest.skipUnless(SOURCE.is_dir(), 'set DS41_B12X_SOURCE to pinned b12x package')
class EngramDiskPatchTests(unittest.TestCase):
    def test_guards_reapplication_and_independent_rollback(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            dest = root/p.RELATIVE
            dest.parent.mkdir(parents=True)
            dest.write_bytes((SOURCE/p.RELATIVE).read_bytes())
            p.patch(root)
            p.patch(root)
            p.patch(root, check=True)
            patched = dest.read_bytes()
            self.assertIn(b'def shard_start(self) -> int:', patched)
            self.assertIn(b'return self._cache.shard_rows', patched)
            self.assertNotIn(b'def shard_start(self) -> int:', (SOURCE/p.RELATIVE).read_bytes())
            ast.parse(patched.decode())
            p.patch(root, revert=True)
            self.assertEqual(dest.read_bytes(), (SOURCE/p.RELATIVE).read_bytes())
            dest.write_bytes(dest.read_bytes()+b'# source drift\n')
            with self.assertRaises(RuntimeError):
                p.patch(root)

    def test_pinned_source_matches_patch_anchor(self):
        # The hash guard fails closed on upstream drift; this asserts the
        # pinned tree still carries the exact anchor the patch replaces.
        text = (SOURCE/p.RELATIVE).read_bytes()
        self.assertIn(p.OLD, text)
        self.assertNotIn(p.NEW, text)


if __name__ == '__main__':
    unittest.main()
