import ast
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('b12x_tuning_patch', ROOT/'patches/b12x_tuning.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
SOURCE = Path(os.environ.get('DS41_VLLM_SOURCE', ROOT.parent/'tmp/jj-audit/local-inference-lab-vllm-5bca5a5/vllm'))


@unittest.skipUnless(SOURCE.is_dir(), 'set DS41_VLLM_SOURCE to pinned vllm package')
class B12xTuningPatchTests(unittest.TestCase):
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
            self.assertIn(b'compile_workers=8,', patched)
            self.assertIn(b'rounds=1, samples=4,', patched)
            self.assertIn(b'race_batch=8, race_budget=4 * (1 << 30),', patched)
            self.assertNotIn(b'rounds=1, samples=4,', (SOURCE/p.RELATIVE).read_bytes())
            ast.parse(patched.decode())
            p.patch(root, revert=True)
            self.assertEqual(dest.read_bytes(), (SOURCE/p.RELATIVE).read_bytes())
            dest.write_bytes(dest.read_bytes()+b'# source drift\n')
            with self.assertRaises(RuntimeError):
                p.patch(root)

    def test_repairs_prior_patch_variant(self):
        # Images built with the previous patch form re-apply cleanly; unknown
        # drift still fails closed.
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            dest = root/p.RELATIVE
            dest.parent.mkdir(parents=True)
            pristine = (SOURCE/p.RELATIVE).read_bytes()
            dest.write_bytes(pristine.replace(p.OLD, p.PRIOR[0], 1))
            p.patch(root)
            self.assertEqual(dest.read_bytes(), pristine.replace(p.OLD, p.NEW, 1))

    def test_pinned_source_matches_patch_anchor(self):
        # The hash guard fails closed on upstream drift; this asserts the
        # pinned tree still carries the exact anchor the patch replaces.
        text = (SOURCE/p.RELATIVE).read_bytes()
        self.assertIn(p.OLD, text)
        self.assertNotIn(p.NEW, text)


if __name__ == '__main__':
    unittest.main()
