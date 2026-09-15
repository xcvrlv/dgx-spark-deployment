import ast
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('roce_collective_patch', ROOT/'patches/roce_collective.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
SOURCE = Path(os.environ.get('DS41_VLLM_SOURCE', ROOT.parent/'tmp/jj-audit/local-inference-lab-vllm-c9dc4e5/vllm'))


@unittest.skipUnless(SOURCE.is_dir(), 'set DS41_VLLM_SOURCE to pinned vllm package')
class RoceCollectivePatchTests(unittest.TestCase):
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
            self.assertIn(b'from b12x.preparation import CollectiveRequirement', patched)
            self.assertIn(b'collective=CollectiveRequirement(', patched)
            self.assertIn(b'autotune=False,', patched)
            self.assertNotIn(b'collective=CollectiveRequirement(', (SOURCE/p.RELATIVE).read_bytes())
            ast.parse(patched.decode())
            p.patch(root, revert=True)
            self.assertEqual(dest.read_bytes(), (SOURCE/p.RELATIVE).read_bytes())
            dest.write_bytes(dest.read_bytes()+b'# source drift\n')
            with self.assertRaises(RuntimeError):
                p.patch(root)

    def test_patched_declaration_world_coordinates_the_priming(self):
        # The prepare call primes a real four-rank exchange, so the request
        # must declare the collective requirement (key and sorted participant
        # ranks) and never land in a per-rank tuning batch.
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            dest = root/p.RELATIVE
            dest.parent.mkdir(parents=True)
            dest.write_bytes((SOURCE/p.RELATIVE).read_bytes())
            p.patch(root)
            patched = dest.read_bytes().decode()
            self.assertIn(
                b'            collective=CollectiveRequirement(\n'
                b'                key=self._request_name(),\n'
                b'                ranks=tuple(sorted(self.global_ranks)),\n'
                b'            ),\n'.decode(),
                patched,
            )
            request_block = patched[patched.index('request = self._plan.request('):patched.index('return (', patched.index('request = self._plan.request('))]
            self.assertIn('collective=CollectiveRequirement(', request_block)
            unit_block = patched[patched.index('B12xPreparationUnit('):]
            self.assertIn('autotune=False,', unit_block)
            self.assertNotIn('autotune=not workload.eager_only', unit_block)

    def test_pinned_source_matches_patch_anchor(self):
        # The hash guard fails closed on upstream drift; this asserts the
        # pinned tree still carries the exact anchors the patch replaces.
        text = (SOURCE/p.RELATIVE).read_bytes()
        self.assertIn(p.IMPORTS_OLD, text)
        self.assertIn(p.REQUEST_OLD, text)
        self.assertIn(p.TUNE_OLD, text)
        self.assertNotIn(p.REQUEST_NEW, text)


if __name__ == '__main__':
    unittest.main()
