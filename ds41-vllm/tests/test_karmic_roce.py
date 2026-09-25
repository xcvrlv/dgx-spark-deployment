"""Cross-tree checks for the sole local patch in the Karmic image."""
import importlib.util
import os
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('roce_karmic', ROOT / 'patches/roce_karmic.py')
patcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patcher)
SOURCE = Path(os.environ.get('DS41_KARMIC_B12X_SOURCE',
    ROOT / '.build/upstream/b12x-a7d7d29b2ef8869086e0ceaa787321f17544e3c9/b12x'))
VLLM_SOURCE = Path(os.environ.get('DS41_KARMIC_VLLM_SOURCE',
    ROOT / '.build/upstream/vllm-1794dcf18454900263e0c66711af8ea4a1283ac1/vllm'))


class KarmicRoceTests(unittest.TestCase):
    def test_image_has_only_roce_patch(self):
        dockerfile = (ROOT / 'Dockerfile.karmic').read_text()
        self.assertIn('patches/roce_karmic.py', dockerfile)
        for excluded in ('roce_dtype.py', 'roce_programs.py', 'roce_collective.py',
                         'prefill_hashes.py', 'b12x_tuning.py', 'engram_disk.py',
                         'display_kv.py', 'display_kv_credit.py'):
            self.assertNotIn('patches/' + excluded, dockerfile)

    @unittest.skipUnless(SOURCE.is_dir() and VLLM_SOURCE.is_dir(),
                         'pinned upstream snapshots unavailable')
    def test_prior_roce_fixes_are_upstream(self):
        adapter = (VLLM_SOURCE / 'distributed/device_communicators/b12x_roce_all_reduce.py').read_text()
        preparation = (SOURCE / 'comm/roce/_preparation.py').read_text()
        self.assertIn('collective=CollectiveRequirement(', adapter)
        self.assertIn('from_exchange_group(', adapter)
        self.assertIn('_DTYPE_NAMES[dtype]', preparation)
        for name in ('_oneshot_cute.py', '_allgather_cute.py'):
            self.assertIn('return attach_programs(run, raw)',
                          (SOURCE / 'comm/roce' / name).read_text())

    @unittest.skipUnless(SOURCE.is_dir(), 'pinned b12x snapshot unavailable')
    def test_pinned_source_apply_and_drift_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in patcher.INPUTS:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(SOURCE / name, target)
            patcher.patch(root)
            patcher.patch(root, check=True)
            patcher.patch(root)
            target = root / 'comm/roce/_roce_proxy.c'
            target.write_text(target.read_text() + '\n// unexpected drift\n')
            with self.assertRaisesRegex(RuntimeError, 'Unexpected upstream'):
                patcher.patch(root)


if __name__ == '__main__':
    unittest.main()
