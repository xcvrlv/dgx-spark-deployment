import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'.build/sparkring-audit-20260928/b12x-d44247b6171f7c2f9787341ae884b537887d7df9/b12x'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BarrierResetTests(unittest.TestCase):
    @unittest.skipUnless(SOURCE.is_dir(), 'latest b12x snapshot required')
    def test_exact_source_guard_roundtrip_and_drift(self):
        overlay = load('moe_barrier_patch', ROOT/'patches/knapcio_decode.py')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/overlay.RELATIVE
            path.parent.mkdir(parents=True)
            original = (SOURCE/overlay.RELATIVE).read_bytes()
            path.write_bytes(original)
            overlay.patch(root)
            overlay.patch(root)
            overlay.patch(root, check=True)
            self.assertEqual(path.read_bytes().count(overlay.NEW), 2)
            overlay.patch(root, revert=True)
            overlay.patch(root, revert=True, check=True)
            self.assertEqual(path.read_bytes(), original)
            path.write_bytes(original+b'\n')
            with self.assertRaises(RuntimeError):
                overlay.patch(root)
            self.assertEqual(path.read_bytes(), original+b'\n')
            self.assertFalse((root/overlay.HELPER).exists())

    def test_real_tensor_aliases_one_fill_and_safe_fallbacks(self):
        import torch
        helper = load('moe_barrier_runtime', ROOT/'patches/knapcio_decode_runtime.py')
        for count_size in (1, 4, 31, 64):
            first = 4
            second = ((first+count_size+3)//4)*4
            arena = torch.full((second+count_size+4,), 71, dtype=torch.int32)
            count, epoch = arena[first:first+count_size], arena[second:second+count_size]
            for flag in ('0', '1'):
                arena.fill_(71)
                with patch.dict(os.environ, DS41_MOE_COALESCE_BARRIERS=flag), \
                        torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
                    helper.clear_barriers(count, epoch)
                fills = sum(e.count for e in trace.key_averages() if e.key == 'aten::zero_')
                self.assertEqual(fills, 1 if flag == '1' else 2)
                self.assertEqual(count.tolist(), [0]*count_size)
                self.assertEqual(epoch.tolist(), [0]*count_size)
                self.assertEqual(arena[:first].tolist(), [71]*first)
                self.assertEqual(arena[second+count_size:].tolist(), [71]*4)
                self.assertEqual(arena[first+count_size:second].tolist(),
                                 [0 if flag == '1' else 71]*(second-first-count_size))
        # Separate allocations, non-contiguous views, different dtypes, a live
        # tensor between the barriers, overlap, and empties all keep two fills.
        arena = torch.full((32,), 71, dtype=torch.int32)
        examples = ((arena[:4], torch.full((4,), 71, dtype=torch.int32)),
                    (arena[:8:2], arena[8:12]),
                    (arena[:4], arena[4:8].float()),
                    (arena[:4], arena[8:12]),
                    (arena[:4], arena[2:6]),
                    (arena[:0], arena[4:8]))
        for count, epoch in examples:
            arena.fill_(71)
            with patch.dict(os.environ, DS41_MOE_COALESCE_BARRIERS='1'), \
                    torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
                helper.clear_barriers(count, epoch)
            self.assertEqual(sum(e.count for e in trace.key_averages() if e.key == 'aten::zero_'), 2)
            self.assertTrue((count == 0).all())
            self.assertTrue((epoch == 0).all())
            if count.numel() == 4 and count.is_contiguous() and epoch.storage_offset() == 8:
                self.assertEqual(arena[4:8].tolist(), [71]*4)


if __name__ == '__main__':
    unittest.main()
