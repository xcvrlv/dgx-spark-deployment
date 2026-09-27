import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fleet

spec = importlib.util.spec_from_file_location('memory_watch', ROOT / 'memory-watch.py')
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)


class PrefillControlsTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / 'cluster-karmic-c16.json').read_text())

    def load(self, **values):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps(dict(self.config, **values)))
            return fleet.load_config(path)

    def test_resident_scales_changes_only_the_engram_argument(self):
        base = fleet.serve_args(self.config, 0)
        candidate = fleet.serve_args(self.load(engram_resident_scales=True), 0)
        index = base.index('--engram-config') + 1
        self.assertEqual(json.loads(candidate[index]), {
            'cpu_offload': False, 'table_memory': 'disk', 'disk_resident_scales': True})
        candidate[index] = base[index]
        self.assertEqual(candidate, base)
        self.assertEqual(fleet.serve_args(self.load(engram_resident_scales=False), 0), base)

    def test_native_diagnostics_and_profiler_can_be_enabled_independently(self):
        c = self.load(graph_memory_debug=True)
        self.assertEqual(fleet.environment(c, 0)['VLLM_DEBUG_GRAPH_MEMORY_ACCOUNTING'], '1')
        self.assertNotIn('VLLM_DEBUG_GRAPH_MEMORY_ACCOUNTING', fleet.environment(self.config, 0))
        self.assertEqual(fleet.serve_args(c, 0), fleet.serve_args(self.config, 0))
        profiled = fleet.serve_args(self.load(torch_profile=True), 0)
        profile = json.loads(profiled[profiled.index('--profiler-config') + 1])
        self.assertEqual(profile['profiler'], 'torch')
        self.assertFalse(profile['torch_profiler_with_memory'])
        self.assertNotIn('--compilation-config', profiled)

    def test_wrong_types_and_legacy_profiles_rejected(self):
        for key in ('engram_resident_scales', 'graph_memory_debug', 'torch_profile'):
            for value in ('false', 1, None):
                with self.subTest(key=key, value=value), self.assertRaises(AssertionError):
                    self.load(**{key: value})
        for key in ('engram_resident_scales', 'graph_memory_debug'):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.load(upstream_branch='dev/jovian-judgement', **{key: True})


class MemoryQualificationTests(unittest.TestCase):
    def test_remote_probe_reads_physical_availability_and_swap_counters(self):
        mem = 'MemTotal: 126877696 kB\nMemFree: 1024 kB\nMemAvailable: 2097152 kB\nSwapTotal: 1048576 kB\nSwapFree: 524288 kB\n'
        vm = 'pswpin 13\npswpout 17\n'
        output = io.StringIO()
        with patch.object(Path, 'read_text', side_effect=[mem, vm]), \
             patch('select.select', return_value=([sys.stdin], [], [])), \
             patch.object(sys, 'argv', ['-c', '0.5']), patch.object(sys, 'stdout', output):
            exec(compile(watch.PROBE, '<remote-probe>', 'exec'), {})
        sample = json.loads(output.getvalue())
        self.assertEqual(sample['available_bytes'], 2 * watch.GIB)
        self.assertEqual(sample['swap_used_bytes'], watch.GIB // 2)
        self.assertEqual(sample['swap_in_pages'], 13)
        self.assertEqual(sample['swap_out_pages'], 17)

    def sample(self, seconds, available=3, swap_in=10, swap_out=20):
        return {'time': seconds, 'available_bytes': available * watch.GIB,
                'swap_used_bytes': watch.GIB, 'swap_in_pages': swap_in,
                'swap_out_pages': swap_out}

    def test_transient_dip_fails_even_when_endpoints_pass(self):
        samples = [self.sample(0), self.sample(1, available=1.9), self.sample(2)]
        result = watch.summarize(samples, 2 * watch.GIB)
        self.assertFalse(result['passed'])
        self.assertAlmostEqual(result['minimum_available_gib'], 1.9)

    def test_exact_boundary_passes_and_existing_inactive_swap_is_not_new_io(self):
        result = watch.summarize([self.sample(0, 2), self.sample(1, 2)], 2 * watch.GIB)
        self.assertTrue(result['passed'])

    def test_swap_and_missing_or_failed_probes_fail_closed(self):
        for end in (self.sample(1, swap_in=11), self.sample(1, swap_out=21),
                    self.sample(1, swap_in=0)):
            self.assertFalse(watch.summarize([self.sample(0), end], 2 * watch.GIB)['passed'])
        self.assertFalse(watch.summarize([], 2 * watch.GIB)['passed'])
        self.assertFalse(watch.summarize([self.sample(0)], 2 * watch.GIB, 'SSH failed')['passed'])

    def test_failed_initial_node_prevents_foreground_command_and_preserves_report(self):
        class FakeSampler:
            def __init__(inner, config, rank, output, interval):
                inner.samples = [self.sample(0, available=1 if rank == 2 else 3)]
                inner.error = None
                inner.ready = type('Ready', (), {'wait': lambda *args: None})()
                inner.stopped = False
            def stop(inner):
                inner.stopped = True
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'measurement'
            argv = ['memory-watch.py', '--config', str(ROOT / 'cluster-karmic-c16.json'),
                    '--output', str(output), '--', 'command-must-not-run']
            with patch.object(sys, 'argv', argv), patch.object(watch, 'Sampler', FakeSampler), \
                 patch.object(watch.subprocess, 'run') as run, patch('builtins.print'):
                self.assertEqual(watch.main(), 1)
            run.assert_not_called()
            report = json.loads((output / 'summary.json').read_text())
            self.assertFalse(report['passed'])
            self.assertIn('rank 2', report['error'])
            self.assertEqual(len(report['ranks']), 4)


if __name__ == '__main__':
    unittest.main()
