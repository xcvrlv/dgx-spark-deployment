import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fleet


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


guard = module('bounded_guard', 'patches/karmic_autotune_runtime.py')
patcher = module('bounded_patch', 'patches/karmic_autotune.py')
builder = module('bounded_builder', 'build-autotune.py')
runner = module('bounded_runner', 'run-autotune.py')
SOURCES = {
    'vllm': ROOT / '.build/upstream/vllm-1794dcf18454900263e0c66711af8ea4a1283ac1/vllm',
    'b12x': ROOT / '.build/upstream/b12x-a7d7d29b2ef8869086e0ceaa787321f17544e3c9/b12x',
}


class BoundedAutotuneTests(unittest.TestCase):
    def test_missing_image_counter_fails_without_waiting_on_a_jit_lock(self):
        with patch.object(guard.Path, 'is_file', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'Missing image-built'):
                guard.memory_counter()

    def test_dynamic_budget_accounts_for_host_and_cuda_and_leaves_reserve(self):
        gib = 1024 * guard.MIB
        self.assertEqual(guard.bounded_budget(gib, 40*gib, 40*gib, 4*gib), gib)
        self.assertEqual(guard.bounded_budget(gib, 40*gib, 5*gib, 4*gib), gib//2)
        self.assertEqual(guard.bounded_budget(gib, gib//2, 40*gib, 4*gib), gib//4)
        for free in (3*gib, 4*gib):
            with self.assertRaises(MemoryError):
                guard.bounded_budget(gib, 40*gib, free, 4*gib)

    def test_switch_off_preserves_session_defaults_and_avoids_proc_reads(self):
        with patch.dict(os.environ, {'DS41_B12X_BOUNDED_AUTOTUNE': '0'}), patch.object(guard, 'available_bytes') as read:
            self.assertEqual(guard.session_options(), {})
            guard.require_headroom()
            guard.write_progress(None)
            read.assert_not_called()

    def test_guard_refuses_invalid_reserve_and_low_host_memory(self):
        with patch.dict(os.environ, {'DS41_B12X_BOUNDED_AUTOTUNE': '1', 'DS41_B12X_RESERVE_MIB': '4096'}):
            self.assertEqual(guard.session_options()['race_batch'], 1)
            with patch.object(guard, 'available_bytes', return_value=3*1024*guard.MIB), self.assertRaises(MemoryError):
                guard.require_headroom()
            with patch.dict(os.environ, {'DS41_B12X_RESERVE_MIB': '2048'}), self.assertRaises(ValueError):
                guard.reserve_bytes()

    @unittest.skipUnless(all(p.is_dir() for p in SOURCES.values()), 'pinned snapshots required')
    def test_patch_roundtrip_idempotence_and_validate_all_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            roots = {name: Path(directory) / name for name in SOURCES}
            originals = {}
            for tree, relative in patcher.PATCHES:
                destination = roots[tree] / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                originals[destination] = (SOURCES[tree] / relative).read_bytes()
                destination.write_bytes(originals[destination])
            patcher.patch(**roots)
            patcher.patch(**roots, check=True)
            patcher.patch(**roots)
            patcher.patch(**roots, revert=True)
            patcher.patch(**roots, revert=True, check=True)
            self.assertTrue(all(path.read_bytes() == data for path, data in originals.items()))
            broken = roots['b12x'] / 'preparation/session.py'
            broken.write_bytes(broken.read_bytes() + b'\n# source drift\n')
            with self.assertRaises(RuntimeError):
                patcher.patch(**roots)
            first = roots['vllm'] / 'model_executor/warmup/b12x_prepare.py'
            self.assertEqual(first.read_bytes(), originals[first])
            self.assertFalse((roots['b12x'] / patcher.HELPER).exists())

    def test_candidate_preserves_operator_fields_and_explicitly_enables_tuning(self):
        source = ROOT / 'cluster-karmic-20260925.json'
        before = fleet.load_config(source)
        config = builder.candidate(source, 'candidate:test', batch_tokens=4096)
        self.assertEqual(config['max_model_len'], before['max_model_len'])
        self.assertEqual(config['gpu_memory_utilization'], before['gpu_memory_utilization'])
        self.assertEqual(config['model_path'], before['model_path'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps(config))
            loaded = fleet.load_config(path)
            environment = fleet.environment(loaded, 0)
            self.assertEqual(environment['B12X_AUTOTUNE'], '1')
            for stage in ('WEIGHTS', 'STATE', 'BIND'):
                self.assertEqual(environment[f'B12X_{stage}_COMPILE_WORKERS'], '1')
            arguments = fleet.serve_args(loaded, 0)
            self.assertEqual(json.loads(arguments[arguments.index('--kernel-config') + 1]), {'enable_b12x_autotune': True})
            config['b12x_compile_workers'] = 4
            path.write_text(json.dumps(config))
            with self.assertRaises(AssertionError):
                fleet.load_config(path)

    def test_failed_build_never_publishes_recipe(self):
        import subprocess
        source = ROOT / 'cluster-karmic-20260925.json'
        config = fleet.load_config(source)
        labels = {'org.opencontainers.image.revision': config['vllm_commit'],
                  'local-inference.b12x.commit': config['b12x_commit']}
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(builder.subprocess, 'check_output', return_value=json.dumps([{'Config': {'Labels': labels}}])), \
                patch.object(builder.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, ['docker'])):
            output = Path(directory) / 'candidate.json'
            with self.assertRaises(subprocess.CalledProcessError):
                builder.build(source, output)
            self.assertFalse(output.exists())

    def test_clock_heartbeat_does_not_count_as_meaningful_progress(self):
        a = [{'running': True, 'progress': {'time': 1, 'phase': 'selecting', 'request': 'norm.vision'}}]
        b = [{'running': True, 'progress': {'time': 99, 'phase': 'selecting', 'request': 'norm.vision'}}]
        self.assertEqual(runner.fingerprint(a), runner.fingerprint(b))
        b[0]['progress']['compiled'] = 1
        self.assertNotEqual(runner.fingerprint(a), runner.fingerprint(b))

    def test_watchdog_catches_transient_dip_and_stale_samples_without_paging_false_positive(self):
        sample = {'available_bytes': 5*runner.watch.GIB, 'swap_in_pages': 9, 'swap_out_pages': 8}
        sampler = NS(error=None, samples=[sample, dict(sample)], last_received=10, maximum_gap=5)
        self.assertIsNone(runner.memory_failure([sampler], 11))
        sampler.samples[-1]['swap_in_pages'] += 6
        self.assertIsNone(runner.memory_failure([sampler], 11))
        self.assertIn('stale', runner.memory_failure([sampler], 16))
        sampler.samples.insert(1, dict(sample, available_bytes=2*runner.watch.GIB))
        self.assertIn('3 GiB', runner.memory_failure([sampler], 11))
        sampler.samples.pop(1)
        sampler.samples[-1]['swap_out_pages'] += 11033
        self.assertIsNone(runner.memory_failure([sampler], 11))

    def test_report_distinguishes_old_swap_reads_from_new_swap_writes(self):
        sample = {'available_bytes': 5*runner.watch.GIB, 'swap_in_pages': 9,
                  'swap_out_pages': 8, 'time': 1, 'swap_used_bytes': runner.watch.GIB}
        sampler = NS(error=None, samples=[sample, dict(sample, time=2, swap_in_pages=15)])
        report = runner.memory_summary(sampler)
        self.assertTrue(report['passed'])
        self.assertEqual(report['swap_in_pages'], 6)
        sampler.samples[-1]['swap_out_pages'] += 11033
        self.assertTrue(runner.memory_summary(sampler)['passed'])
        self.assertEqual(runner.memory_summary(sampler)['monitored_swap_out_pages'], 11033)
        sampler.samples[-1]['available_bytes'] = runner.watch.GIB
        self.assertFalse(runner.memory_summary(sampler)['passed'])

    def test_loader_and_preparation_swap_are_separately_recorded(self):
        sample = {'available_bytes': 40*runner.watch.GIB, 'swap_in_pages': 9,
                  'swap_out_pages': 8, 'time': 1, 'swap_used_bytes': runner.watch.GIB}
        loaded = dict(sample, time=2, swap_out_pages=100)
        sampler = NS(error=None, samples=[sample, loaded], last_received=10, maximum_gap=5)
        self.assertIsNone(runner.memory_failure([sampler], 11, swap_baselines=[None]))
        self.assertTrue(runner.memory_summary(sampler, loaded)['passed'])
        self.assertEqual(runner.memory_summary(sampler, loaded)['swap_out_pages'], 92)
        sampler.samples.append(dict(loaded, time=3, swap_out_pages=11133))
        self.assertIsNone(runner.memory_failure([sampler], 11, swap_baselines=[loaded]))
        self.assertTrue(runner.memory_summary(sampler, loaded)['passed'])
        self.assertEqual(runner.memory_summary(sampler, loaded)['monitored_swap_out_pages'], 11033)
        sampler.samples[-1]['available_bytes'] = 2*runner.watch.GIB
        self.assertIn('3 GiB', runner.memory_failure([sampler], 11, swap_baselines=[None]))

    def test_progress_completion_is_counted_once(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
                'DS41_B12X_BOUNDED_AUTOTUNE': '1', 'B12X_PREPARATION_TRACE_DIR': directory}):
            job = NS(_result=object(), _started=123, _benchmarked=7, _cache_hits=3,
                     _error=None, _compilations=2, _phase='finishing', _active_request=None,
                     _completed_requests=10, _total_requests=10, _candidates_prepared=2,
                     _batch_index=4, _completed_rounds=3, _timing=NS(counts={}), autotune=True,
                     session=NS(_pool=None, _stop=NS(is_set=lambda: False)))
            old = guard._totals['measured']
            guard.write_progress(job, force=True)
            guard.write_progress(job, force=True)
            result = json.loads(next(Path(directory).glob('*.json')).read_text())
            self.assertEqual(result['totals']['measured'], old + 7)
            self.assertEqual(len(list(Path(directory).glob('*.tmp'))), 0)


if __name__ == '__main__':
    unittest.main()
