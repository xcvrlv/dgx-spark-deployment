"""Diagnostics must distinguish a configured optimization from observed execution."""
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('diagnose_prefill', ROOT / 'diagnose-prefill.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class PrefillDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.config = m.fleet.load_config(ROOT / 'cluster-karmic-c16.json')
        self.config.update(dspark_skip_prefill_draft=True, dspark_compact_context_graph=True,
                           engram_resident_scales=True, graph_memory_debug=True)
        argv = m.fleet.serve_args(self.config, 0)
        self.data = {
            'container': {
                'image_id': 'sha256:running', 'state': {'Running': True, 'OOMKilled': False},
                'labels': {'org.opencontainers.image.revision': self.config['vllm_commit'],
                           'local-inference.b12x.commit': self.config['b12x_commit'],
                           'local-inference.prefill-hash-overlay': 'ds41-bounded-hashes-v1',
                           'local-inference.dspark-prefill-overlay': 'ds41-dspark-prefill-v1'},
                'environment': m.fleet.environment(self.config, 0),
                'arguments': {key: argv[argv.index(key) + 1] for key in m.ARG_KEYS if key in argv},
            },
            'tag_image_id': 'sha256:running',
            'source': {'patch_files': {name: {'matches_checkout': True} for name in m.PATCH_FILES},
                       'patch_checks': {'prefill_hashes.py': 'passed', 'dspark_prefill.py': 'passed'}},
            'logs': {'activation': {key: False for key in m.MARKERS}},
            'memory': {'available_gib': 3, 'swap_used_gib': 1},
        }

    def test_absent_markers_and_existing_swap_are_not_reported_as_disabled_or_active_swapping(self):
        self.assertEqual(m.assess(self.config, 0, self.data), [])
        self.data['memory']['available_gib'] = 1.9
        self.assertIn('below 2 GiB', ' '.join(m.assess(self.config, 0, self.data)))

    def test_stale_tag_pin_drift_and_hash_patch_failure_are_detected(self):
        self.data['tag_image_id'] = 'sha256:new-build'
        self.data['container']['labels']['local-inference.b12x.commit'] = 'wrong-pin'
        self.data['source']['patch_checks']['prefill_hashes.py'] = 'Unexpected patch state'
        issues = ' '.join(m.assess(self.config, 0, self.data))
        for evidence in ('Running image ID differs', 'local-inference.b12x.commit', 'prefill_hashes.py'):
            self.assertIn(evidence, issues)

    def test_logs_do_not_override_disabled_flags_or_missing_resident_scales(self):
        self.data['logs']['activation'] = dict.fromkeys(m.MARKERS, True)
        self.data['container']['environment']['DS41_COMPACT_CONTEXT_GRAPH'] = '0'
        self.data['container']['environment']['VLLM_USE_BREAKABLE_CUDAGRAPH'] = '0'
        self.data['container']['arguments']['--engram-config'] = '{}'
        issues = ' '.join(m.assess(self.config, 0, self.data))
        self.assertIn('DS41_COMPACT_CONTEXT_GRAPH', issues)
        self.assertIn('--engram-config', issues)
        self.assertIn('Breakable CUDA graphs are explicitly disabled', issues)

    def test_failed_remote_check_cannot_pass(self):
        with patch.object(m.fleet, 'remote', side_effect=RuntimeError('SSH unavailable')):
            result = m.collect(self.config, 2)
        self.assertEqual(result['rank'], 2)
        self.assertEqual(result['issues'], ['SSH unavailable'])

    def test_host_probe_reads_only_and_whitelists_environment(self):
        argv = m.fleet.serve_args(self.config, 0)
        inspect = {'Image': 'sha256:running', 'State': {'Running': True},
                   'Config': {'Image': self.config['image'], 'Cmd': argv,
                              'Labels': self.data['container']['labels'],
                              'Env': ['SECRET_TOKEN=must-not-appear', 'DS41_SKIP_PREFILL_DRAFT=1']}}
        log = '\n'.join(m.MARKERS.values()) + '\nGraph capturing finished\n'
        calls = []
        def run(args, **kwargs):
            calls.append(args)
            if args[:3] == ['docker', 'container', 'inspect']:
                output = json.dumps([inspect])
            elif args[:3] == ['docker', 'image', 'inspect']:
                output = '[{"Id": "sha256:running"}]'
            elif args[:2] == ['docker', 'logs']:
                return subprocess.CompletedProcess(args, 0, '', log)
            elif args[:3] == ['docker', 'exec', '-i']:
                self.assertEqual(args[-2:], ['python3', '-'])
                self.assertEqual(kwargs['input'], 'probe')
                output = json.dumps(self.data['source'])
            else:
                self.fail(f'Unexpected Docker action: {args}')
            return subprocess.CompletedProcess(args, 0, output, '')
        namespace = dict(ENV_KEYS=m.ENV_KEYS, ARG_KEYS=m.ARG_KEYS, MARKERS=m.MARKERS,
                         CONTAINER_PROBE='probe')
        meminfo = 'MemAvailable: 3145728 kB\nSwapTotal: 1048576 kB\nSwapFree: 524288 kB\n'
        with patch.object(sys, 'argv', ['probe', 'ds41-jj-0', self.config['image']]), \
                patch.object(subprocess, 'run', side_effect=run), \
                patch.object(Path, 'read_text', return_value=meminfo), \
                patch('sys.stdout', new_callable=io.StringIO) as out:
            exec(m.HOST_PROBE, namespace)
        result = json.loads(out.getvalue())
        self.assertEqual(len(calls), 4)
        self.assertNotIn('must-not-appear', out.getvalue())
        self.assertTrue(all(result['logs']['activation'].values()))
        self.assertEqual(result['memory']['available_gib'], 3)
        self.assertEqual(result['memory']['swap_used_gib'], .5)
        self.assertEqual(result['container']['arguments']['--engram-config'],
                         self.data['container']['arguments']['--engram-config'])

    def test_probe_quotes_image_and_embedded_scripts_compile(self):
        config = dict(self.config, image="tag:quoted'$(touch should-not-run)")
        source = m.probe_script(config, 0)
        import shlex
        first = source.splitlines()[0].split(" <<'", 1)[0]
        self.assertEqual(shlex.split(first), ['python3', '-', 'ds41-jj-0', config['image']])
        host = source.split('\n', 1)[1].rsplit('\nDS41_DIAGNOSTIC_PY', 1)[0]
        compile(host, '<host>', 'exec')
        compile(m.CONTAINER_PROBE.replace('EXPECTED_HASHES', '{}'), '<container>', 'exec')


if __name__ == '__main__':
    unittest.main()
