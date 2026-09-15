import copy
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import struct
import sys
import tempfile
import unittest
import runpy
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fleet

spec = importlib.util.spec_from_file_location('model_check', ROOT / 'model-check.py')
model_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_check)


class RoceCheckTests(unittest.TestCase):
    def test_hca_comparison_accepts_sequence_types_but_preserves_order(self):
        tree = ast.parse((ROOT / 'roce-check.py').read_text())
        # The comparison lives inside main(): spawned compiler workers
        # re-import the module, so its body is guard-protected.
        check = next(node for node in ast.walk(tree) if isinstance(node, ast.Assert)
                     and 'rt.hca_names' in ast.unparse(node.test))
        expression = compile(ast.Expression(check.test), '<hca-check>', 'eval')
        expected = ['rocep1s0f0', 'roceP2p1s0f0']
        for actual, passes in ((expected, True), (tuple(expected), True),
                               (expected[::-1], False), (expected[:1], False)):
            scope = {'rt': SimpleNamespace(hca_names=actual),
                     'os': SimpleNamespace(environ={'B12X_ROCE_HCA': ','.join(expected)})}
            self.assertEqual(eval(expression, scope), passes)

    def test_preparation_uses_the_world_coordinator(self):
        # The RoCE request declares a collective, so session.prepare() without
        # a coordinator raises "collective preparation requires a coordinator";
        # the check must qualify the serving path's coordinated rounds instead.
        tree = ast.parse((ROOT / 'roce-check.py').read_text())
        unparsed = ast.unparse(tree)
        self.assertNotIn('session.prepare(', unparsed)
        self.assertIn('B12xPreparationCoordinator', unparsed)
        self.assertIn("outcome['error'] is None", unparsed)

    def test_coordinator_batches_are_request_autotune_pairs(self):
        # The coordinator unpacks (requests, autotune) per batch; a bare
        # request tuple fails at construction with "not enough values to
        # unpack (expected 2, got 1)".
        tree = ast.parse((ROOT / 'roce-check.py').read_text())
        call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == 'B12xPreparationCoordinator')
        expression = compile(ast.Expression(call.args[1]), '<batches>', 'eval')
        units = [SimpleNamespace(requests=[object()])]
        batches = eval(expression, {'units': units})
        self.assertEqual(len(batches), 1)
        for requests, autotune in batches:
            self.assertEqual(len(requests), 1)
            self.assertIs(autotune, False)


class ImageCheckTests(unittest.TestCase):
    def test_build_check_does_not_import_driver_dependent_vllm(self):
        proxy = ModuleType('b12x.comm.roce._proxy')
        proxy.load = lambda: SimpleNamespace(_name='proxy.so')
        storage = ModuleType('b12x.loader._native')
        storage._build = lambda: 'storage.so'
        modules = {proxy.__name__: proxy, storage.__name__: storage}
        import builtins
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == 'vllm' or name.startswith('vllm.'):
                raise ImportError('libcuda.so.1 unavailable during docker build')
            return original_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as d:
            package = Path(d)
            (package / '_C_stable_libtorch.abi3.so').touch()
            wheel = SimpleNamespace(locate_file=lambda _: package)
            with patch.dict(sys.modules, modules), \
                 patch.object(sys, 'argv', ['image-check.py']), \
                 patch('platform.machine', return_value='aarch64'), \
                 patch('importlib.metadata.distribution', return_value=wheel), \
                 patch('importlib.metadata.version', return_value='4.6.2'), \
                 patch('subprocess.run'), \
                 patch('builtins.__import__', side_effect=guarded_import):
                runpy.run_path(str(ROOT / 'image-check.py'), run_name='__main__')


class FleetTests(unittest.TestCase):
    def setUp(self):
        self.c = fleet.load_config(ROOT / 'cluster-c8.json')

    def test_four_distinct_ranks_and_native_context(self):
        # Exercise the native cap independently of the operator's saved profile.
        self.c['max_model_len'] = 1048576
        for rank in range(4):
            args = fleet.serve_args(self.c, rank)
            self.assertEqual(args[args.index('--node-rank') + 1], str(rank))
            self.assertEqual('--headless' in args, rank != 0)
            self.assertEqual(args[args.index('--max-model-len') + 1], '1048576')
            self.assertEqual(args[args.index('--decode-context-parallel-size') + 1], '1')
            self.assertNotIn('--disable-custom-all-reduce', args)
            self.assertNotIn('--hf-overrides', args)
            self.assertNotIn('--speculative-config', args)

    def test_b12x_autotune_switch_keeps_backends(self):
        # The kernel-config JSON must not wipe the per-field backend flags;
        # create_engine_config deepcopies and applies those on top.
        self.assertIn('--kernel-config', fleet.serve_args(self.c, 0))
        self.assertNotIn('--kernel-config', fleet.serve_args(dict(self.c, b12x_autotune=True), 0))
        bad = dict(self.c, b12x_autotune='off')
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'bad.json'
            path.write_text(json.dumps(bad))
            with self.assertRaises(AssertionError):
                fleet.load_config(path)
        self.c['b12x_autotune'] = False
        args = fleet.serve_args(self.c, 0)
        config = json.loads(args[args.index('--kernel-config') + 1])
        self.assertFalse(config['enable_b12x_autotune'])
        self.assertEqual(args[args.index('--moe-backend') + 1], 'b12x')
        self.assertEqual(args[args.index('--linear-backend') + 1], 'b12x')

    def test_draft_capture_covers_every_c8_depth(self):
        self.c['draft_tokens'] = 3
        args = fleet.serve_args(self.c, 0)
        config = json.loads(args[args.index('--compilation-config') + 1])
        # Minimal base plus the cap: every captured size is a separate b12x
        # specialization, so the sparse spread bounds preparation volume.
        self.assertEqual(config['cudagraph_capture_sizes'], [1, 2, 8, 32])
        draft = json.loads(args[args.index('--speculative-config') + 1])
        self.assertEqual(draft['draft_tensor_parallel_size'], 4)

    def test_c16_limits_capture_and_communication_switches(self):
        c = fleet.load_config(ROOT / 'cluster-c16.json')
        args = fleet.serve_args(c, 0)
        self.assertEqual(args[args.index('--max-num-seqs') + 1], '16')
        self.assertEqual(args[args.index('--max-model-len') + 1], '393216')
        self.assertEqual(args[args.index('--gpu-memory-utilization') + 1], '0.88')
        compilation = json.loads(args[args.index('--compilation-config') + 1])
        self.assertEqual(compilation['cudagraph_capture_sizes'], [1, 2, 8, 128])
        draft = json.loads(args[args.index('--speculative-config') + 1])
        self.assertEqual(draft['num_speculative_tokens'], 7)
        self.assertEqual(draft['method'], 'dspark')
        c['draft_tokens'] = 3
        args = fleet.serve_args(c, 0)
        compilation = json.loads(args[args.index('--compilation-config') + 1])
        self.assertEqual(compilation['cudagraph_capture_sizes'], [1, 2, 8, 64])
        self.assertTrue(all(fleet.environment(c, 0)[key] == '1' for key in fleet.ROCE_OPTIONS.values()))
        c['roce_optimizations']['inline_payload'] = False
        self.assertEqual(fleet.environment(c, 0)['B12X_ROCE_INLINE_PAYLOAD'], '0')
        self.assertEqual(fleet.environment(c, 0)['B12X_ROCE_SKIP_EMPTY_CQ'], '1')

    def test_configure_explicitly_selects_dspark7_and_preserves_paths(self):
        with tempfile.TemporaryDirectory() as d:
            source, output = Path(d) / 'old.json', Path(d) / 'new.json'
            old = copy.deepcopy(self.c)
            old['draft_tokens'] = 0
            old['model_path'] = '/srv/operator-checkpoint'
            source.write_text(json.dumps(old))
            with patch.object(sys, 'argv', ['configure-c16.py', '--from-config', str(source), '--output', str(output)]):
                runpy.run_path(str(ROOT / 'configure-c16.py'), run_name='__main__')
            result = fleet.load_config(output)
            self.assertEqual(result['draft_tokens'], 7)
            self.assertEqual(result['model_path'], old['model_path'])
            self.assertEqual(result['gpu_memory_utilization'], 0.88)
            result['max_num_batched_tokens'] = 128
            output.write_text(json.dumps(result))
            with self.assertRaisesRegex(AssertionError, 'profiling rows'):
                fleet.load_config(output)

    def test_adaptive_profile_generation_launch_and_rollback(self):
        with tempfile.TemporaryDirectory() as d:
            source, adaptive, fixed = [Path(d) / n for n in ('source.json', 'adaptive.json', 'fixed.json')]
            source.write_text(json.dumps(self.c))
            with patch.object(sys, 'argv', ['configure-c16.py', '--from-config', str(source), '--output', str(adaptive), '--adaptive-window', '32', '--adaptive-initial', '7']):
                runpy.run_path(str(ROOT / 'configure-c16.py'), run_name='__main__')
            c = fleet.load_config(adaptive)
            args = fleet.serve_args(c, 0)
            spec = json.loads(args[args.index('--speculative-config') + 1])
            self.assertEqual(spec['adaptive_speculative_tokens_window'], 32)
            self.assertEqual(spec['adaptive_speculative_tokens_initial'], 7)
            self.assertEqual(spec['num_speculative_tokens'], 7)
            graph = json.loads(args[args.index('--compilation-config') + 1])
            self.assertEqual(graph['cudagraph_capture_sizes'], [1, 2, 8, 128])
            with patch.object(sys, 'argv', ['configure-c16.py', '--from-config', str(adaptive), '--output', str(fixed)]):
                runpy.run_path(str(ROOT / 'configure-c16.py'), run_name='__main__')
            self.assertNotIn('adaptive_speculative_tokens_window', fleet.load_config(fixed))
            for updates in ({'adaptive_speculative_tokens_window': 0},
                            {'adaptive_speculative_tokens_initial': 8},
                            {'draft_tokens': 0}):
                bad = dict(c, **updates)
                fixed.write_text(json.dumps(bad))
                with self.assertRaises(AssertionError):
                    fleet.load_config(fixed)

    def test_profile_is_opt_in_and_preserves_serving_limits(self):
        c = copy.deepcopy(self.c)
        self.assertNotIn('--profiler-config', fleet.serve_args(c, 0))
        c['torch_profile'] = True
        c['gpu_memory_utilization'] = 0.85
        args = fleet.serve_args(c, 0)
        profile = json.loads(args[args.index('--profiler-config') + 1])
        self.assertEqual(profile['profiler'], 'torch')
        self.assertEqual(profile['torch_profiler_dir'], '/cache/profiles')
        self.assertEqual(args[args.index('--gpu-memory-utilization') + 1], '0.85')

    def test_rdma_counter_units_and_reset_detection(self):
        import observe
        key = '/sys/class/infiniband/hca/ports/1/counters/port_xmit_data'
        before = {'0': {'time': 10, 'counters': {key: 100, 'errors': 3}}}
        after = {'0': {'time': 12, 'counters': {key: 200, 'errors': 1}}}
        counters = observe.delta(before, after)['0']['counters']
        self.assertEqual(counters[key]['bytes_per_second'], 200)
        self.assertTrue(counters['errors']['reset_or_wrap'])

    def test_omp_default_and_operator_override(self):
        c = copy.deepcopy(self.c)
        c.pop('omp_num_threads', None)
        self.assertEqual(fleet.environment(c, 0)['OMP_NUM_THREADS'], '2')
        c['omp_num_threads'] = 1
        self.assertEqual(fleet.environment(c, 0)['OMP_NUM_THREADS'], '1')

    def test_r38_caps_c8_and_selects_cache_geometry(self):
        c = fleet.load_config(ROOT / 'cluster-r38-c8.json')
        args = fleet.serve_args(c, 0)
        for flag, value in (('--max-num-seqs','8'), ('--max-model-len','393216'),
                            ('--gpu-memory-utilization','0.85'), ('--block-size','256'),
                            ('--swa-block-size','128')):
            self.assertEqual(args[args.index(flag)+1], value)
        graphs = json.loads(args[args.index('--compilation-config')+1])
        # Minimal base plus the cap bounds both preparation stages: the state
        # stage declares the same counts as the weights stage.
        self.assertEqual(graphs['cudagraph_capture_sizes'],[1, 2, 8, 48])
        self.assertEqual(c['draft_tokens'],5)
        with tempfile.TemporaryDirectory() as d:
            source, target = Path(d)/'old.json', Path(d)/'r38.json'
            old = dict(self.c, model_path='/srv/original', omp_num_threads=1,
                       adaptive_speculative_tokens_window=32, b12x_autotune=True)
            source.write_text(json.dumps(old))
            with patch.object(sys,'argv',['configure-r38.py','--from-config',str(source),'--output',str(target)]):
                runpy.run_path(str(ROOT/'configure-r38.py'),run_name='__main__')
            updated = fleet.load_config(target)
            self.assertEqual(updated['model_path'],'/srv/original')
            self.assertEqual(updated['omp_num_threads'],1)
            self.assertEqual(updated['max_num_seqs'],8)
            self.assertFalse(updated['b12x_autotune'])
            self.assertNotIn('graph_request_buckets',updated)
            self.assertFalse(updated['reduced_tuning'])
            self.assertNotIn('adaptive_speculative_tokens_window',updated)

    def test_flat_checkpoint_used_by_serving_and_preflight(self):
        self.c['model_path'] = '/srv/DeepSeek-V4.1-Flash-MXFP4-FP4-Engram'
        self.c['model_subpath'] = '.'
        self.assertEqual(fleet.serve_args(self.c, 0)[2], '/checkpoint')
        with patch.object(fleet, 'remote', return_value='sha256:same') as remote:
            fleet.preflight(self.c)
        for call in remote.call_args_list:
            script = call.args[2]
            self.assertIn(self.c['model_path'] + '/config.json', script)
            self.assertIn('/opt/ds41/model-check.py /checkpoint\n', script)
            self.assertNotIn('/snapshots/', script)

    def test_preflight_guards_b12x_tuning_label_and_rolls_back(self):
        with patch.object(fleet, 'remote', return_value='sha256:same') as remote:
            fleet.preflight(self.c)
        self.assertTrue(all('local-inference.b12x-tuning' in call.args[2] for call in remote.call_args_list))
        c = copy.deepcopy(self.c)
        c['reduced_tuning'] = False
        with patch.object(fleet, 'remote', return_value='sha256:same') as remote:
            fleet.preflight(c)
        self.assertFalse(any('local-inference.b12x-tuning' in call.args[2] for call in remote.call_args_list))

    def test_preflight_guards_roce_collective_label(self):
        # The RoCE prepare call primes a real four-rank exchange, so every
        # serving image must carry the coordinating patch label.
        with patch.object(fleet, 'remote', return_value='sha256:same') as remote:
            fleet.preflight(self.c)
        self.assertTrue(all('local-inference.roce-collective' in call.args[2] for call in remote.call_args_list))

    def test_cluster_configs_match_versions_env_image(self):
        # A stale image field serves an image without the latest fixes; the
        # distributed tag must resolve to exactly the pinned IMAGE string.
        image = dict(
            line.split('=', 1) for line in (ROOT/'versions.env').read_text().splitlines()
            if line.startswith('IMAGE=')
        )['IMAGE']
        for name in ('cluster-c8.json', 'cluster-c16.json', 'cluster-r38-c8.json'):
            self.assertEqual(fleet.load_config(ROOT / name)['image'], image, name)

    def test_checkpoint_subpath_cannot_escape_mount(self):
        for subpath in ('../outside', '/outside'):
            self.c['model_subpath'] = subpath
            with self.assertRaises(AssertionError):
                fleet.checkpoint_path(self.c, '/checkpoint')

    def test_shell_roundtrip_keeps_paths_and_json_literal(self):
        self.c['model_path'] = '/srv/model with spaces/$(touch SHOULD_NOT_EXIST)'
        cmd = fleet.docker(self.c, 1, 'probe') + fleet.serve_args(self.c, 1)
        self.assertEqual(shlex.split(shlex.join(cmd)), cmd)
        mount = next(x for x in cmd if x.startswith('type=bind,src=/srv/model'))
        self.assertIn('$(touch SHOULD_NOT_EXIST)', mount)

    def test_fabric_uses_per_rank_address_and_exact_hcas(self):
        a, b = fleet.environment(self.c, 0), fleet.environment(self.c, 1)
        self.assertNotEqual(a['VLLM_HOST_IP'], b['VLLM_HOST_IP'])
        self.assertEqual(a['NCCL_IB_HCA'], '=rocep1s0f0,roceP2p1s0f0')
        self.assertEqual(a['B12X_ROCE_HCA'], 'rocep1s0f0,roceP2p1s0f0')
        self.assertEqual(a['VLLM_ENABLE_ROCE_ALLREDUCE'], '1')
        self.assertEqual(a['VLLM_ENABLE_PCIE_ALLREDUCE'], '0')

    def test_fabric_check_gate(self):
        # fabric_check false skips the four-rank comparison inside start;
        # an absent key and the standalone fabric action still run it
        # (fail-closed). Each start probe ends at the startup deadline.
        def dispatch(config, action):
            with patch.object(sys, 'argv', ['fleet.py', '--config', str(config), action]), \
                 patch.object(fleet, 'preflight'), \
                 patch.object(fleet, 'fabric') as fabric, \
                 patch.object(fleet, 'remote', return_value=''), \
                 patch.object(fleet, 'request', side_effect=OSError), \
                 patch.object(fleet, 'smoke'), \
                 patch.object(fleet.time, 'monotonic', side_effect=[0, 10 ** 9]):
                if action == 'start':
                    with self.assertRaises(TimeoutError):
                        fleet.main()
                else:
                    fleet.main()
            return fabric

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'skipped.json'
            raw = json.loads((ROOT / 'cluster-r38-c8.json').read_text())
            raw['fabric_check'] = False
            p.write_text(json.dumps(raw))
            self.assertFalse(dispatch(p, 'start').called)
            self.assertTrue(dispatch(p, 'fabric').called)
            raw_default = json.loads((ROOT / 'cluster-r38-c8.json').read_text())
            raw_default.pop('fabric_check', None)
            p2 = Path(d) / 'default.json'
            p2.write_text(json.dumps(raw_default))
            self.assertTrue(dispatch(p2, 'start').called)

    def test_duplicate_node_rejected(self):
        self.c['nodes'][1] = copy.deepcopy(self.c['nodes'][0])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'bad.json'
            p.write_text(json.dumps(self.c))
            with self.assertRaises(AssertionError):
                fleet.load_config(p)

    def test_remote_failure_is_not_a_success(self):
        result = type('Result', (), {'returncode': 1, 'stdout': '', 'stderr': 'RDMA failed'})()
        with patch.object(fleet.subprocess, 'run', return_value=result):
            with self.assertRaisesRegex(RuntimeError, 'RDMA failed'):
                fleet.remote(self.c, 2, 'false')

    def test_silent_remote_failure_reports_host_and_status(self):
        result = SimpleNamespace(returncode=1, stdout='', stderr='')
        with patch.object(fleet.subprocess, 'run', return_value=result) as run:
            with self.assertRaisesRegex(RuntimeError, r'rank 0 .*exit 1'):
                fleet.remote(self.c, 0, 'test -r /missing/config.json')
        script = run.call_args.kwargs['input']
        self.assertIn('set -Eeuo pipefail', script)
        self.assertIn('$BASH_COMMAND', script)
        self.assertIn(' ERR\n', script)


class ModelTests(unittest.TestCase):
    def checkpoint(self, root, dtype='F8_E4M3', width=256):
        config = {'quantization_config': {'expert_dtype': 'fp4'},
                  'text_config': {'engram_layer_ids': [1]}}
        raw = json.dumps(config).encode()
        (root / 'config.json').write_bytes(raw)
        name = 'layers.1.engram.embed.weight'
        header = json.dumps({name: {'dtype': dtype, 'shape': [1, width],
                                    'data_offsets': [0, width]}}).encode()
        (root / 'model-1.safetensors').write_bytes(struct.pack('<Q', len(header)) + header + bytes(width))
        (root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {name: 'model-1.safetensors'}}))
        return hashlib.sha256(raw).hexdigest()

    def test_original_fp8_header_passes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            digest = self.checkpoint(root)
            with patch.object(model_check, 'CONFIG_SHA256', digest):
                model_check.validate(root)

    def test_fp4_hybrid_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            digest = self.checkpoint(root, 'U8', 128)
            with patch.object(model_check, 'CONFIG_SHA256', digest), self.assertRaises(AssertionError):
                model_check.validate(root)

    def test_truncated_shard_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            digest = self.checkpoint(root)
            shard = root / 'model-1.safetensors'
            shard.write_bytes(shard.read_bytes()[:-1])
            with patch.object(model_check, 'CONFIG_SHA256', digest), self.assertRaisesRegex(AssertionError, 'Truncated'):
                model_check.validate(root)

    def test_modified_config_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.checkpoint(root)
            with self.assertRaisesRegex(AssertionError, 'original config'):
                model_check.validate(root)


if __name__ == '__main__':
    unittest.main()
