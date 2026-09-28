"""Exercise latest source composition, async copy ownership and config rollback."""
import ast
from contextlib import contextmanager
import copy
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fleet


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


settings = load('performance_settings', ROOT / 'performance-settings.py')
ports = load('sparkring_ports', ROOT / 'patches/sparkring_performance.py')
gc_runtime = load('sparkring_gc', ROOT / 'patches/sparkring_gc_runtime.py')
checker = load('performance_checker', ROOT / 'performance-check.py')
AUDIT = ROOT / '.build/sparkring-audit-20260928'
SOURCES = {
    'vllm': Path(os.environ.get('DS41_LATEST_VLLM_SOURCE', AUDIT /
        'vllm-502d6cb5acd2ba2a62ecf58497be558c9d86089f/vllm')),
    'b12x': Path(os.environ.get('DS41_LATEST_B12X_SOURCE', AUDIT /
        'b12x-d44247b6171f7c2f9787341ae884b537887d7df9/b12x')),
}


def method(source, cls, name, scope):
    tree = ast.parse(source)
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    node = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec('from __future__ import annotations\n' + ast.unparse(node), scope)
    return scope[name]


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.source = fleet.load_config(ROOT / 'cluster-karmic-c16.json')

    def validate(self, c):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps(c))
            return fleet.load_config(path)

    def test_migration_preserves_operator_identity_and_memory_and_never_contacts_fleet(self):
        source = copy.deepcopy(self.source)
        source.update(model_path='/model with space', cache_path='/var/tmp/operator-cache',
                      ssh_identity='/home/user/private key', gpu_memory_utilization=0.83,
                      max_model_len=430080, engram_resident_scales=True,
                      max_num_batched_tokens=8192)
        before = copy.deepcopy(source)
        with tempfile.TemporaryDirectory() as directory, patch.object(fleet, 'remote') as remote:
            path, output = Path(directory)/'source.json', Path(directory)/'new.json'
            path.write_text(json.dumps(source))
            settings.write(path, output)
            c = fleet.load_config(output)
            for key in ('model_path', 'cache_path', 'ssh_identity', 'nodes', 'hcas',
                        'port', 'revision', 'gpu_memory_utilization', 'max_model_len',
                        'engram_resident_scales', 'max_num_batched_tokens'):
                self.assertEqual(c[key], before[key])
            remote.assert_not_called()
            self.assertEqual(json.loads(path.read_text()), before)
            with self.assertRaises(ValueError):
                settings.write(path, output)

    def test_each_control_reaches_every_rank_without_changing_backends(self):
        candidate = self.validate(settings.candidate(self.source))
        control = self.validate(settings.candidate(self.source, control=True))
        for rank in range(4):
            env, off = fleet.environment(candidate, rank), fleet.environment(control, rank)
            for key, variable in fleet.PERFORMANCE_ENV.items():
                self.assertEqual(env[variable], str(int(candidate[key])))
                self.assertEqual(off[variable], str(int(control[key])))
            args = fleet.serve_args(candidate, rank)
            spec = json.loads(args[args.index('--speculative-config')+1])
            self.assertEqual(spec['draft_sample_method'], 'probabilistic')
            self.assertEqual(spec['rejection_sample_method'], 'block')
            self.assertTrue(spec['enable_adaptive_verification'])
            self.assertEqual(args[args.index('--moe-backend')+1], 'b12x')
            for key in fleet.PERFORMANCE_ENV:
                changed = self.validate(dict(candidate, **{key: not candidate[key]}))
                changed_env = fleet.environment(changed, rank)
                self.assertEqual({k for k in env if env[k] != changed_env[k]},
                                 {fleet.PERFORMANCE_ENV[key]})

    def test_exact_graphs_cover_draft_and_every_adaptive_verification_depth(self):
        for depth in (0, 1, 3, 5, 7):
            for seqs in (8, 16):
                c = settings.candidate(dict(self.source, draft_tokens=depth, max_num_seqs=seqs))
                self.validate(c)
                sizes = fleet.exact_decode_graph_sizes(c)
                for width in range(1, depth+2):
                    for requests in range(1, seqs+1):
                        self.assertIn(width*requests, sizes)
                argv = fleet.serve_args(c, 0)
                graph = json.loads(argv[argv.index('--compilation-config')+1])
                self.assertEqual(graph['cudagraph_capture_sizes'], sizes)
        fixed = dict(c, enable_adaptive_verification=False)
        fixed['draft_tokens'] = 5
        argv = fleet.serve_args(fixed, 0)
        self.assertFalse(json.loads(argv[argv.index('--speculative-config')+1])['enable_adaptive_verification'])

    def test_native_projection_and_confidence_controls_validate_and_reach_engine(self):
        c = settings.candidate(self.source, overrides={
            'engram_projection_tp': True, 'adaptive_verification_cost_scale': 1.25})
        self.validate(c)
        for rank in range(4):
            argv = fleet.serve_args(c, rank)
            self.assertTrue(json.loads(argv[argv.index('--engram-config')+1])['projection_tp'])
            self.assertEqual(json.loads(argv[argv.index('--speculative-config')+1])[
                'adaptive_verification_cost_scale'], 1.25)
        for bad in ({'adaptive_verification_cost_scale': float('nan')},
                    {'adaptive_verification_cost_scale': float('inf')},
                    {'adaptive_verification_cost_scale': 0},
                    {'enable_adaptive_verification': False},
                    {'engram_projection_tp': '1'}, {'draft_tokens': 0}):
            with self.assertRaises(AssertionError):
                self.validate(dict(c, **bad))

    def test_larger_prefill_graph_and_tuning_have_independent_switches(self):
        c = settings.candidate(self.source, batch_tokens=8192, autotune=False)
        self.validate(c)
        self.assertTrue(c['prefill_8192_graph'])
        self.assertFalse(c['b12x_bounded_autotune'])
        tuned = settings.candidate(c, autotune=True)
        self.validate(tuned)
        env = fleet.environment(tuned, 0)
        self.assertEqual(env['DS41_B12X_RACE_BUDGET_MIB'], '1024')
        self.assertEqual(env['B12X_BIND_COMPILE_WORKERS'], '1')
        self.validate(dict(tuned, prefill_8192_graph=False))
        for bad in ({'h2d_staging': 'on'}, {'shm_busy_loop_s': float('nan')},
                    {'shm_busy_loop_s': True}, {'b12x_commit': 'a7d7d29b2ef8869086e0ceaa787321f17544e3c9'},
                    {'prefill_8192_graph': True, 'max_num_batched_tokens':4096}):
            with self.assertRaises(AssertionError):
                self.validate(dict(tuned, **bad))

    def test_native_source_guards_and_bundle_required_even_for_same_image_control(self):
        c = settings.candidate(self.source, control=True)
        with patch.object(fleet, 'remote', return_value='sha256:same') as remote:
            fleet.preflight(c)
        for call in remote.call_args_list:
            self.assertIn('performance-check.py', call.args[2])
            self.assertIn('local-inference.performance-bundle', call.args[2])
            self.assertIn('--expected-manifest-sha256', call.args[2])

    def test_latest_image_cannot_be_accidentally_repatched_by_legacy_builders(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'new.json'
            for filename in ('build-autotune.py', 'build-prefill-8192.py', 'configure-dspark-prefill.py'):
                legacy = load('legacy_'+filename.replace('-', '_'), ROOT/filename)
                with self.assertRaisesRegex(ValueError, 'performance-settings.py'):
                    if filename == 'configure-dspark-prefill.py':
                        legacy.configure(ROOT/'cluster-karmic-c16.json', output, 'bad:child')
                    else:
                        legacy.candidate(ROOT/'cluster-karmic-c16.json', 'bad:child')
                self.assertFalse(output.exists())


@unittest.skipUnless(all(p.is_dir() for p in SOURCES.values()), 'latest source snapshots unavailable')
class SourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.roots = {name: Path(self.temp.name)/name for name in SOURCES}
        for (name, relative) in ports.PATCHES:
            path = self.roots[name]/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((SOURCES[name]/relative).read_bytes())

    def test_patch_roundtrip_and_all_inputs_checked_before_any_write(self):
        originals = {key:(SOURCES[key[0]]/key[1]).read_bytes() for key in ports.PATCHES}
        ports.patch(**self.roots)
        ports.patch(**self.roots)
        ports.patch(**self.roots, check=True)
        ports.patch(**self.roots, revert=True)
        ports.patch(**self.roots, revert=True, check=True)
        for (name,rel), data in originals.items():
            self.assertEqual((self.roots[name]/rel).read_bytes(),data)
        last = self.roots['b12x']/'preparation/_measurement.py'
        last.write_bytes(last.read_bytes()+b'\n')
        before = {key:(self.roots[key[0]]/key[1]).read_bytes() for key in ports.PATCHES}
        with self.assertRaises(RuntimeError):
            ports.patch(**self.roots)
        self.assertFalse((self.roots['b12x']/ports.HELPER).exists())
        self.assertEqual(before,{key:(self.roots[key[0]]/key[1]).read_bytes() for key in ports.PATCHES})

    def test_composition_matches_immutable_manifest_and_rejects_final_drift(self):
        manifest = ROOT/'patches/performance-manifest.json'
        data = json.loads(manifest.read_text())
        for entry in data['files']:
            if entry['source_sha256'] is not None:
                path = self.roots[entry['tree']]/entry['path']
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes((SOURCES[entry['tree']]/entry['path']).read_bytes())
        for name in ('roce_karmic','dspark_prefill','karmic_autotune','prefill_8192_graph','sparkring_performance','knapcio_decode'):
            m = load('composed_'+name,ROOT/'patches'/f'{name}.py')
            if name in ('roce_karmic','knapcio_decode'): m.patch(self.roots['b12x'])
            elif name in ('dspark_prefill','prefill_8192_graph'): m.patch(self.roots['vllm'])
            else: m.patch(**self.roots)
        self.assertEqual(checker.check(self.roots,manifest),len(data['files']))
        expected = hashlib.sha256(manifest.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        self.assertEqual(checker.check(self.roots, manifest, expected_manifest_sha256=expected), len(data['files']))
        truncated = Path(self.temp.name)/'old-manifest.json'
        truncated.write_text(json.dumps(dict(data, files=data['files'][:-2])))
        with self.assertRaisesRegex(ValueError, 'operator checkout'):
            checker.check(self.roots, truncated, expected_manifest_sha256=expected)
        drift = self.roots['vllm']/'v1/worker/gpu/model_runner.py'
        drift.write_bytes(drift.read_bytes()+b'\n')
        with self.assertRaisesRegex(ValueError,'source mismatch'):
            checker.check(self.roots,manifest)

    def test_real_async_copy_keeps_two_batches_alive_until_queued_reads_finish(self):
        ports.patch(**self.roots)
        source = (self.roots['vllm']/'v1/utils.py').read_text()
        queued, allocations = [], []

        class Tensor:
            def __init__(self, data, gpu=False):
                self.data, self.gpu = data, gpu
            def __getitem__(self, key):
                return Tensor(self.data[key],self.gpu)
            def pin_memory(self):
                return self
            def copy_(self, source, non_blocking=False):
                if self.gpu and non_blocking:
                    queued.append(lambda:self.data.__setitem__(slice(None),source.data))
                else:
                    self.data[:] = source.data
                return self

        def allocate(cpu, pin_memory):
            self.assertTrue(pin_memory)
            allocations.append(cpu.data.size)
            return Tensor(np.empty_like(cpu.data))

        scope = dict(os=os,torch=NS(empty_like=allocate),PIN_MEMORY=True)
        copy_to_gpu = method(source,'CpuGpuBuffer','copy_to_gpu',scope)
        for enabled in ('0','1'):
            for n in (None,0,3):
                with patch.dict(os.environ,DS41_H2D_STAGING=enabled):
                    queued.clear();allocations.clear()
                    cpu = Tensor(np.arange(5))
                    first, second = Tensor(np.zeros(5,dtype=int),True),Tensor(np.zeros(5,dtype=int),True)
                    copy_to_gpu(NS(cpu=cpu,gpu=first),n)
                    cpu.data[:] += 100
                    copy_to_gpu(NS(cpu=cpu,gpu=second),n)
                    cpu.data[:] += 100
                    for event in queued: event()
                    count = 5 if n is None else n
                    self.assertEqual(first.data[:count].tolist(),list(range(0 if enabled=='1' else 200,count+(0 if enabled=='1' else 200))))
                    self.assertEqual(second.data[:count].tolist(),list(range(100 if enabled=='1' else 200,count+(100 if enabled=='1' else 200))))
                    self.assertEqual(len(allocations),2 if enabled=='1' else 0)

    def test_real_ipc_default_explicit_override_and_idle_transition(self):
        ports.patch(**self.roots)
        source = (self.roots['vllm']/'distributed/device_communicators/shm_broadcast.py').read_text()
        calls = []
        class Socket(NS):
            __hash__ = object.__hash__
            __eq__ = object.__eq__

        socket = Socket(setsockopt=lambda *a: None, setsockopt_string=lambda *a: None,
                    connect=lambda *a: None, bind=lambda *a: None,
                    recv=lambda **k: calls.append('recv'))
        poller = NS(register=lambda *a: None, poll=lambda **k: calls.append('poll') or [])
        scope = dict(os=os, time=time, zmq=NS(CONFLATE=1, POLLIN=2, PAIR=3, NOBLOCK=4,
                                            Poller=lambda: poller), SUB=5, SUBSCRIBE=6,
                     get_open_zmq_inproc_path=lambda: 'inproc://fake',
                     sched_yield=lambda: calls.append('yield'), logger=NS(debug=lambda *a: None))
        init = method(source, 'SpinCondition', '__init__', scope)
        wait = method(source, 'SpinCondition', 'wait', scope)
        context = NS(socket=lambda *a: socket)
        for value, explicit, expected in (('1', None, 1), ('0', None, 0),
                                          ('0.01', None, 0.01), ('invalid', 0.2, 0.2)):
            with patch.dict(os.environ, DS41_SHM_BUSY_LOOP_S=value):
                owner = NS()
                init(owner, True, context, 'fake', explicit)
                self.assertEqual(owner.busy_loop_s, expected)
                for age, action in ((0.005, 'yield'), (2, 'poll')):
                    calls.clear()
                    now = time.monotonic()
                    owner.last_read = now - age
                    with patch.object(time, 'monotonic', return_value=now):
                        wait(owner, 10)
                    self.assertEqual(calls, ['poll' if expected == 0 else action])
        for value in ('nan', 'inf', '-1', '2', 'bad'):
            with patch.dict(os.environ, DS41_SHM_BUSY_LOOP_S=value):
                with self.assertRaisesRegex(ValueError, 'between 0 and 1'):
                    init(NS(), True, context, 'fake')

    def test_real_bounded_hashes_preserve_cache_events_and_partial_promotion(self):
        ports.patch(**self.roots)
        source = (self.roots['vllm']/'v1/core/block_pool.py').read_text()
        copies, calls = [], []

        class Hashes:
            def __getitem__(self, key):
                result = list(range(1024))[key]
                if isinstance(key, slice):
                    copies.append(len(result))
                return result

        scope = dict(os=os, resolve_block_hashes=lambda *a: Hashes(),
                     make_block_hash_with_group_id=lambda h, g: (h, g))
        cache = method(source, 'BlockPool', 'cache_full_blocks', scope)
        owner = NS(hash_block_size=256, enable_kv_cache_events=True,
                   _insert_block_hash=lambda h, b, **k: calls.append(('insert', h, b.id, k)),
                   _remove_cached_block_hashes=lambda b: [('old', b.id)],
                   _emit_block_removed_events=lambda h: calls.append(('remove', h)),
                   _emit_stored_block_runs=lambda r, ids, *a, **k: calls.append(('event', ids, a, k)),
                   _published_full_block_context_start=lambda *a: 0)
        blocks = [NS(id=i, is_null=i % 17 == 0, block_hash='old' if i % 19 == 0 else None,
                     block_hash_num_tokens=1) for i in range(1024)]
        results = []
        for flag in ('0', '1'):
            copies.clear(); calls.clear()
            with patch.dict(os.environ, DS41_BOUNDED_PREFIX_HASHES=flag):
                for start in range(0, 1024, 16):
                    mask = None if start % 32 else [i % 3 != 0 for i in range(16)]
                    cache(owner, NS(block_hashes=[]), blocks, start, start+16, 256, 2, mask)
                cache(owner, NS(block_hashes=[]), blocks, 1024, 1024, 256, 2)
            results.append((list(calls), sum(copies)))
        self.assertEqual(results[0][0], results[1][0])
        self.assertEqual(results[1][1], 1024)
        self.assertGreater(results[0][1], 32000)

    def test_actual_gate_restores_gc_only_after_nested_waits_release(self):
        ports.patch(**self.roots)
        source = (self.roots['b12x']/'preparation/_measurement.py').read_text()
        hold = method(source,'_StreamGate','hold',dict(contextmanager=contextmanager,ds41_gc=gc_runtime))
        def owner(failure=None):
            def wait(*args):
                if failure: raise RuntimeError('enqueue')
                return (0,)
            return NS(streams={},sequence=0,device_pointer=1,flag=NS(value=0),
                      _check=lambda x:None,driver=NS(cuStreamWaitValue32=wait,
                         CUstreamWaitValue_flags=NS(CU_STREAM_WAIT_VALUE_GEQ=0)))
        originally_enabled = gc.isenabled()
        try:
            for initial in (False,True):
                gc.enable() if initial else gc.disable()
                with patch.dict(os.environ,DS41_DEFER_AUTOTUNE_GC='1'):
                    first,second = owner(),owner()
                    with hold(first,NS(cuda_stream=1)):
                        with hold(second,NS(cuda_stream=2)):
                            self.assertFalse(gc.isenabled())
                        self.assertFalse(gc.isenabled())
                        self.assertEqual(second.flag.value,1)
                    self.assertEqual(first.flag.value,1)
                    self.assertEqual(gc.isenabled(),initial)
                    for fail in ('enqueue','body'):
                        o = owner(fail if fail=='enqueue' else None)
                        with self.assertRaisesRegex(RuntimeError,fail):
                            with hold(o,NS(cuda_stream=3)):
                                raise RuntimeError('body')
                        self.assertEqual(o.flag.value,1)
                        self.assertEqual(gc.isenabled(),initial)
            gc.enable()
            with patch.dict(os.environ,DS41_DEFER_AUTOTUNE_GC='0'):
                with hold(owner(),NS(cuda_stream=4)):
                    self.assertTrue(gc.isenabled())
        finally:
            gc.enable() if originally_enabled else gc.disable()


if __name__ == '__main__':
    unittest.main()
