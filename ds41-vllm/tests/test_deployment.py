import copy
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

    def test_draft_capture_covers_every_c8_depth(self):
        self.c['draft_tokens'] = 3
        args = fleet.serve_args(self.c, 0)
        config = json.loads(args[args.index('--compilation-config') + 1])
        self.assertEqual(config['cudagraph_capture_sizes'], list(range(1, 33)))
        draft = json.loads(args[args.index('--speculative-config') + 1])
        self.assertEqual(draft['draft_tensor_parallel_size'], 4)

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
