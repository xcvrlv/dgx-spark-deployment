"""Contract checks against actual patched methods, without a CUDA dependency."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
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


p = load('dspark_patch', ROOT / 'patches/dspark_prefill.py')
r = load('dspark_runtime', ROOT / 'patches/dspark_prefill_runtime.py')
SOURCE = Path(os.environ.get('DS41_KARMIC_VLLM_SOURCE', ROOT /
    '.build/upstream/vllm-1794dcf18454900263e0c66711af8ea4a1283ac1/vllm'))


class Tensor(np.ndarray):
    def copy_(self, source):
        self[...] = source
        return self

    def zero_(self):
        self.fill(0)
        return self

    def fill_(self, value):
        self.fill(value)
        return self

    def masked_fill(self, mask, value):
        result = self.copy()
        result[mask] = value
        return result

    def numel(self):
        return self.size


def tensor(value):
    return np.asarray(value).view(Tensor)


def gather(source, indices):
    result = source[np.maximum(indices, 0)].copy()
    result[indices < 0] = 0
    return result


def config():
    return NS(parallel_config=NS(data_parallel_size=1, pipeline_parallel_size=1,
        prefill_context_parallel_size=1, decode_context_parallel_size=1),
        scheduler_config=NS(max_num_batched_tokens=4096),
        compilation_config=NS(max_cudagraph_capture_size=96))


def batch(scheduled=(4096,), computed=(0,), lengths=(8192,)):
    return NS(num_reqs=len(scheduled), num_tokens=sum(scheduled),
        num_scheduled_tokens=np.array(scheduled), num_draft_tokens=0,
        num_computed_prefill_tokens_np=np.array(computed),
        prefill_len_np=np.array(lengths),
        is_prefilling_np=np.array(computed) < np.array(lengths),
        has_prefill=any(c < n for c, n in zip(computed, lengths)),
        has_structured_output_reqs=False,
        seq_lens_cpu_upper_bound=tensor(np.array(computed) + scheduled))


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, DS41_SKIP_PREFILL_DRAFT='1', DS41_COMPACT_CONTEXT_GRAPH='1')
        self.env.start()
        self.addCleanup(self.env.stop)
        self.owner = NS(_speculator_name='DSpark', pcp_manager=None, vllm_config=config(),
            model=NS(capture_context_preparation=lambda: None))

    def skip(self, b, **kw):
        args = dict(dummy_run=False, is_profile=False, dp_sync=None, context_kv_is_restored=False)
        args.update(kw)
        return r.skip_prefill_draft(self.owner, b, **args)

    def test_chunk_boundaries_cached_prefix_and_mixed_batches(self):
        self.assertTrue(self.skip(batch()))
        self.assertTrue(self.skip(batch(computed=(4095,))))  # one prompt token remains
        self.assertFalse(self.skip(batch(computed=(4096,))))  # exactly final chunk
        self.assertFalse(self.skip(batch(computed=(5000,))))  # upper bound overshoots
        self.assertTrue(self.skip(batch((1024,), (6144,), (8192,))))  # cached prefix
        self.assertFalse(self.skip(batch((1,), (8192,), (8192,))))  # decode
        self.assertFalse(self.skip(batch((4096, 1), (0, 8192), (8192, 8192))))
        self.assertTrue(self.skip(batch((1024, 2048), (0, 0), (8192, 8192))))
        self.assertFalse(self.skip(batch((1024, 2048), (0, 6144), (8192, 8192))))
        self.assertFalse(self.skip(batch((0,), (0,), (8192,))))
        self.assertFalse(self.skip(batch((), (), ())))

    def test_conservative_fallbacks_and_independent_flags(self):
        for kw in ({'dummy_run': True}, {'is_profile': True}, {'dp_sync': object()},
                   {'context_kv_is_restored': True}):
            self.assertFalse(self.skip(batch(), **kw))
        for field, value in [('has_structured_output_reqs', True), ('num_draft_tokens', 1)]:
            b = batch()
            setattr(b, field, value)
            self.assertFalse(self.skip(b))
        for field in vars(self.owner.vllm_config.parallel_config):
            setattr(self.owner.vllm_config.parallel_config, field, 2)
            self.assertFalse(self.skip(batch()))
            self.assertFalse(r.use_compact_context_graph(self.owner, batch(), 128))
            setattr(self.owner.vllm_config.parallel_config, field, 1)
        for skip, graph in [('0', '0'), ('1', '0'), ('0', '1'), ('1', '1')]:
            with patch.dict(os.environ, DS41_SKIP_PREFILL_DRAFT=skip, DS41_COMPACT_CONTEXT_GRAPH=graph):
                self.assertEqual(self.skip(batch()), skip == '1')
                self.assertEqual(r.use_compact_context_graph(self.owner, batch(), 128), graph == '1')
        self.owner._speculator_name = 'DFlash'
        self.assertFalse(self.skip(batch()))
        self.assertFalse(r.use_compact_context_graph(self.owner, batch(), 128))
        with patch.dict(os.environ, DS41_SKIP_PREFILL_DRAFT='0', DS41_COMPACT_CONTEXT_GRAPH='0'):
            self.assertFalse(r.use_compact_context_graph(NS(), NS(), 128))
            self.assertFalse(r.skip_prefill_draft(NS(), NS(), dummy_run=False,
                is_profile=False, dp_sync=None, context_kv_is_restored=False))

    def test_compact_graph_only_for_supported_single_request_shape(self):
        for b, rows, expected in [(batch(), 128, True), (batch(), 4096, False),
                (batch((64,), (0,), (8192,)), 64, False),
                (batch((2048, 2048), (0, 0), (8192, 8192)), 256, False),
                (batch((6,), (8192,), (8192,)), 128, False)]:
            self.assertEqual(r.use_compact_context_graph(self.owner, b, rows), expected)


@unittest.skipUnless(SOURCE.is_dir(), 'set DS41_KARMIC_VLLM_SOURCE to pinned package')
class PatchedSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for relative in p.PATCHES:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((SOURCE / relative).read_bytes())
        p.patch(self.root)

    def extract(self, relative, name, method=None, scope=None):
        tree = ast.parse((self.root / relative).read_text())
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
        if method:
            node = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == method)
            node.decorator_list = []
        scope = scope or {}
        exec('from __future__ import annotations\n' + ast.unparse(node), scope)
        return scope[method or name]

    def test_hash_guard_roundtrip_and_no_partial_write_on_drift(self):
        p.patch(self.root)
        p.patch(self.root, check=True)
        p.patch(self.root, revert=True)
        p.patch(self.root, revert=True, check=True)
        for relative in p.PATCHES:
            self.assertEqual((self.root / relative).read_bytes(), (SOURCE / relative).read_bytes())
        last = self.root / list(p.PATCHES)[-1]
        last.write_bytes(last.read_bytes() + b'\n')
        before = {rel: (self.root / rel).read_bytes() for rel in p.PATCHES}
        with self.assertRaises(RuntimeError):
            p.patch(self.root)
        self.assertFalse((self.root / p.RUNTIME).exists())
        self.assertEqual(before, {rel: (self.root / rel).read_bytes() for rel in p.PATCHES})

    def test_real_propose_context_before_skip_and_query_metadata_preserved(self):
        for skip, graph, final, mixed, restored in [
                (False, False, False, False, False), (True, False, False, False, False),
                (False, True, False, False, False), (True, True, False, False, False),
                (True, True, True, False, False), (True, True, False, True, False),
                (True, True, False, False, True)]:
            with self.subTest(skip=skip, graph=graph, final=final, mixed=mixed, restored=restored):
                self.run_propose(skip, graph, final, mixed, restored)

    def run_propose(self, skip, graph, final, mixed, restored):
        events, writes = [], []
        b = batch(computed=(4096 if final else 0,)) if not mixed else batch(
            (2048, 1), (0, 8192), (8192, 8192))
        indices = tensor(np.r_[np.arange(b.num_tokens - 128, b.num_tokens - 2), -1, -1])
        hidden = tensor(np.zeros((4096, 2)))
        positions = tensor(np.zeros(4096, dtype=np.int64))
        slots = tensor(np.zeros((2, 4096), dtype=np.int64))
        query_slots = tensor(np.full((2, 4096), -900))
        query_positions = tensor(np.full(32, -800))
        aux = [tensor(np.arange(b.num_tokens * 2).reshape(-1, 2) + i) for i in range(3)]

        def combine(x):
            events.append('combine')
            return tensor(x.reshape(-1, 3, 2).sum(axis=1))

        def precompute(h, pos, groups=None):
            events.append('context')
            writes.append((h.copy(), pos.copy(), [x.copy() for x in groups]))

        def prepare(*args):
            gid = len([e for e in events if e == 'prepare'])
            events.append('prepare')
            positions[:b.num_tokens] = np.arange(b.num_tokens) + 100
            slots[gid, :b.num_tokens] = np.arange(b.num_tokens) + 256 * (gid + 1)
            slots[gid, b.num_tokens - 3] = -1  # a null/evicted page
            query_slots[gid, :10] = 42 + gid
            query_positions[:10] = 7000

        def graph_run(states, rows):
            events.append('graph')
            precompute(combine(tensor(np.concatenate(states, axis=-1))),
                positions[:rows], [slots[i, :rows] for i in (1, 0, 1)])

        owner = NS(_speculator_name='DSpark', pcp_manager=None, vllm_config=config(),
            model=NS(combine_hidden_states=combine, precompute_and_store_context_kv=precompute,
                capture_context_preparation=lambda: None),
            model_state=NS(get_ced_indices=lambda: indices),
            _context_preparer=NS(can_run=lambda n: n <= 128, run=graph_run),
            _layer_group_idx=[1, 0, 1], hidden_states=hidden, context_positions=positions,
            _context_slot_mappings=slots, draft_kv_cache_group_id=0,
            draft_kv_cache_group_ids=[0, 1], num_query_per_req=5, num_speculative_steps=5,
            max_num_reqs=16, max_num_tokens=4096, max_model_len=10000,
            block_tables=NS(get_group_cp_parameters=lambda gid: (0, 1, 1),
                slot_mappings=query_slots, input_block_tables=[None, None], kernel_block_sizes=[128, 128]),
            input_buffers=NS(positions=query_positions), sample_indices=None, sample_pos=None,
            sample_idx_mapping=None, temperature=None, seeds=None, parallel_drafting_token_id=0,
            sample_from_anchor=True, dp_size=1, dp_rank=0, query_cudagraph_manager=None,
            _build_uniform_attn_metadata=lambda **kw: None, _group_causal=False, kv_cache_config=None,
            _prepare_eplb_forward=lambda n: None, _generate_draft=lambda *a, **kw: events.append('query'),
            draft_tokens=tensor(np.ones((16, 5), dtype=np.int64)))
        scope = dict(logger=NS(info_once=lambda *a: None), skip_prefill_draft=r.skip_prefill_draft,
            use_compact_context_graph=r.use_compact_context_graph, PAD_SLOT_ID=-1,
            torch=NS(cat=lambda xs, dim: tensor(np.concatenate(xs, axis=dim))),
            prepare_dflash_inputs=prepare, CUDAGraphMode=NS(FULL='full', NONE='none'),
            dispatch_cg_and_sync_dp=lambda *a, **kw: (NS(num_tokens=5*b.num_reqs, cg_mode='none'), None),
            build_slot_mappings_by_layer=lambda *a: None)
        propose = self.extract('v1/worker/gpu/spec_decode/dflash/speculator.py',
            'DFlashSpeculator', 'propose', scope)
        ced_module = NS(gather_rows=gather)
        with patch.dict(sys.modules, {'vllm.models.deepseek_v4_1.ced': ced_module}), patch.dict(
                os.environ, DS41_SKIP_PREFILL_DRAFT=str(int(skip)), DS41_COMPACT_CONTEXT_GRAPH=str(int(graph))):
            result = propose(owner, b, {}, {}, aux[0], aux, None, None, None, None, None, None,
                context_kv_is_restored=restored)
        skipped = skip and not final and not mixed and not restored
        self.assertEqual(result.shape, (b.num_reqs, 0 if skipped else 5))
        self.assertEqual(events.count('query'), 0 if skipped else 1)
        self.assertEqual(events.count('context'), 0 if restored else 1)
        self.assertEqual(events.count('graph'), int(graph and not mixed and not restored))
        np.testing.assert_array_equal(query_slots[:, :10], np.array([[42]*10, [43]*10]))
        np.testing.assert_array_equal(query_slots[:, 10:], -900)
        np.testing.assert_array_equal(query_positions[:10], 7000)
        if not restored:
            h, pos, groups = writes[0]
            np.testing.assert_array_equal(h, sum(gather(x, indices) for x in aux))
            np.testing.assert_array_equal(pos, gather(tensor(np.arange(b.num_tokens) + 100), indices))
            for slot, gid in zip(groups, (1, 0, 1)):
                expected = gather(tensor(np.arange(b.num_tokens) + 256 * (gid + 1)), indices)
                expected[(indices < 0) | (indices == b.num_tokens - 3)] = -1
                np.testing.assert_array_equal(slot, expected)
            if not skipped:
                self.assertLess(events.index('context'), events.index('query'))

    def test_context_capacity_preserves_decode_buckets_and_bounds_new_storage(self):
        class Manager:
            def __init__(self, c, *a, **kw):
                self.compilation_config = c.compilation_config
        scope = dict(copy=copy, COMPACT_ROWS=128, compact_context_enabled=r.compact_context_enabled,
            torch=NS(zeros=lambda shape, **kw: np.zeros(shape)), CudaGraphManager=Manager,
            CUDAGraphMode=NS(FULL='full'))
        context = self.extract('models/deepseek_v4_1/nvidia/dspark.py', 'DSparkContextCudaGraphs', scope=scope)
        for enabled, expected in [('0', [1, 2, 4, 8, 16, 32, 64, 96]),
                                  ('1', [1, 2, 4, 8, 16, 32, 64, 96, 128])]:
            c = config()
            with patch.dict(os.environ, DS41_COMPACT_CONTEXT_GRAPH=enabled):
                owner = context(NS(config=NS(hidden_size=5120, dspark_target_layer_ids=(37, 38, 39))),
                    c, NS(shape=(4096, 5120), dtype='bf16', device='cuda'), None, None, None, 96)
            self.assertEqual(owner.manager.compilation_config.cudagraph_capture_sizes, expected)
            self.assertEqual(owner.aux.shape, (expected[-1], 15360))
            self.assertEqual(c.compilation_config.max_cudagraph_capture_size, 96)

    def test_real_context_run_scrubs_padding_and_restored_context_is_noop(self):
        from collections import namedtuple
        Desc = namedtuple('Desc', 'num_tokens')
        context = self.extract('models/deepseek_v4_1/nvidia/dspark.py',
            'DSparkContextCudaGraphs', scope={'PAD_SLOT_ID': -1})
        owner = context.__new__(context)
        owner.width, owner.num_aux = 2, 3
        owner.aux = tensor(np.zeros((128, 6)))
        owner.hidden_states = tensor(np.zeros((128, 2)))
        owner.positions = tensor(np.zeros(128, dtype=np.int64))
        owner.slot_mappings = tensor(np.zeros((2, 128), dtype=np.int64))
        owner.layer_group_idx = [1, 0, 1]
        pool = tensor(np.full((3, 256, 2), -7.0))
        def write(hidden, positions, slots):
            self.assertLess(int(positions.max()), 512)
            for i, mapping in enumerate(slots):
                valid = mapping >= 0
                pool[i, mapping[valid]] = hidden[valid] + positions[valid, None]
        owner.model = NS(combine_hidden_states=lambda x: tensor(x.reshape(-1, 3, 2).sum(axis=1)),
            precompute_and_store_context_kv=write)
        capacities = [1, 2, 4, 8, 16, 32, 64, 96, 128]
        owner.manager = NS(graphs={Desc(n): None for n in capacities},
            dispatch=lambda reqs, rows, *a: Desc(next((n for n in capacities if n >= rows), rows)),
            run_fullgraph=lambda desc: owner._forward(desc.num_tokens))
        for rows in (128, 6, 3, 96, 1):
            aux = [tensor(np.arange(rows * 2).reshape(rows, 2) + i) for i in range(3)]
            owner.positions.fill_(1000000)
            owner.positions[:rows] = np.arange(rows) + 1
            owner.slot_mappings[:] = np.arange(128) + np.array([[32], [64]])
            owner.slot_mappings[:, rows - 1] = -1
            original_slots = owner.slot_mappings[:, :rows].copy()
            pool.fill_(-7)
            owner.run(aux, rows)
            expected = np.full((3, 256, 2), -7.0)
            main = sum(aux)
            for layer, group in enumerate(owner.layer_group_idx):
                valid = original_slots[group] >= 0
                expected[layer, original_slots[group][valid]] = main[valid] + np.arange(1, rows + 1)[valid, None]
            np.testing.assert_array_equal(pool, expected)
            np.testing.assert_array_equal(owner.hidden_states[:rows], main)
            for source in aux:
                source.fill_(-99999)
            owner.slot_mappings.fill_(0)
            owner.run(aux, rows, context_kv_is_restored=True)
            np.testing.assert_array_equal(pool, expected)
        self.assertFalse(owner.can_run(129))
        with self.assertRaises(ValueError):
            owner.run(aux, 129)

    def test_empty_drafts_do_not_publish_stale_confidence(self):
        tree = ast.parse((self.root / 'v1/worker/gpu/model_runner.py').read_text())
        guard = next(n for n in ast.walk(tree) if isinstance(n, ast.If) and
            any(isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call) and
                isinstance(stmt.value.func, ast.Attribute) and stmt.value.func.attr == 'record_confidences'
                for stmt in n.body))
        expression = compile(ast.Expression(guard.test), '<confidence guard>', 'eval')
        for count in (0, 1, 5):
            self.assertEqual(eval(expression, {'self': NS(adaptive_verification=object()),
                'num_draft_tokens': count}), count > 0)


class FleetControlsTests(unittest.TestCase):
    def test_config_helper_preserves_operator_recipe_and_refuses_overwrite(self):
        helper = load('configure_dspark_prefill', ROOT / 'configure-dspark-prefill.py')
        source = json.loads((ROOT / 'cluster-karmic-20260925.json').read_text())
        source.update(engram_resident_scales=True, gpu_memory_utilization=0.87,
                      adaptive_speculative_tokens_window=100)
        with tempfile.TemporaryDirectory() as directory:
            old, new = Path(directory) / 'old.json', Path(directory) / 'new.json'
            old.write_text(json.dumps(source))
            before = old.read_bytes()
            helper.configure(old, new, 'candidate:v1', skip=False, graph=True)
            candidate = fleet.load_config(new)
            changed = {'image', 'dspark_skip_prefill_draft', 'dspark_compact_context_graph', 'graph_memory_debug'}
            self.assertEqual({k: v for k, v in candidate.items() if k not in changed},
                             {k: v for k, v in source.items() if k not in changed})
            self.assertEqual(old.read_bytes(), before)
            self.assertFalse(candidate['dspark_skip_prefill_draft'])
            self.assertTrue(candidate['dspark_compact_context_graph'])
            with self.assertRaises(ValueError):
                helper.configure(old, new, 'candidate:v1')
            with self.assertRaises(ValueError):
                helper.configure(old, old, 'candidate:v1')
            with self.assertRaises(ValueError):
                helper.configure(old, Path(directory) / 'same-image.json', source['image'])

    def test_flags_validation_and_image_gate(self):
        c = json.loads((ROOT / 'cluster-karmic-c16.json').read_text())
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'config.json'
            for skip, graph in ((False, False), (True, False), (False, True), (True, True)):
                candidate = dict(c, dspark_skip_prefill_draft=skip, dspark_compact_context_graph=graph)
                path.write_text(json.dumps(candidate))
                checked = fleet.load_config(path)
                env = fleet.environment(checked, 0)
                self.assertEqual(env['DS41_SKIP_PREFILL_DRAFT'], str(int(skip)))
                self.assertEqual(env['DS41_COMPACT_CONTEXT_GRAPH'], str(int(graph)))
                self.assertEqual(fleet.serve_args(checked, 0), fleet.serve_args(c, 0))
            for changes in ({'draft_tokens': 0}, {'vllm_commit': '0'*40}, {'b12x_commit': '0'*40},
                            {'dspark_skip_prefill_draft': 'false'}, {'upstream_branch': 'dev/jovian-judgement'}):
                path.write_text(json.dumps(dict(c, dspark_skip_prefill_draft=True) | changes))
                with self.assertRaises(AssertionError):
                    fleet.load_config(path)
        with patch.object(fleet, 'remote', return_value='same-image') as remote:
            fleet.preflight(dict(c, dspark_skip_prefill_draft=True))
        self.assertEqual(remote.call_count, 4)
        for call in remote.call_args_list:
            self.assertIn('local-inference.dspark-prefill-overlay', call.args[2])
            self.assertIn('ds41-dspark-prefill-v1', call.args[2])


if __name__ == '__main__':
    unittest.main()
