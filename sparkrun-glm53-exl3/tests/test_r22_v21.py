"""CPU regression gates for the archived R7 paired FC2 M8 weight reuse."""
import ast
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT.parent/'tmp/safe-v20-feasibility/v20/kernel.py'
MIXED = ROOT.parent/'tmp/safe-v20-feasibility/v20/mixed_trellis.py'
V16_KERNEL = ROOT.parent/'tmp/exl3-v16/base/b12x/moe/_shared/kernels/w4a16/kernel.py'
V19_MIXED = ROOT.parent/'tmp/exl3-v11/patched/b12x/moe/_shared/kernels/w4a16/mixed_trellis.py'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


p = load('patch_r22_v21', ROOT/'overlay/patch_r22_v21.py')
v16 = load('patch_r22_v16_reloaded', ROOT/'overlay/patch_r22_v16.py')
v19 = load('patch_r22_v19_reloaded', ROOT/'overlay/patch_r22_v19.py')
v20 = load('patch_r22_v20_reloaded', ROOT/'overlay/patch_r22_v20.py')


def extract(source, name, ns):
    node = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec('from __future__ import annotations\n'+ast.unparse(node), ns)
    return ns[name]


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (KERNEL.exists() and MIXED.exists()):
            raise unittest.SkipTest('pinned v20-state fixtures unavailable')
        cls.k_before = KERNEL.read_text(encoding='utf-8')
        cls.m_before = MIXED.read_text(encoding='utf-8')
        assert hashlib.sha256(cls.k_before.encode()).hexdigest() == p.INPUTS[p.KERNEL]
        assert hashlib.sha256(cls.m_before.encode()).hexdigest() == p.INPUTS[p.MIXED]
        # Close the fixture chain: the v16 sigmoid merge and the v19/v20
        # overlays reproduce both pinned v20-state snapshots.
        if V16_KERNEL.exists():
            source = V16_KERNEL.read_text(encoding='utf-8')
            assert hashlib.sha256(source.encode()).hexdigest() == v16.INPUTS[v16.KERNEL]
            chained = v20.transform(v16.kernel(source))
            assert hashlib.sha256(chained.encode()).hexdigest() == p.INPUTS[p.KERNEL], \
                'kernel fixture chain broke'
        if V19_MIXED.exists():
            source = V19_MIXED.read_text(encoding='utf-8')
            assert hashlib.sha256(source.encode()).hexdigest() == v19.INPUTS[v19.MIXED]
            chained = v19.transform(source)
            assert hashlib.sha256(chained.encode()).hexdigest() == p.INPUTS[p.MIXED], \
                'mixed fixture chain broke'
        cls.k_after = p.transform_kernel(cls.k_before)
        cls.m_after = p.transform_mixed(cls.m_before)

    def test_hash_preflight_idempotence_and_rejection(self):
        self.assertEqual(hashlib.sha256(self.k_before.encode()).hexdigest(), p.INPUTS[p.KERNEL])
        self.assertEqual(p.INPUTS[p.KERNEL], v20.OUTPUTS[v20.KERNEL])
        self.assertEqual(hashlib.sha256(self.k_after.encode()).hexdigest(), p.OUTPUTS[p.KERNEL])
        self.assertEqual(hashlib.sha256(self.m_before.encode()).hexdigest(), p.INPUTS[p.MIXED])
        self.assertEqual(p.INPUTS[p.MIXED], v19.OUTPUTS[v19.MIXED])
        self.assertEqual(hashlib.sha256(self.m_after.encode()).hexdigest(), p.OUTPUTS[p.MIXED])
        compile(self.k_after, 'kernel.py', 'exec')
        compile(self.m_after, 'mixed_trellis.py', 'exec')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for rel, before, after in ((p.KERNEL, self.k_before, self.k_after),
                                       (p.MIXED, self.m_before, self.m_after)):
                target = root/rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(before, encoding='utf-8')
                with self.assertRaises(RuntimeError): p.patch(root, check=True)
            p.patch(root)
            p.patch(root)
            p.patch(root, check=True)
            target.write_text(before+'\n# unknown edit\n', encoding='utf-8')
            previous = target.read_bytes()
            with self.assertRaises(RuntimeError): p.patch(root)
            self.assertEqual(target.read_bytes(), previous)

    def test_helper_architecture_and_value_gating(self):
        def helper(capability=(12, 1)):
            return extract(self.k_after, '_gb10_fc2_m8_pair_enabled',
                           dict(os=os, torch=NS(cuda=NS(get_device_capability=lambda: capability))))
        gb10, other = helper(), helper((12, 0))
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(gb10())
        for value, expected in (('0', False), ('1', True)):
            with patch.dict(os.environ, VLLM_GB10_EXL3_FC2_M8_PAIR=value):
                self.assertEqual(gb10(), expected)
                self.assertFalse(other(), 'non-SM121 must ignore the switch')
        for invalid in ('2', '-1', 'true', 'typo'):
            with patch.dict(os.environ, VLLM_GB10_EXL3_FC2_M8_PAIR=invalid):
                with self.assertRaises(ValueError): gb10()

    def test_gemm_validation_rejects_ineligible_geometry(self):
        text = self.k_after
        start = '        # v21 archived R7 paired FC2. The switch may only ever reach\n'
        end = '                "with an even schedule_route_block_factor of 2 or 4"\n            )\n'
        self.assertEqual(text.count(start), 1)
        block = text[text.index(start):text.index(end)+len(end)]
        dedented = '\n'.join(line[8:] if line.startswith('        ') else line
                             for line in block.splitlines())

        def construct(flag, block_size, whole=True, factor=2):
            self_ns = NS(uses_m_block_8=block_size == 8, schedule_whole_tiles=whole,
                         schedule_route_block_factor=factor)
            ns = dict(self=self_ns, paired_m8_routes=flag,
                      moe_block_size=block_size, int=int, ValueError=ValueError)
            exec(dedented, ns)
            return self_ns.paired_m8_routes
        self.assertIs(construct(True, 8, True, 2), True)
        self.assertIs(construct(True, 8, True, 4), True)
        self.assertIs(construct(False, 8), False)
        for args in ((True, 16), (True, 32), (True, 8, False), (True, 8, True, 1),
                     (True, 8, True, 3)):
            with self.assertRaises(ValueError):
                construct(*args)

    def test_pair_dispatch_model_keeps_expert_boundaries_and_locks(self):
        """Python model of the v21 dispatcher conditional: each pair call
        covers two adjacent M8 subtiles within one packed block, and every
        subtile keeps exactly one consecutive lock slot in both arms."""
        for block in (16, 32, 64):
            for factor in (2, 4):
                group = min(factor, block // 8)
                sub = 8
                self.assertEqual(block % (sub*group), 0)
                visited = []
                slots = set()
                for job in range(11*block//(sub*group)):
                    expert_block = job//(block//(sub*group))
                    if group == 2:
                        calls = ((0, 0), (0, 1))
                    else:
                        calls = ((0, 0), (0, 1), (2, 0), (2, 1))
                    for base, half in calls:
                        subtile = (job*group+base) + half
                        lock = job*group + base + half
                        slots.add(lock)
                        for row in range(subtile*sub, (subtile+1)*sub):
                            self.assertEqual(row//block, expert_block)
                            visited.append(row)
                self.assertEqual(visited, list(range(11*block)))
                self.assertEqual(slots, set(range(11*block//8)),
                                 'each subtile owns one consecutive lock slot')
        # Odd factors cannot arise from the supported grouping values.
        self.assertIn('or self.schedule_route_block_factor not in (2, 4)\n', self.k_after)
        self.assertIn('grouping values and are rejected at construction', self.m_after)

    def test_fc2_only_wiring_and_cache_key(self):
        text = self.k_after
        wiring = '            paired_m8_routes=self.fc2_paired_m8_routes,\n'
        self.assertEqual(text.count(wiring), 1)
        fc1_call = text.index('        self.fc1 = W4A16GemmKernel(\n')
        fc2_call = text.index('        self.fc2 = W4A16GemmKernel(\n')
        fused_end = text.index('        self.cta_threads = max(self.fc1.cta_threads')
        self.assertTrue(fc1_call < fc2_call < text.index(wiring) < fused_end)
        self.assertNotIn('paired_m8_routes', text[fc1_call:fc2_call])
        # The switch participates in the compiled GEMM identity and reaches
        # the fused key through fc2.__cache_key__, not as a direct field.
        gemm_key = text[text.index('class W4A16GemmKernel:'):text.index(
            '    @cute.jit\n    def _activation_smem_permuted_offset')]
        self.assertEqual(gemm_key.count('            self.paired_m8_routes,\n'), 1)
        self.assertIn('            self.whole_tile_tail_splitk,\n'
                      '            self.paired_m8_routes,\n', gemm_key)
        fused_flag = '        self.fc2_paired_m8_routes = (\n'
        self.assertEqual(text.count(fused_flag), 1)
        for term in ('_gb10_fc2_m8_pair_enabled()', 'and not self.small_m_splitk',
                     'and self.schedule_whole_tiles', 'and self.fc2_moe_block_size == 8',
                     'and self.fc2_schedule_route_block_factor in (2, 4)',
                     'and not self.direct_topk_routes'):
            self.assertIn(term, text[text.index(fused_flag):text.index(fused_flag)+420])

    def test_grouping_values_match_the_stock_schedule(self):
        helper = extract(self.m_after, '_gb10_fc2_schedule',
                         dict(os=os, torch=NS(cuda=NS(get_device_capability=lambda: (12, 1)))))
        for block in (16, 32, 64):
            for requested in (1, 2, 4):
                with patch.dict(os.environ, VLLM_GB10_EXL3_FC2_GROUP=str(requested)):
                    sub, group = helper(block)
                self.assertEqual(sub, 8)
                self.assertEqual(group, min(requested, block // 8))
                # The pair engages exactly on even grouped factors; an odd
                # factor cannot arise from the supported values and the
                # construction rejects it.
                if group >= 2:
                    self.assertIn(group, (2, 4))

    def test_cumulative_smoke_merges_overrides(self):
        source = (ROOT/'overlay/smoke_r22_v21.py').read_text(encoding='utf-8')
        seen = []
        fn = extract(source, 'main', dict(Path=Path, json=NS(dumps=lambda *a, **k: ''),
            print=lambda *a, **k: None, sys=NS(argv=[])))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for rel, content in ((p.KERNEL, self.k_after), (p.MIXED, self.m_after)):
                target = root/rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding='utf-8', newline='\n')
            modules = dict(b12x=NS(__file__=str(root/'__init__.py')),
                patch_r22_v21=p, patch_r22_v19=v19, patch_r22_v20=v20,
                smoke_r22_v18=NS(main=lambda **k: seen.append(k)))
            with patch.dict(sys.modules, modules):
                fn()
                self.assertEqual(seen, [dict(source_overrides={
                    **v19.OUTPUTS, **v20.OUTPUTS, **p.OUTPUTS})])
                # Final-state validation must still reject drift in either
                # superseded file before entering the inherited chain.
                for rel in (p.KERNEL, p.MIXED):
                    target = root/rel
                    original = target.read_bytes()
                    target.write_bytes(original+b'\n# unknown edit\n')
                    with self.assertRaisesRegex(RuntimeError, 'unexpected v21 source'):
                        fn()
                    target.write_bytes(original)
                self.assertEqual(len(seen), 1)

    def test_recipe_preserves_baseline_and_builder(self):
        before = (ROOT/'recipes/glm53-exl3-v20-instanttensor-r1-4x.yaml').read_text(encoding='utf-8')
        after = (ROOT/'recipes/glm53-exl3-v21-4x.yaml').read_text(encoding='utf-8')
        restored = after.replace('r22-v21-4x', 'r22-v20-instanttensor-r1-mtp3-4x')
        restored = restored.replace('spark-vllm-glm53-exl3:r22-dflash2-sm121-v21',
                                    'spark-vllm-glm53-exl3:r22-dflash2-sm121-v20-instanttensor-r1')
        restored = restored.replace('v20 MTP3 with archived R7 paired FC2 M8 weight reuse as a prefill option.',
                                    'v20 MTP3 with buffered, local, owned-copy InstantTensor loading (experimental).')
        kept, stripping = [], False
        for line in restored.split('\n'):
            if '# v21:' in line:
                stripping = True
                continue
            if stripping:
                if 'VLLM_GB10_EXL3_FC2_M8_PAIR:' in line:
                    stripping = False
                continue
            if 'v21_overlay:' in line:
                continue
            kept.append(line)
        restored = '\n'.join(kept)
        self.assertEqual(restored, before)
        self.assertIn('VLLM_GB10_EXL3_FC2_M8_PAIR: "1"', after)
        self.assertIn('VLLM_GB10_EXL3_FC2_GROUP: "4"', after)
        builder = (ROOT/'scripts/build-r22-v21-image.sh').read_text(encoding='utf-8')
        self.assertIn('r22-dflash2-sm121-v20-instanttensor-r1', builder)
        self.assertIn('Dockerfile.r22-dflash2-v21', builder)
        self.assertIn('smoke_r22_v21.py --gpu', builder)
        docker = (ROOT/'Dockerfile.r22-dflash2-v21').read_text(encoding='utf-8')
        self.assertIn('FROM spark-vllm-glm53-exl3:r22-dflash2-sm121-v20-instanttensor-r1', docker)
        self.assertIn('ENV VLLM_GB10_EXL3_FC2_M8_PAIR=0', docker)
        self.assertIn('LABEL local-inference.v21-overlay="glm53-r22-v21-1"', docker)
        for root_line in ('Path(\'/opt/b12x-r22/b12x\')', 'Path(b12x.__file__).parent'):
            self.assertIn(root_line, docker)


if __name__ == '__main__':
    unittest.main()