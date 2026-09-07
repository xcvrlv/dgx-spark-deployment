"""CPU regression gates for the pinned mixed-K schedule constructors."""
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
BASE = ROOT.parent/'tmp/exl3-v11/patched/b12x'
KERNEL = ROOT.parent/'tmp/exl3-v16/base/b12x/moe/_shared/kernels/w4a16/kernel.py'
spec = importlib.util.spec_from_file_location('patch_r22_v19', ROOT/'overlay/patch_r22_v19.py')
p = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = p
spec.loader.exec_module(p)


def extract(source, name, ns):
    node = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec('from __future__ import annotations\n'+ast.unparse(node), ns)
    return ns[name]


class Done(Exception):
    pass


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (BASE/p.MIXED).exists():
            raise unittest.SkipTest('pinned mixed-K fixture unavailable')
        cls.before = (BASE/p.MIXED).read_text(encoding='utf-8')
        cls.after = p.transform(cls.before)

    def helper(self, capability=(12, 1)):
        return extract(self.after, '_gb10_fc2_schedule',
                       dict(os=os, torch=NS(cuda=NS(get_device_capability=lambda: capability))))

    def constructors(self, source, block, group):
        seen = []
        def fuse(**kwargs):
            seen.append(kwargs)
            return NS(**kwargs)
        def finish(**kwargs):
            raise Done
        ns = dict(os=os, torch=NS(int32='int32', int64='int64',
            iinfo=lambda _: NS(max=2**31-1), cuda=NS(get_device_capability=lambda: (12, 1))),
            W4A16FusedMoeKernel=fuse, W4A16MixedTrellisKernel=finish,
            W4A16MixedTrellis3Kernel=finish, _MAX_TIER_EXPERTS=256)
        if '_gb10_fc2_schedule' in source:
            extract(source, '_gb10_fc2_schedule', ns)
        for tiers in (2, 3):
            fn = extract(source, 'compile_mixed_trellis'+('3' if tiers == 3 else ''), ns)
            with patch.dict(os.environ, VLLM_GB10_EXL3_FC2_GROUP=str(group)):
                with self.assertRaises(Done):
                    fn(size_m=129, hidden_size=6144, intermediate_size=512,
                        **{f'tier{i}_num_experts': 8 for i in range(tiers)},
                        top_k=8, max_m_blocks=128, sms=48, max_shared_mem=101376,
                        force_tile_config=(128, 128, 32, 512), moe_block_size=block)
        return seen

    def test_hash_preflight_idempotence_and_rejection(self):
        self.assertEqual(hashlib.sha256(self.before.encode()).hexdigest(), p.INPUTS[p.MIXED])
        self.assertEqual(hashlib.sha256(self.after.encode()).hexdigest(), p.OUTPUTS[p.MIXED])
        compile(self.after, 'mixed_trellis.py', 'exec')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root/p.MIXED
            target.parent.mkdir(parents=True)
            target.write_text(self.before, encoding='utf-8')
            with self.assertRaises(RuntimeError): p.patch(root, check=True)
            p.patch(root)
            p.patch(root)
            p.patch(root, check=True)
            target.write_text(self.before+'\n# unknown edit\n', encoding='utf-8')
            previous = target.read_bytes()
            with self.assertRaises(RuntimeError): p.patch(root)
            self.assertEqual(target.read_bytes(), previous)

    def test_m16_uses_existing_register_specializations_in_both_compilers(self):
        if not KERNEL.exists(): self.skipTest('kernel fixture unavailable')
        source = KERNEL.read_text(encoding='utf-8')
        table = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == '_W4A16_REGS_SM121' for t in n.targets))
        ns = {}
        exec(ast.unparse(table), ns)
        regs = extract(source, '_w4a16_num_regs', ns)
        def phase_regs(kwargs, phase):
            block = kwargs['moe_block_size'] if phase == 'fc1' else kwargs['fc2_moe_block_size']
            n, k = kwargs[phase+'_tile_n'], kwargs[phase+'_tile_k']
            return regs(cta_threads=n*k//64, cta_m_blocks=(block+15)//16,
                        cta_n_blocks=n//16, cta_k_blocks=k//16, uses_m_block_8=block == 8)
        for kwargs in self.constructors(self.before, 16, 2):
            with self.assertRaisesRegex(ValueError, 'missing W4A16 register count'):
                phase_regs(kwargs, 'fc2')
        for kwargs in self.constructors(self.after, 16, 2):
            self.assertEqual(phase_regs(kwargs, 'fc1'), 158)
            self.assertEqual(phase_regs(kwargs, 'fc2'), 118)
            self.assertEqual(kwargs['fc2_schedule_route_block_factor'], 2)

    def test_existing_default_and_decode_constructors_unchanged(self):
        for block in (8, 32, 64):
            self.assertEqual(self.constructors(self.before, block, 2),
                             self.constructors(self.after, block, 2))
        self.assertEqual(self.constructors(self.before, 8, 2),
                         self.constructors(self.after, 8, 4))

    def test_grouping_cannot_cross_packed_expert_boundaries(self):
        helper = self.helper()
        for block in (16, 32, 64):
            for requested in (1, 2, 4):
                with patch.dict(os.environ, VLLM_GB10_EXL3_FC2_GROUP=str(requested)):
                    sub, group = helper(block)
                self.assertEqual(sub, 8)
                self.assertEqual(block % (sub*group), 0)
                visited = []
                for job in range(11*block//(sub*group)):
                    expert_block = job//(block//(sub*group))
                    for subtile in range(group):
                        for row in range((job*group+subtile)*sub, (job*group+subtile+1)*sub):
                            self.assertEqual(row//block, expert_block)
                            visited.append(row)
                self.assertEqual(visited, list(range(11*block)))

    def test_architecture_gating_and_invalid_settings(self):
        with patch.dict(os.environ, VLLM_GB10_EXL3_FC2_GROUP='4'):
            self.assertEqual(self.helper()(16), (8, 2))
            self.assertEqual(self.helper()(32), (8, 4))
            self.assertEqual(self.helper((12, 0))(32), (8, 2))
        for invalid in ('0', '-1', '3', '8', 'typo'):
            with patch.dict(os.environ, VLLM_GB10_EXL3_FC2_GROUP=invalid):
                with self.assertRaises(ValueError): self.helper()(32)
                self.assertEqual(self.helper()(8), (8, 1))

    def test_cumulative_smoke_passes_latest_hash_override(self):
        source = (ROOT/'overlay/smoke_r22_v18.py').read_text(encoding='utf-8')
        seen = []
        modules = dict(vllm=NS(__file__='fake/vllm/__init__.py'),
            patch_r22_v18=NS(patch=lambda *a, **k: None, VERSION='v18', OUTPUTS={'indexer':'v18'}),
            smoke_r22_v17=NS(main=lambda **k: seen.append(k)))
        fn = extract(source, 'main', dict(Path=Path, sys=NS(argv=[]),
                                         json=NS(dumps=lambda *a, **k: ''), print=lambda *a, **k: None))
        with patch.dict(sys.modules, modules): fn(source_overrides=p.OUTPUTS)
        self.assertEqual(seen, [dict(source_overrides={'indexer':'v18', **p.OUTPUTS})])

    def test_recipe_preserves_baseline_and_builder_is_cumulative(self):
        before = (ROOT/'recipes/glm53-exl3-v18-4x.yaml').read_text(encoding='utf-8')
        after = (ROOT/'recipes/glm53-exl3-v19-4x.yaml').read_text(encoding='utf-8')
        restored = after.replace('r22-v19-mtp', 'r22-v18-mtp').replace('sm121-v19', 'sm121-v18')
        restored = '\n'.join(line for line in restored.split('\n') if not any(k in line for k in
            ('v19_overlay:', 'VLLM_GB10_EXL3_FC2_GROUP:')))
        self.assertEqual(restored, before)
        builder = (ROOT/'scripts/build-r22-v19-image.sh').read_text(encoding='utf-8')
        self.assertIn('for version in 11 12 13 14 15 16 17 18;', builder)
        self.assertIn('GLM53_R22_SMOKE_SCRIPT=smoke_r22_v19.py', builder)


if __name__ == '__main__': unittest.main()
