"""CPU regression gates for the GB10 FC1 whole-tile tail split."""
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
KERNEL_FIXTURE = ROOT.parent/'tmp/exl3-v16/base/b12x/moe/_shared/kernels/w4a16/kernel.py'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


p = load('patch_r22_v20', ROOT/'overlay/patch_r22_v20.py')
v16 = load('patch_r22_v16_reloaded', ROOT/'overlay/patch_r22_v16.py')


def extract(source, name, ns):
    node = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec('from __future__ import annotations\n'+ast.unparse(node), ns)
    return ns[name]


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not KERNEL_FIXTURE.exists():
            raise unittest.SkipTest('pinned kernel fixture unavailable')
        source = KERNEL_FIXTURE.read_text(encoding='utf-8')
        assert hashlib.sha256(source.encode()).hexdigest() == v16.INPUTS[v16.KERNEL]
        # The v16 sigmoid merge fix is the last kernel.py overlay before v20.
        cls.before = v16.kernel(source)
        assert hashlib.sha256(cls.before.encode()).hexdigest() == v16.OUTPUTS[v16.KERNEL]
        cls.after = p.transform(cls.before)

    def test_hash_preflight_idempotence_and_rejection(self):
        self.assertEqual(hashlib.sha256(self.before.encode()).hexdigest(), p.INPUTS[p.KERNEL])
        self.assertEqual(p.INPUTS[p.KERNEL], v16.OUTPUTS[v16.KERNEL])
        self.assertEqual(hashlib.sha256(self.after.encode()).hexdigest(), p.OUTPUTS[p.KERNEL])
        compile(self.after, 'kernel.py', 'exec')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root/p.KERNEL
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

    def test_helper_architecture_and_value_gating(self):
        def helper(capability=(12, 1)):
            return extract(self.after, '_gb10_fc1_tail_splitk_enabled',
                           dict(os=os, torch=NS(cuda=NS(get_device_capability=lambda: capability))))
        gb10, other = helper(), helper((12, 0))
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(gb10())
        for value, expected in (('0', False), ('1', True)):
            with patch.dict(os.environ, VLLM_GB10_EXL3_FC1_TAILSPLIT=value):
                self.assertEqual(gb10(), expected)
                self.assertFalse(other(), 'non-SM121 must ignore the switch')
        for invalid in ('2', '-1', 'true', 'typo'):
            with patch.dict(os.environ, VLLM_GB10_EXL3_FC1_TAILSPLIT=invalid):
                with self.assertRaises(ValueError): gb10()

    def test_gemm_validation_rejects_ineligible_geometry(self):
        text = self.after
        start = '        # GB10 FC1 whole-tile tail split (v20). The switch may only\n'
        end = '                "whole-tile schedule"\n            )\n'
        self.assertEqual(text.count(start), 1)
        block = text[text.index(start):text.index(end)+len(end)]
        dedented = '\n'.join(line[8:] if line.startswith('        ') else line
                             for line in block.splitlines())

        def construct(flag, block_size, whole=True, direct=False, dense=False):
            self_ns = NS(schedule_whole_tiles=whole, direct_topk_routes=direct,
                         dense_route_fast_path=dense)
            ns = dict(self=self_ns, whole_tile_tail_splitk=flag,
                      moe_block_size=block_size, int=int, ValueError=ValueError)
            exec(dedented, ns)
            return self_ns.whole_tile_tail_splitk
        self.assertIs(construct(True, 8), True)
        self.assertIs(construct(False, 32), False)
        for args in ((True, 32), (True, 16), (True, 8, False), (True, 8, True, False, True),
                     (True, 8, True, True)):
            with self.assertRaises(ValueError):
                construct(*args)

    def test_scheduler_branch_is_whole_tile_scoped(self):
        arm = '        if cutlass.const_expr(self.schedule_whole_tiles):\n'
        outside = '        if cutlass.const_expr(not self.schedule_whole_tiles):\n'
        branch = '            if cutlass.const_expr(self.whole_tile_tail_splitk):\n'
        text = self.after
        self.assertEqual(text.count(branch), 1)
        self.assertLess(text.index(arm), text.index(branch))
        self.assertLess(text.index(branch), text.index(outside))
        # The stock whole-tile arm still computes the ceiling first, and the
        # split only overrides it when a complete wave AND a ragged remainder
        # both exist at runtime.
        self.assertIn(
            '                if full_waves > Int32(0) and ragged_tiles > Int32(0):\n',
            text)

    def test_tail_partition_model_covers_cells_once_and_bounds_locks(self):
        """Python model of the v20 runtime conditional plus the stock tail
        arithmetic it activates (iters/slices/slots mirror
        _run_persistent_gemm). Coverage and lock bounds are the invariants
        the striped remainder must keep for every M8 decode shape."""

        def plan(tiles, grid, k_tiles, flag):
            full, tail = (tiles + grid - 1)//grid, 0
            if flag:
                waves, ragged = divmod(tiles, grid)
                if waves > 0 and ragged > 0:
                    full, tail = waves, ragged
            return full, tail

        for flag in (False, True):
            for blocks in range(1, 41):
                tiles = 8*blocks
                grid = k_tiles = 48
                full, tail = plan(tiles, grid, k_tiles, flag)
                if not flag or tiles < grid or tiles % grid == 0:
                    self.assertEqual((full, tail), ((tiles + grid - 1)//grid, 0))
                    continue
                self.assertEqual(full*grid + tail, tiles)
                self.assertGreater(tail, 0)
                self.assertEqual(tail % 8, 0, 'FC1 remainder is n_tiles-aligned')
                iters = (k_tiles*tail + grid - 1)//grid
                covered = [0]*(k_tiles*tail)
                slots = set()
                for cta in range(grid):
                    start = iters*cta
                    if start >= k_tiles*tail:
                        continue
                    work, k_off = divmod(start, k_tiles)
                    lock = start//k_tiles - 1
                    while work < tail:
                        count = min(iters*(cta + 1) - (k_tiles*work + k_off), k_tiles - k_off)
                        if count <= 0:
                            break
                        for cell in range(k_tiles*work + k_off, k_tiles*work + k_off + count):
                            covered[cell] += 1
                        lock += 1
                        slots.add(lock)
                        k_off, work = 0, work + 1
                self.assertEqual(covered, [1]*(k_tiles*tail),
                                 f'ragged remainder must be covered exactly once')
                self.assertEqual(slots, set(range(tail)),
                                 'each remainder tile owns one lock slot')
        # Every decode remainder uses at most 40 of the 192 workspace locks.
        for blocks in (7, 11, 13, 17, 23):
            _, tail = plan(8*blocks, 48, 48, True)
            self.assertLess(tail, 48)
            self.assertLessEqual(tail, 40)

    def test_fc1_only_wiring_and_cache_key(self):
        text = self.after
        wiring = '            whole_tile_tail_splitk=self.fc1_tail_splitk,\n'
        self.assertEqual(text.count(wiring), 1)
        fc1_call = text.index('        self.fc1 = W4A16GemmKernel(\n')
        fc2_call = text.index('        self.fc2 = W4A16GemmKernel(\n')
        fused_end = text.index('        self.cta_threads = max(self.fc1.cta_threads')
        self.assertTrue(fc1_call < text.index(wiring) < fc2_call)
        self.assertNotIn('whole_tile_tail_splitk', text[fc2_call:fused_end])
        # The switch participates in the compiled GEMM identity and reaches
        # the fused key through fc1.__cache_key__, not as a direct field.
        gemm_key = text[text.index('class W4A16GemmKernel:'):text.index('    @cute.jit\n    def _activation_smem_permuted_offset')]
        self.assertEqual(gemm_key.count('            self.whole_tile_tail_splitk,\n'), 1)
        self.assertIn('            self.small_m_splitk,\n            self.whole_tile_tail_splitk,\n', gemm_key)
        fused_flag = '        self.fc1_tail_splitk = (\n'
        self.assertEqual(text.count(fused_flag), 1)
        for term in ('_gb10_fc1_tail_splitk_enabled()', 'and not self.small_m_splitk',
                     'and self.schedule_whole_tiles', 'and self.moe_block_size == 8',
                     'and not self.direct_topk_routes'):
            self.assertIn(term, text[text.index(fused_flag):text.index(fused_flag)+400])

    def test_cumulative_smoke_merges_v19_and_v20_overrides(self):
        source = (ROOT/'overlay/smoke_r22_v20.py').read_text(encoding='utf-8')
        seen, checked = [], []
        v19_outputs = {'moe/_shared/kernels/w4a16/mixed_trellis.py': 'v19'}
        modules = dict(b12x=NS(__file__='fake/b12x/__init__.py'),
            patch_r22_v20=NS(patch=lambda *a, **k: None, VERSION='v20', OUTPUTS={'k': 'v20k'}),
            patch_r22_v19=NS(patch=lambda *a, **k: checked.append(a), OUTPUTS=v19_outputs),
            smoke_r22_v18=NS(main=lambda **k: seen.append(k)))
        fn = extract(source, 'main', dict(Path=Path, importlib=NS(), json=NS(dumps=lambda *a, **k: ''),
            print=lambda *a, **k: None, sys=NS(argv=[])))
        with patch.dict(sys.modules, modules):
            fn()
        self.assertEqual(seen, [dict(source_overrides={**v19_outputs, 'k': 'v20k'})])
        self.assertEqual(len(checked), 1, 'v19 mixed_trellis state must be verified directly')

    def test_recipe_preserves_baseline_and_builder_is_cumulative(self):
        before = (ROOT/'recipes/glm53-exl3-v19-4x.yaml').read_text(encoding='utf-8')
        after = (ROOT/'recipes/glm53-exl3-v20-4x.yaml').read_text(encoding='utf-8')
        restored = after.replace('r22-v20-mtp', 'r22-v19-mtp').replace('sm121-v20', 'sm121-v19')
        restored = '\n'.join(line for line in restored.split('\n') if not any(k in line for k in
            ('v20_overlay:', 'VLLM_GB10_EXL3_FC1_TAILSPLIT:', '# v20:')))
        self.assertEqual(restored, before)
        self.assertIn('VLLM_GB10_EXL3_FC1_TAILSPLIT: "1"', after)
        builder = (ROOT/'scripts/build-r22-v20-image.sh').read_text(encoding='utf-8')
        self.assertIn('for version in 11 12 13 14 15 16 17 18 19;', builder)
        self.assertIn('GLM53_R22_SMOKE_SCRIPT=smoke_r22_v20.py', builder)
        self.assertIn('Dockerfile.r22-dflash2-v20', builder)
        docker = (ROOT/'Dockerfile.r22-dflash2-v20').read_text(encoding='utf-8')
        self.assertIn('FROM spark-vllm-glm53-exl3:r22-dflash2-sm121-v19', docker)
        self.assertIn('ENV VLLM_GB10_EXL3_FC1_TAILSPLIT=0', docker)
        self.assertIn('LABEL local-inference.v20-overlay="glm53-r22-v20-1"', docker)


if __name__ == '__main__':
    unittest.main()
