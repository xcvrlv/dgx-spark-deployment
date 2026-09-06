"""CPU semantic checks; the build script runs the real SM121 kernels."""
import ast
import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace as NS
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = Path(os.getenv("GLM53_V14_BASELINE", ROOT.parent / "tmp/exl3-v14/base"))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


patcher = load("v14patch", ROOT / "overlay/patch_r22_v14.py")
v12 = load("v12test", ROOT / "tests/test_r22_v12.py")
HELPER = (ROOT / "overlay/gb10_argmax.py").read_text(encoding="utf-8")


class TL(v12.TL):
    int64, float32 = np.int64, np.float32
    ids = (0, 0)
    program_id = classmethod(lambda cls, axis: cls.ids[axis])
    min = staticmethod(lambda a, axis: v12.vec(np.min(a, axis=axis)))
    max = staticmethod(lambda a, axis: v12.vec(np.max(a, axis=axis)))
    where = staticmethod(lambda cond, a, b: v12.vec(np.where(cond, a, b)))


def functions(source, names, namespace):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            exec("from __future__ import annotations\n" + ast.unparse(node), namespace)
    return namespace


class V14Tests(unittest.TestCase):
    def test_actual_split_kernels_padding_ties_nan_and_tp_order(self):
        ns = functions(HELPER, ("_winner", "_partial", "_local_finish", "_global_finish"), {"tl": TL})
        rng = np.random.default_rng(5314)
        for valid in (0, 1, 513, 1023, 1025):
            for mode in ("random", "ties", "nan", "negative_inf"):
                rows, width, block, tp = 3, 1025, 512, 4
                chunks = (width + block - 1) // block
                pairs, references = [], []
                for rank in range(tp):
                    x = rng.normal(size=(rows, width)).astype(np.float32)
                    if mode == "ties":
                        x[:, 0] = x[:, 512] = 100
                    elif mode == "nan":
                        x[:, 0] = x[:, 512] = np.nan
                    elif mode == "negative_inf":
                        x.fill(-np.inf)
                    # An all-padding last rank should never beat a real token.
                    count = valid if rank == 3 else width
                    partial = np.empty((rows, chunks, 2), np.float32)
                    pair = np.empty((rows, 4), np.float32)
                    for row in range(rows):
                        for chunk in range(chunks):
                            TL.ids = row, chunk
                            ns['_partial'](v12.Ptr(x), v12.Ptr(partial), width, count, chunks, block)
                        ns['_local_finish'](v12.Ptr(partial), v12.Ptr(pair), chunks, rank * width, 4)
                    masked = x.copy()
                    masked[:, count:] = -np.inf
                    ix = masked.argmax(-1)
                    expected = np.stack((masked[np.arange(rows), ix], ix + rank * width), -1)
                    np.testing.assert_equal(pair[:, :2], expected)
                    np.testing.assert_array_equal(pair[:, 2:], 0)
                    pairs.append(pair)
                    references.append(masked)
                gathered = np.concatenate(pairs, -1)
                out = np.empty(rows, np.int64)
                for row in range(rows):
                    TL.ids = row, 0
                    ns['_global_finish'](v12.Ptr(gathered), v12.Ptr(out), tp, 4)
                np.testing.assert_array_equal(out, np.concatenate(references, -1).argmax(-1))

    def test_shared_rotation_actual_route_addressing(self):
        if not BASE.is_dir():
            self.skipTest("set GLM53_V14_BASELINE to the exact v13 source fixture")
        source = patcher.kernel((BASE / "b12x" / patcher.KERNEL).read_text(encoding="utf-8"))
        cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'W4A16FusedMoeKernel')
        # Execute the actual route loops on CPU with an identity butterfly.
        # GPU tests separately verify FP16 rounding and warp shuffles exactly.
        class NoCast(ast.NodeTransformer):
            def visit_Call(self, node):
                node = self.generic_visit(node)
                return node.func.value if isinstance(node.func, ast.Attribute) and node.func.attr == 'to' else node
        ns = {"Int32": int, "cutlass": NS(Float16=float, Float32=float,
              const_expr=bool, range_constexpr=range)}
        for name in ('_run_input_rotation_shared', '_run_input_rotation'):
            node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
            node.decorator_list = []
            node = NoCast().visit(node)
            exec('from __future__ import annotations\n' + ast.unparse(node), ns)
        m, h, k = 5, 256, 3
        driver = NS(hidden_size=h, top_k=k, cta_threads=128, broadcast_suh=True,
                    direct_topk_routes=False, moe_block_size=1,
                    _had128_quad=lambda a, b, c, d, lane: (a, b, c, d))
        driver._run_input_rotation_shared = lambda *args: ns['_run_input_rotation_shared'](driver, *args)
        rng = np.random.default_rng(14)
        x, sg, su = rng.normal(size=m*h), rng.normal(size=h), rng.normal(size=h)
        routes = rng.permutation(m*k)
        routes[::5] = m*k
        experts = np.arange(m*k) % 2
        experts[::4] = -1
        outputs = []
        for enabled in (False, True):
            driver.gb10_shared_input = enabled
            g, u = np.full(m*k*h, np.nan), np.full(m*k*h, np.nan)
            for cta in range(2):
                for tid in range(128):
                    ns['_run_input_rotation'](driver, x, g, u, sg, su, routes, experts,
                        np.array([m*k]), np.arange(2), 2, 2, tid, cta, 2, m)
            outputs.append((g.reshape(-1,h), u.reshape(-1,h)))
        live = routes[(routes < m*k) & (experts >= 0)]
        for old, new in zip(*outputs):
            np.testing.assert_equal(old[live], new[live])
        # The shared path writes every route, even those the packed FC1 skips.
        np.testing.assert_equal(outputs[1][0], np.repeat((x.reshape(m,h)*sg), k, axis=0))

    def test_pinned_patch_preflight_and_idempotence(self):
        if not BASE.is_dir():
            self.skipTest("set GLM53_V14_BASELINE to the exact v13 source fixture")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'source'
            shutil.copytree(BASE, root)
            b, v = root / 'b12x', root / 'vllm'
            patcher.patch(b, v)
            patcher.patch(b, v)
            patcher.patch(b, v, check=True)
            (v / patcher.LOGITS).write_text('# unexpected source\n', encoding='utf-8')
            before = {p:p.read_bytes() for p in root.rglob('*.py')}
            with self.assertRaises(RuntimeError):
                patcher.patch(b, v)
            self.assertEqual(before, {p:p.read_bytes() for p in root.rglob('*.py')})

    def test_recipe_and_smoke_integration(self):
        source = (ROOT / 'recipes/glm53-exl3-v14-4x.yaml').read_text(encoding='utf-8')
        self.assertIn('gpu_memory_utilization: 0.87', source)
        self.assertIn('use_local_argmax_reduction\\\":true', source)
        self.assertIn('VLLM_GB10_SHARED_INPUT_ROTATION: "1"', source)
        builder = (ROOT / 'scripts/build-r22-v14-image.sh').read_text(encoding='utf-8')
        self.assertIn('GLM53_R22_V14_SMOKE=1', builder)
        common = (ROOT / 'scripts/build-r22-dflash2-image.sh').read_text(encoding='utf-8')
        self.assertEqual(common.count('/opt/compose/smoke_r22_v14.py --gpu'), 2)
        smoke = (ROOT / 'overlay/smoke_r22_v14.py').read_text(encoding='utf-8')
        self.assertIn('_load_exl3_ext()', smoke)
        self.assertIn('inherited_v13=v13_gpu()', smoke)
        self.assertIn('result.update(mixed_gpu())', smoke)
        self.assertEqual(hashlib.sha256(HELPER.encode()).hexdigest(), patcher.OUTPUTS[patcher.HELPER])


if __name__ == '__main__':
    unittest.main()
