"""CPU layout/arithmetic checks plus exact v12 source and integration checks."""
import ast
import importlib.util
import os
from pathlib import Path
import shutil
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

patcher = load("v13patch", ROOT / "overlay/patch_r22_v13.py")
v12test = load("v12test", ROOT / "tests/test_r22_v12.py")
SOURCE = (ROOT / "overlay/gb10_rotations.py").read_text(encoding="utf-8")
BASE = Path(os.environ.get("GLM53_V13_BASELINE", ROOT.parent / "tmp/exl3-v13/base"))


class TL(v12test.TL):
    float16 = np.float16
    float32 = np.float32
    broadcast_to = staticmethod(lambda x, shape: v12test.vec(np.broadcast_to(x, shape)))
    gather = staticmethod(lambda x, idx, axis: v12test.vec(np.take_along_axis(x, idx, axis)))
    where = staticmethod(lambda cond, a, b: v12test.vec(np.where(cond, a, b)))
    static_range = staticmethod(range)
    reshape = staticmethod(lambda x, shape: v12test.vec(np.reshape(x, shape)))
    split = staticmethod(lambda x: (v12test.vec(x[..., 0]), v12test.vec(x[..., 1])))
    join = staticmethod(lambda a, b: v12test.vec(np.stack((a, b), axis=-1)))


def extract(name, namespace):
    fn = next(n for n in ast.parse(SOURCE).body if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.decorator_list = []
    exec("from __future__ import annotations\n" + ast.unparse(fn), namespace)
    return namespace[name]


class V13Tests(unittest.TestCase):
    def test_slab_policy_preserves_48_sm_coverage(self):
        select = extract("launch_slabs", {})
        for total in (1, 4, 48, 191):
            self.assertEqual(select(total, 48), 1)
        for total in (192, 768, 196608):
            self.assertEqual(select(total, 48), 4)

    def test_actual_rotation_kernel_matches_hadamard_matrix(self):
        run = extract("_rotate", {"tl": TL})
        h = np.ones((1, 1), dtype=np.float32)
        for _ in range(7):
            h = np.block([[h, h], [h, -h]])
        rng = np.random.default_rng(5313)
        # Integer/power-of-two operands make the matrix reference's sums exact;
        # the GPU gate separately tests arbitrary floats against native EXL3.
        for rows, cols in ((1, 128), (3, 384), (4, 6144)):
            x = (rng.integers(-16, 17, (rows, cols)) / 16).astype(np.float16)
            scales = rng.choice([-1, 1], cols).astype(np.float16)
            for pre in (True, False):
                value = (x * scales).astype(np.float16) if pre else x
                expected = ((value.reshape(-1, 128).astype(np.float32) @ h)
                            * np.float32(0.088388347648)).astype(np.float16).reshape(rows, cols)
                if not pre:
                    expected = (expected * scales).astype(np.float16)
                for slabs in (1, 4):
                    out = np.full_like(x, np.nan)
                    total = rows * (cols // 128)
                    for block in range((total + slabs - 1) // slabs):
                        TL.row = block
                        run(v12test.Ptr(x), v12test.Ptr(scales), v12test.Ptr(out), rows, cols, pre, slabs)
                    np.testing.assert_array_equal(out, expected)

    def test_patch_preflight_idempotence_and_bias_fallback(self):
        if not BASE.is_dir():
            self.skipTest("set GLM53_V13_BASELINE to the exact v12 source fixture")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            shutil.copytree(BASE, root)
            b, v = root / "b12x", root / "vllm"
            patcher.patch(b, v)
            patcher.patch(b, v)
            patcher.patch(b, v, check=True)
            source = (v / patcher.EXL3).read_text(encoding="utf-8")
            self.assertIn("if bias is None and enabled(x):", source)
            self.assertIn("torch.float16 if enabled(x) else x.dtype", source)
            source = (b / patcher.KERNEL).read_text(encoding="utf-8")
            self.assertIn("hadamard_128 is None and trellis_bits == 6", source)
            self.assertIn("trellis_codebook == 'mcg'", source)
            helper = b / patcher.HELPER
            helper.write_text("# unknown source\n", encoding="utf-8")
            before = {p: p.read_bytes() for p in root.rglob("*.py")}
            with self.assertRaises(RuntimeError):
                patcher.patch(b, v)
            self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_build_preserves_inherited_gpu_checks_and_087(self):
        recipe = (ROOT / "recipes/glm53-exl3-v13-4x.yaml").read_text(encoding="utf-8")
        self.assertIn("gpu_memory_utilization: 0.87", recipe)
        self.assertIn('VLLM_GB10_DENSE_ROTATIONS: "1"', recipe)
        builder = (ROOT / "scripts/build-r22-dflash2-image.sh").read_text(encoding="utf-8")
        self.assertEqual(builder.count("/opt/compose/smoke_r22_v13.py --gpu"), 2)
        smoke = (ROOT / "overlay/smoke_r22_v13.py").read_text(encoding="utf-8")
        self.assertIn("inherited = check_gpu()", smoke)
        self.assertIn("graph.replay()", smoke)


if __name__ == "__main__":
    unittest.main()
