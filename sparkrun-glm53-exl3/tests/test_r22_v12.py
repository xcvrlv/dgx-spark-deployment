"""CPU semantics and exact-source tests; GPU compilation is a build gate."""
import ast
import hashlib
import importlib.util
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("v12", ROOT / "overlay/patch_r22_v12.py")
patcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patcher)
HELPER = (ROOT / "overlay/r22_v12_ckv.py").read_text(encoding="utf-8")
BASE = Path(os.environ.get("GLM53_V12_BASELINE", ROOT.parent / "tmp/r22-prefill-comparison/performance-source/vllm"))


class Vec(np.ndarray):
    def to(self, dtype):
        return self.astype(dtype)


def vec(value):
    return np.asarray(value).view(Vec)


class Ptr:
    def __init__(self, data, offset=0):
        self.data, self.offset = data.reshape(-1), offset

    def __add__(self, offset):
        return Ptr(self.data, self.offset + offset)


class TL:
    int32 = np.int32
    row = 0
    @classmethod
    def program_id(cls, axis):
        return cls.row
    arange = staticmethod(lambda start, end: vec(np.arange(start, end)))
    cumsum = staticmethod(lambda value: vec(np.cumsum(value, dtype=np.int32)))
    sum = staticmethod(lambda value: vec(np.sum(value, dtype=np.int32)))
    minimum = staticmethod(np.minimum)

    @staticmethod
    def load(ptr, mask=True, other=0):
        offset, mask = np.broadcast_arrays(ptr.offset, mask)
        result = np.full(offset.shape, other, dtype=ptr.data.dtype)
        result[mask] = ptr.data[offset[mask]]
        return vec(result)

    @staticmethod
    def store(ptr, value, mask=True):
        offset, value, mask = np.broadcast_arrays(ptr.offset, value, mask)
        ptr.data[offset[mask]] = value[mask]


def extract(name, namespace):
    fn = next(n for n in ast.parse(HELPER).body if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.decorator_list = []
    exec("from __future__ import annotations\n" + ast.unparse(fn), namespace)
    return namespace[name]


class V12Tests(unittest.TestCase):
    def test_kernel_mapping_causal_cap_and_tail(self):
        kernel = extract("_v12_ckv_metadata_kernel", {"tl": TL})
        rng = np.random.default_rng(5312)
        for world in (1, 2, 4):
            for interleave in (1, 16):
                rows, width = 9, 32
                req = np.array([0] * 4 + [1] * 5, dtype=np.int32)
                seq = np.array([7, 33], dtype=np.int32)
                qsl = np.array([0, 4, 9], dtype=np.int32)
                lens = np.array([[sum((t // interleave) % world == rank for t in range(n))
                                  for n in seq] for rank in range(world)], dtype=np.int32)
                starts = np.zeros_like(lens)
                starts[:, 1] = lens[:, 0]
                padded = int(lens.sum(1).max()) + 16
                ids = rng.integers(-1, 34, (rows, width), dtype=np.int32)
                ids[:, ::7] = -1
                ids[:, 1] = 10000
                ids[0] = -1
                out = np.full((rows, width), -999, dtype=np.int32)
                counts, causal = np.zeros(rows, dtype=np.int32), np.zeros(rows, dtype=np.int32)
                for row in range(rows):
                    TL.row = row
                    kernel(*[Ptr(a) for a in (req, ids, starts, lens, seq, qsl, out, counts, causal)],
                           width, 1, 2, 1, 2, 1, width, 1, padded,
                           WORLD=world, INTERLEAVE=interleave, WIDTH=width, BLOCK=width)
                    r = req[row]
                    expected_causal = seq[r] - qsl[r + 1] + row + 1
                    mapped = []
                    for token in ids[row]:
                        if token < 0:
                            continue
                        owner = (token // interleave) % world
                        local = (token // (world * interleave)) * interleave + token % interleave
                        if local < lens[owner, r]:
                            mapped.append(owner * padded + starts[owner, r] + local)
                    mapped = mapped[:expected_causal]
                    np.testing.assert_array_equal(out[row], mapped + [-1] * (width - len(mapped)))
                    self.assertEqual(counts[row], len(mapped))
                    self.assertEqual(causal[row], expected_causal)

    def test_query_borrow_rejects_alias_shape_dtype_and_alignment(self):
        borrow = extract("_v12_can_borrow_query", {"torch": NS(bfloat16="bf16")})
        def tensor(address, shape=(4, 64, 576), dtype="bf16", contiguous=True):
            return NS(dtype=dtype, shape=shape, is_contiguous=lambda: contiguous,
                      data_ptr=lambda: address, numel=lambda: int(np.prod(shape)), element_size=lambda: 2)
        q, scratch = tensor(4096), tensor(1000000, (1024,))
        self.assertTrue(borrow(q, scratch, True, 4, 64, 576))
        self.assertFalse(borrow(q, q, True, 4, 64, 576))
        self.assertFalse(borrow(q, tensor(8192), True, 4, 64, 576))
        self.assertFalse(borrow(q, scratch, False, 4, 64, 576))
        for bad in (tensor(4098), tensor(4096, dtype="fp16"), tensor(4096, contiguous=False), tensor(4096, (1, 64, 576))):
            self.assertFalse(borrow(bad, scratch, True, 4, 64, 576))

    def test_exact_v11_input_and_idempotent_patch(self):
        if not (BASE / patcher.TARGET).is_file():
            self.skipTest("set GLM53_V12_BASELINE to the v11 vllm package")
        source = (BASE / patcher.TARGET).read_text(encoding="utf-8")
        self.assertEqual(hashlib.sha256(source.encode()).hexdigest(), patcher.INPUT_HASH)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / patcher.TARGET
            path.parent.mkdir(parents=True)
            path.write_text(source, encoding="utf-8", newline="\n")
            patcher.patch(tmp)
            patcher.patch(tmp)
            patcher.patch(tmp, check=True)
            patched = path.read_bytes()
            self.assertEqual(hashlib.sha256(patched).hexdigest(), patcher.OUTPUT_HASH)
            path.write_bytes(patched + b"\n# drift\n")
            before = path.read_bytes()
            with self.assertRaises(RuntimeError):
                patcher.patch(tmp)
            self.assertEqual(before, path.read_bytes())

    def test_build_gates_and_memory_setting(self):
        recipe = (ROOT / "recipes/glm53-exl3-v12-4x.yaml").read_text(encoding="utf-8")
        self.assertIn("gpu_memory_utilization: 0.87", recipe)
        self.assertIn("r22-dflash2-sm121-v12", recipe)
        builder = (ROOT / "scripts/build-r22-dflash2-image.sh").read_text(encoding="utf-8")
        self.assertEqual(builder.count("/opt/compose/smoke_r22_v12.py --gpu"), 2)
        self.assertIn("else", builder)
        wrapper = (ROOT / "scripts/build-r22-v12-image.sh").read_text(encoding="utf-8")
        self.assertIn("GLM53_R22_V11_SMOKE=1 GLM53_R22_V12_SMOKE=1", wrapper)


if __name__ == "__main__":
    unittest.main()
