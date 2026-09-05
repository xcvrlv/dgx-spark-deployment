#!/usr/bin/env python3
"""CPU behavioral checks; set GLM53_R22_PERF_BASELINE to a v9 vllm package.

The optional exact-source suite runs the transformed runtime functions with
NumPy-backed tensors. Image builds additionally run real CUDA attention checks.
"""

from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace as NS
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("performance", ROOT / "overlay/patch_r22_performance.py")
perf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(perf)
BASELINE = os.environ.get("GLM53_R22_PERF_BASELINE")


def extract(source, name, cls=None, namespace=None):
    nodes = ast.parse(source).body
    if cls:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == cls).body
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    namespace = dict(namespace or {})
    exec(compile(ast.fix_missing_locations(module), "<pinned-runtime>", "exec"), namespace)
    return namespace[name]


class TestBuildContract(unittest.TestCase):
    def test_late_overlay_and_gpu_checks(self):
        docker = (ROOT / "Dockerfile.r22-dflash2").read_text()
        self.assertGreater(docker.index("COPY overlay/patch_r22_performance.py"), docker.index("/opt/vllm-r22\n"))
        self.assertIn("patch(package, check=True)", docker)
        smoke = (ROOT / "overlay/smoke_r22_performance.py").read_text()
        self.assertIn("patch(Path(vllm.__file__).parent, check=True)", smoke)
        self.assertIn("result.update(check_ckv_attention())", smoke)
        builder = (ROOT / "scripts/build-r22-dflash2-image.sh").read_text()
        self.assertEqual(builder.count("/opt/compose/smoke_r22_performance.py --gpu"), 2)
        self.assertEqual(set(perf.TRANSFORMS), set(perf.INPUT_HASHES))
        self.assertEqual(set(perf.TRANSFORMS), set(perf.OUTPUT_HASHES))

    def test_recipe_bounds_and_rollback(self):
        recipe = (ROOT / "recipes/glm53-exl3-dflash2-4x.yaml").read_text()
        self.assertIn('VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS: "131072"', recipe)
        self.assertIn('VLLM_ROCE_DCP_RS_MAX_BYTES: "262144"', recipe)
        self.assertNotIn('VLLM_DCP_Q_REPLICATE:', recipe)
        self.assertIn('"FULL_DECODE_ONLY"', recipe)


@unittest.skipUnless(BASELINE, "set GLM53_R22_PERF_BASELINE for exact-source behavioral checks")
class TestExactSource(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = {}
        for relative, transform in perf.TRANSFORMS.items():
            source = (Path(BASELINE) / relative).read_text(encoding="utf-8")
            assert perf.digest(source) == perf.INPUT_HASHES[relative], relative
            cls.sources[relative] = transform(source)
            assert perf.digest(cls.sources[relative]) == perf.OUTPUT_HASHES[relative], relative
            compile(cls.sources[relative], relative, "exec")

    def test_atomic_preflight_idempotence_and_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in perf.TRANSFORMS:
                dest = root / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(Path(BASELINE) / relative, dest)
            last = root / "envs.py"
            original = last.read_text(encoding="utf-8")
            last.write_text(original + "\n# drift\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                perf.patch(root)
            first = next(iter(perf.TRANSFORMS))
            self.assertEqual(perf.digest((root / first).read_text()), perf.INPUT_HASHES[first])
            last.write_text(original, encoding="utf-8")
            perf.patch(root)
            perf.patch(root)
            perf.patch(root, check=True)
            for relative in perf.TRANSFORMS:
                self.assertEqual(perf.digest((root / relative).read_text(encoding="utf-8")), perf.OUTPUT_HASHES[relative])

    def test_ckv_eligibility_keeps_decode_and_mixed_batches_out(self):
        source = self.sources["v1/attention/backends/mla/b12x_mla_sparse.py"]
        eligible = extract(source, "_use_b12x_full_ckv_gather")
        args = dict(enabled=True, is_glm_next=False, is_glm_dsa=True,
                    dcp_world_size=4, max_query_len=4096, num_tokens=4096,
                    num_decode_tokens=0, min_tokens=16, max_tokens=131072)
        self.assertTrue(eligible(**args))
        for override in (dict(enabled=False), dict(dcp_world_size=1),
                         dict(is_glm_dsa=False), dict(num_decode_tokens=4),
                         dict(num_tokens=16), dict(num_tokens=131073), dict(max_query_len=1)):
            self.assertFalse(eligible(**(args | override)), override)
        self.assertTrue(eligible(**(args | dict(is_glm_next=True, is_glm_dsa=False))))

    def test_dsa_builder_does_not_require_glmnext_kpool(self):
        source = self.sources["v1/attention/backends/mla/b12x_mla_sparse.py"]
        cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "B12xMLASparseMetadataBuilder")
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        start = next(i for i, n in enumerate(init.body) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "self._ckv_glm_dsa")
        body = init.body[start:start + 3]
        self.assertIsInstance(body[-1], ast.If)
        code = compile(ast.Module(body=body, type_ignores=[]), "<builder>", "exec")
        cfg = NS(model_config=NS(hf_text_config=NS(model_type="glm_moe_dsa", index_topk=2048)))
        for pcp, dtype, expected in ((False, "fp8_ds_mla", True), (True, "fp8_ds_mla", False), (False, "nvfp4_ds_mla", False)):
            obj = NS(kv_cache_spec=NS(cache_dtype_str=dtype), requires_glm_next_selector_metadata=False,
                     use_pcp=pcp, dcp_world_size=4)
            ns = dict(self=obj, vllm_config=cfg, envs=NS(VLLM_B12X_MLA_CKV_GATHER=True),
                      _is_glm_dsa_config=lambda c: c.model_type == "glm_moe_dsa",
                      torch=NS(empty=lambda shape, **kw: shape, int32="int32"),
                      max_tokens=4096, max_reqs=8, device="cuda")
            exec(code, ns)
            self.assertEqual(obj._ckv_gather_requested, expected)
            if expected:
                self.assertEqual(obj.ckv_selected_indices_buffer, (4096, 2048))

    def test_nvidia_attention_skips_both_collectives_only_for_full_ckv(self):
        source = self.sources["models/deepseek_v32/attention.py"]
        tree = ast.parse(source)
        function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                        and any(isinstance(t, ast.Name) and t.id == "full_ckv_dcp"
                                for s in n.body if isinstance(s, ast.Assign) for t in s.targets))
        start = next(i for i, n in enumerate(function.body) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "full_ckv_dcp")
        end = next(i for i, n in enumerate(function.body[start:], start) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "x")
        code = compile(ast.Module(body=function.body[start:end], type_ignores=[]), "<attention-dispatch>", "exec")
        for eligible in (True, False):
            calls = []
            obj = NS(use_pcp=False, impl=NS(dcp_world_size=4,
                uses_full_ckv_dcp=lambda *a: eligible,
                forward_mqa=lambda *a: (calls.append("attention") or "out", "lse")),
                dcp_manager=NS(query_gather=lambda q: calls.append("gather") or q,
                               combine=lambda *a, **kw: calls.append("combine") or "out"))
            exec(code, dict(self=obj, mqa_q_arg="q", kv_cache=None, attn_metadata=NS(), num_actual=4096))
            self.assertEqual(calls, ["attention"] if eligible else ["gather", "attention", "combine"])

    def test_roce_middle_dimension_gather_and_small_reduce_scatter(self):
        import numpy as np

        class Tensor:
            def __init__(self, data): self.data = np.asarray(data)
            @property
            def shape(self): return self.data.shape
            @property
            def dtype(self): return self.data.dtype
            device = "cuda"
            def dim(self): return self.data.ndim
            def size(self): return self.shape
            def numel(self): return self.data.size
            def element_size(self): return self.data.itemsize
            def is_contiguous(self): return self.data.flags.c_contiguous
            def contiguous(self): return Tensor(np.ascontiguousarray(self.data))
            def flatten(self, start_dim=0): return Tensor(self.data.reshape(*self.shape[:start_dim], -1))
            def view(self, shape): return Tensor(self.data.reshape(shape))
            def reshape(self, shape): return self.view(shape)
            def movedim(self, src, dst): return Tensor(np.moveaxis(self.data, src, dst))
            def narrow(self, dim, start, count):
                slices = [slice(None)] * self.dim()
                slices[dim] = slice(start, start + count)
                return Tensor(self.data[tuple(slices)])

        envs = NS(VLLM_BATCH_INVARIANT=False, VLLM_ROCE_DCP_RS_MAX_BYTES=262144)
        ns = dict(envs=envs, torch=NS(empty=lambda shape, dtype, device: Tensor(np.empty(shape, dtype))),
                  should_nccl_symm_mem_ag_rs=lambda: False,
                  current_platform=NS(is_rocm=lambda: False))
        source = self.sources["distributed/device_communicators/cuda_communicator.py"]
        gather = extract(source, "all_gather", "CudaCommunicator", ns)
        reduce = extract(source, "reduce_scatter", "CudaCommunicator", ns)
        calls = []
        class Roce:
            disabled = False
            def should_all_gather(self, x, dim): return x.is_contiguous() and dim in (0, x.dim() - 1) and x.numel() * x.element_size() <= 16 * 1024**2
            def all_gather(self, x, dim):
                calls.append("roce-ag")
                return Tensor(np.concatenate([x.data + r for r in range(4)], axis=dim))
            def should_custom_ar(self, x): return x.is_contiguous()
            def custom_all_reduce(self, x):
                calls.append("roce-ar")
                return Tensor(x.data * 4)
        class Nccl:
            disabled = False
            def reduce_scatter(self, output, x):
                calls.append("nccl-rs")
                output.data[:] = (x.data * 4)[rank * output.shape[0]:(rank + 1) * output.shape[0]]
            def all_gather(self, output, x):
                calls.append("nccl-ag")
                output.data[:] = np.concatenate([x.data + r for r in range(4)])
        obj = NS(b12x_ar_comm=Roce(), use_roce_allreduce=True, world_size=4,
                 unique_name="dcp:0", rank_in_group=0, pynccl_comm=Nccl())
        for shape, dim in (((4, 16, 576), 1), ((2, 3, 5, 7), 1), ((2, 3, 5, 7), 2)):
            x = Tensor(np.arange(np.prod(shape), dtype=np.float32).reshape(shape))
            calls.clear()
            out = gather(obj, x, dim)
            np.testing.assert_array_equal(out.data, np.concatenate([x.data + r for r in range(4)], axis=dim))
            self.assertEqual(calls, ["roce-ag"])
        # Four MTP rows fit the cap; larger shapes use NCCL. Check every rank.
        for rank in range(4):
            obj.rank_in_group = rank
            for rows in (1, 4, 8, 32):
                x = Tensor(np.linspace(-1, 1, rows * 64 * 512, dtype=np.float32)
                           .reshape(rows, 64, 512).astype(np.float16))
                calls.clear()
                out = reduce(obj, x, 1)
                np.testing.assert_array_equal(out.data, (x.data * 4)[:, rank * 16:(rank + 1) * 16])
                self.assertEqual(calls, ["roce-ar"] if rows <= 4 else ["nccl-rs"])
        envs.VLLM_BATCH_INVARIANT = True
        calls.clear()
        reduce(obj, Tensor(np.ones((4, 64, 512), dtype=np.float16)), 1)
        self.assertEqual(calls, ["nccl-rs"])
        envs.VLLM_BATCH_INVARIANT = False
        envs.VLLM_ROCE_DCP_RS_MAX_BYTES = 0
        calls.clear()
        reduce(obj, Tensor(np.ones((4, 64, 512), dtype=np.float16)), 1)
        self.assertEqual(calls, ["nccl-rs"])


if __name__ == "__main__":
    unittest.main()
