"""CPU tests against the pinned B12X fixture; no CUDA imports required."""
import ast
import dataclasses
import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("v11", ROOT / "overlay/patch_r22_v11.py")
overlay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(overlay)
BASE = Path(os.environ.get("GLM53_V11_BASELINE", ROOT.parent / "tmp/exl3-v11/baseline/b12x"))


class BuildTests(unittest.TestCase):
    def test_v11_is_separate_and_runs_gpu_gate_on_every_node(self):
        docker = (ROOT / "Dockerfile.r22-dflash2-v11").read_text()
        builder = (ROOT / "scripts/build-r22-v11-image.sh").read_text()
        base = (ROOT / "scripts/build-r22-dflash2-image.sh").read_text()
        recipe = (ROOT / "recipes/glm53-exl3-v11-4x.yaml").read_text()
        self.assertIn("FROM spark-vllm-glm53-exl3:r22-dflash2-sm121-v10", docker)
        self.assertIn("GLM53_R22_V11_SMOKE=1", builder)
        self.assertIn("Dockerfile.r22-dflash2-v11", builder)
        self.assertEqual(base.count("/opt/compose/smoke_r22_v11.py --gpu"), 2)
        self.assertIn('VLLM_EXL3_MIXED_FUSED_OUTPUT: "1"', recipe)
        self.assertIn("r22-dflash2-sm121-v11", recipe)
        self.assertIn("r22-dflash2-sm121-v10", base)


class V11Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not BASE.is_dir():
            raise unittest.SkipTest("set GLM53_V11_BASELINE to the pinned v10 B12X package")
        cls.sources = {name: transform((BASE / name).read_text(encoding="utf-8"))
                       for name, transform in overlay.TRANSFORMS.items()}

    def test_kernel_fixture_matches_upstream_git_blob(self):
        # Independent of the overlay's SHA256 constants: a Windows-default
        # decode/re-encode previously corrupted comments in both the fixture
        # and its expected hashes, allowing the self-consistent tests to pass.
        source = (BASE / "moe/_shared/kernels/w4a16/kernel.py").read_text(encoding="utf-8")
        raw = source.encode("utf-8")
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        # b12x commit 1e59a1fd09f782d302b1068b15c8a0bd66103894
        self.assertEqual(blob, "537ce1c7b75514a4f26988cf1b928452debe5059")
        self.assertEqual(hashlib.sha256(raw).hexdigest(),
                         "591d06f211229703fc465f745db87241675d4b70fbcbb7af96c3d203642567c8")

    def test_hashes_idempotence_and_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "b12x"
            shutil.copytree(BASE, root)
            overlay.patch(root)
            overlay.patch(root)
            overlay.patch(root, check=True)
            for name, source in self.sources.items():
                self.assertEqual((root / name).read_text(encoding="utf-8"), source)
            shutil.rmtree(root)
            shutil.copytree(BASE, root)
            last = root / list(overlay.TRANSFORMS)[-1]
            last.write_text(last.read_text(encoding="utf-8") + "\n# drift\n", encoding="utf-8")
            before = {p: p.read_bytes() for p in root.rglob("*.py")}
            with self.assertRaises(RuntimeError):
                overlay.patch(root)
            self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_compiler_dtype_cache_and_legacy_default(self):
        tree = ast.parse(self.sources["moe/_shared/kernels/w4a16/kernel.py"])
        nodes = [x for x in tree.body if getattr(x, "name", "") in
                 ("W4A16TopKSumCompileResult", "compile_w4a16_topk_sum")]
        torch = NS(int32="i32", int64="i64")
        calls = []
        def compiler(*args, **kwargs):
            result = NS(args=args, kwargs=kwargs)
            calls.append(result)
            return result
        ns = dict(dataclass=dataclasses.dataclass, replace=dataclasses.replace,
                  torch=torch, cutlass=NS(Float32="fp32", Float16="fp16", Int32="i32", Int64="i64"),
                  _cutlass_element_dtype=lambda dtype: dtype, _SUM_CACHE={},
                  make_ptr=lambda dtype, *a, **k: dtype, cute=NS(AddressSpace=NS(gmem=0)),
                  W4A16TopKSumKernel=lambda **kw: kw, b12x_compile=compiler,
                  raise_if_kernel_resolution_frozen=lambda *a, **k: None,
                  current_cuda_stream=lambda: 0, Int32=int,
                  KernelCompileSpec=NS(from_key=lambda *args: args))
        exec("from __future__ import annotations\n" + ast.unparse(ast.Module(body=nodes, type_ignores=[])), ns)
        compile_sum = ns["compile_w4a16_topk_sum"]
        common = dict(m=4, topk=8, hidden_size=6144)
        legacy = compile_sum(**common)
        self.assertEqual(legacy.output_element_dtype, "bf16")
        common.update(full_rotation=True, element_dtype="fp16", num_experts=6)
        fp32 = compile_sum(**common)
        bf16 = compile_sum(**common, output_element_dtype="bf16")
        fp16 = compile_sum(**common, output_element_dtype="fp16")
        self.assertEqual([x.compiled.args[2] for x in (fp32, bf16, fp16)], ["fp32", "bf16", "fp16"])
        self.assertIsNot(fp32.compiled, bf16.compiled)
        self.assertIs(compile_sum(**common, output_element_dtype="bf16").compiled, bf16.compiled)
        self.assertEqual(len(calls), 4)
        with self.assertRaises(ValueError):
            compile_sum(**common, output_element_dtype="int8")

    def test_both_mixed_tiers_bind_the_compiled_dtype(self):
        tree = ast.parse(self.sources["moe/_shared/kernels/w4a16/mixed_trellis.py"])
        for name in ("compile_mixed_trellis", "compile_mixed_trellis3"):
            node = next(n for n in tree.body if getattr(n, "name", "") == name)
            call = next(n for n in ast.walk(node) if isinstance(n, ast.Call)
                        and isinstance(n.func, ast.Name) and n.func.id == "compile_w4a16_topk_sum")
            expr = next(k.value for k in call.keywords if k.arg == "output_element_dtype")
            code = compile(ast.Expression(expr), "dtype", "eval")
            for enabled, expected in (("0", "fp32"), ("1", "bf16")):
                self.assertEqual(eval(code, {"os": NS(environ={"VLLM_EXL3_MIXED_FUSED_OUTPUT": enabled}),
                                             "rotation_input_dtype": "bf16"}), expected)
        text = ast.unparse(tree)
        self.assertEqual(text.count("_cutlass_element_dtype(launch.topk_sum.output_element_dtype)"), 2)
        self.assertIn("[launch.topk_sum.output_element_dtype]", text)

    def test_roce_grid_uses_matching_counters_and_launch(self):
        text = self.sources["comm/roce/roce_oneshot.py"]
        tree = ast.parse(text)
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_launch_gather")
        source = ast.unparse(method)
        self.assertIn("_grid_blocks(nbytes // PACK_BYTES, self._threads, self._blocks)", source)
        self.assertIn("self._counter_addresses(grid_blocks)", source)
        self.assertIn("self.spin_limit, grid_blocks)", source)


if __name__ == "__main__":
    unittest.main()
