"""Reproduce the recipe's RoCEnante enable gates without CUDA or RDMA."""
import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent/'tmp/r22-prefill-comparison/baseline-source/vllm'


class Tests(unittest.TestCase):
    def test_recipe_exposes_transport_and_does_not_disable_it(self):
        source = (ROOT/'recipes/glm53-exl3-v19-4x.yaml').read_text()
        for expected in ('VLLM_ENABLE_ROCE_ALLREDUCE: "1"', 'VLLM_ROCE_DCP_ENABLE: "1"',
                         'VLLM_ENABLE_PCIE_ALLREDUCE: "0"', 'network: host', 'ipc: host',
                         'memlock=-1:-1', '/dev/infiniband:/dev/infiniband'):
            self.assertIn(expected, source)
        self.assertNotIn('--disable-custom-all-reduce', source)

    def test_real_tp_and_dcp_selection_with_recipe_cli_flag(self):
        relative = 'distributed/device_communicators/cuda_communicator.py'
        if not (BASE/relative).exists(): self.skipTest('pinned communicator fixture unavailable')
        spec = importlib.util.spec_from_file_location('perf', ROOT/'overlay/patch_r22_performance.py')
        perf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(perf)
        original = (BASE/relative).read_text(encoding='utf-8')
        self.assertEqual(perf.digest(original), perf.INPUT_HASHES[relative])
        source = perf.patch_communicator(original)
        cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'CudaCommunicator')
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        # Execute the actual two selection regions, before constructing GPU
        # communicators. Imports receive the same flag the worker sets from CLI.
        gates = [n for n in init.body if isinstance(n, ast.If) and 'unique_name' in ast.unparse(n.test)]
        self.assertEqual(len(gates), 2)
        code = compile(ast.fix_missing_locations(ast.Module(body=gates, type_ignores=[])), '<R22 gates>', 'exec')
        for disabled in (True, False):
            for group in ('tp:0', 'dcp:0', 'pp:0'):
                for transport_enabled in (True, False):
                    ns = dict(unique_name=group, envs=NS(
                        VLLM_ALLREDUCE_USE_SYMM_MEM=False, VLLM_ALLREDUCE_USE_FLASHINFER=False,
                        VLLM_BATCH_INVARIANT=False, VLLM_ENABLE_PCIE_ALLREDUCE=False,
                        VLLM_PCIE_ALLREDUCE_BACKEND='b12x', VLLM_ENABLE_ROCE_ALLREDUCE=transport_enabled,
                        VLLM_ROCE_DCP_ENABLE=True), rocm_aiter_ops=NS(is_custom_all_reduce_enabled=lambda: False))
                    with patch.dict(sys.modules, {'vllm.distributed.parallel_state':
                                                  NS(_ENABLE_CUSTOM_ALL_REDUCE=not disabled)}):
                        exec(code, ns)
                    self.assertEqual(ns['use_roce_allreduce'],
                                     not disabled and transport_enabled and group != 'pp:0')
                    self.assertFalse(ns['use_b12x_allreduce'])


if __name__ == '__main__': unittest.main()
