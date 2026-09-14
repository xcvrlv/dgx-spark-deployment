"""Execute pinned JJ's actual graph enumeration/compatibility on CPU."""
import ast
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import importlib.util
from itertools import groupby, product
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get('DS41_VLLM_SOURCE', ROOT.parent / 'tmp/jj-audit/local-inference-lab-vllm-c9dc4e5/vllm'))
spec = importlib.util.spec_from_file_location('graph_patch', ROOT / 'patches/graph_requests.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)

class Mode(Enum):
    FULL = 1
    PIECEWISE = 2

class Combined:
    def decode_mode(self): return Mode.FULL
    def mixed_mode(self): return Mode.PIECEWISE
    def separate_routine(self): return True


def planner(text, maximum, depth=5, enabled=True, varlen=True, cap=None):
    tree = ast.parse(text)
    manager = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CudaGraphManager')
    method = next(n for n in manager.body if isinstance(n, ast.FunctionDef) and n.name == '_init_candidates')
    desc = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BatchExecutionDescriptor')
    compatible = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_is_compatible')
    scope = dict(dataclass=dataclass, defaultdict=defaultdict, product=product, groupby=groupby,
                 CUDAGraphMode=Mode, os=NS(getenv=lambda *a: '1' if enabled else '0'),
                 _DENSE_VARLEN_DECODE_MAX_REQS=2, round_up=lambda a,b: (a+b-1)//b*b,
                 logger=NS(info=lambda *a: None))
    exec('from __future__ import annotations\n'+ '\n'.join(ast.unparse(n) for n in (desc, compatible, method)), scope)
    limit = maximum*(depth+1) if cap is None else cap
    config = NS(num_speculative_tokens=depth, speculative_config=NS(
        uses_acceptance_length_adaptation=lambda: False,
        uses_batch_size_dynamic_speculative_decoding=lambda: False, use_dspark=lambda: True))
    owner = NS(compilation_config=NS(cudagraph_capture_sizes=list(range(1,limit+1)),max_cudagraph_capture_size=limit),
               max_num_reqs=maximum, decode_query_len=depth+1, cudagraph_mode=Combined(),
               specialize_full_decode=False, vllm_config=config, varlen_decode=varlen,
               lora_capture_cases=[0], full_capture_request_sizes=None,
               single_request_prefill_tokens=0, use_breakable_cg=False,
               _capture_descs={}, _candidates={})
    scope['_init_candidates'](owner)
    def choose(reqs, tokens, width=depth+1):
        for candidate in owner._candidates.get((tokens,0),[]):
            if scope['_is_compatible'](candidate, reqs, tokens, None, 0, width):
                return candidate
    return owner, choose


@unittest.skipUnless(SOURCE.is_dir(), 'set DS41_VLLM_SOURCE to pinned JJ package')
class GraphCoverageTests(unittest.TestCase):
    def setUp(self):
        self.original = (SOURCE / p.RELATIVE).read_bytes()
        self.updated = p.transform(self.original).decode()

    def test_all_c8_mixed_lengths_match_max8_with_max16(self):
        for depth in (1,3,5,7):
            old, baseline = planner(self.original.decode(),8,depth)
            current, select = planner(self.updated,16,depth)
            for tokens in range(8,8*(depth+1)+1):
                a,b = baseline(8,tokens), select(8,tokens)
                self.assertEqual((a.cg_mode,a.num_tokens,a.num_reqs), (b.cg_mode,b.num_tokens,b.num_reqs))
            for reqs in range(1,17):
                for tokens in range(reqs,reqs*(depth+1)+1):
                    d=select(reqs,tokens)
                    self.assertEqual(d.cg_mode,Mode.FULL)
                    self.assertEqual(d.num_tokens,tokens)
                    if reqs <= 8:
                        self.assertEqual(d.num_reqs,reqs)
                    self.assertGreaterEqual(d.num_reqs,reqs)
                    self.assertLessEqual(d.num_reqs,16)
            self.assertEqual(len(current._capture_descs[Mode.FULL]),len(set(current._capture_descs[Mode.FULL])))

    def test_c8_capped_recipe_covers_every_request_count(self):
        owner, choose = planner(self.updated,8,5)
        for reqs in range(1,9):
            for tokens in range(reqs,reqs*6+1):
                d=choose(reqs,tokens)
                self.assertEqual((d.cg_mode,d.num_tokens,d.num_reqs),(Mode.FULL,tokens,reqs))
        self.assertEqual(len(owner._capture_descs[Mode.FULL]),188)

    def test_c4_exact_and_prefill_does_not_enter_decode(self):
        _, choose=planner(self.updated,16)
        for tokens in range(4,25):
            d=choose(4,tokens)
            self.assertEqual((d.num_tokens,d.num_reqs),(tokens,4))
        self.assertEqual(choose(4,32,8).cg_mode,Mode.PIECEWISE)

    def test_runtime_rollback_and_fixed_verification_unchanged(self):
        for varlen in (False,True):
            old,_=planner(self.original.decode(),16,varlen=varlen)
            updated,_=planner(self.updated,16,enabled=False,varlen=varlen)
            norm=lambda m: {k:[(d.num_tokens,d.num_reqs,d.uniform_token_count,d.max_query_len) for d in v] for k,v in m._capture_descs.items()}
            self.assertEqual(norm(old),norm(updated))
            if not varlen:
                enabled,_=planner(self.updated,16,varlen=False)
                self.assertEqual(norm(old),norm(enabled))

    def test_cap_and_hash_rollback(self):
        owner,_=planner(self.updated,16,cap=48)
        self.assertTrue(all(d.num_tokens<=48 for v in owner._capture_descs.values() for d in v))
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/p.RELATIVE
            target.parent.mkdir(parents=True)
            target.write_bytes(self.original)
            p.patch(directory)
            p.patch(directory)
            p.patch(directory,check=True)
            p.patch(directory,revert=True)
            self.assertEqual(target.read_bytes(),self.original)
            target.write_bytes(self.original+b'\n')
            with self.assertRaisesRegex(RuntimeError,'Unexpected upstream'):
                p.patch(directory)

if __name__=='__main__': unittest.main()
