"""Execute the pinned splitter and compact metadata kernel without a GPU."""
import ast
from dataclasses import dataclass
import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = Path(os.getenv('GLM53_V18_BASELINE', ROOT.parent/'tmp/exl3-v18/base'))


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'overlay'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


p = load('patch_r22_v18')


class Tensor:
    def __init__(self, a, device='cpu'):
        self.a = np.asarray(a)
        self.device = device
    def __len__(self): return len(self.a)
    @property
    def shape(self): return self.a.shape
    def __getitem__(self, key): return Tensor(self.a[key], self.device)
    def item(self):
        assert self.device == 'cpu', 'GPU-to-CPU synchronization'
        return self.a.item()
    def clamp(self, max): return Tensor(np.minimum(self.a, max), self.device)
    def __add__(self, index): return Pointer(self, index)


class Pointer:
    def __init__(self, tensor, index): self.tensor, self.index = tensor, index
    def __add__(self, index): return Pointer(self.tensor, self.index + index)


class Language:
    constexpr = int
    block = 0
    def program_id(self, _): return self.block
    arange = staticmethod(np.arange)
    minimum = staticmethod(np.minimum)
    maximum = staticmethod(np.maximum)
    def load(self, ptr, mask=True, other=0):
        if not isinstance(ptr, Pointer): ptr = Pointer(ptr, 0)
        index = np.asarray(ptr.index)
        mask = np.broadcast_to(mask,index.shape)
        result = np.full(index.shape,other,dtype=ptr.tensor.a.dtype)
        result[mask] = ptr.tensor.a.reshape(-1)[index[mask]]
        return result
    def store(self, ptr, value, mask=True):
        if not isinstance(ptr, Pointer): ptr = Pointer(ptr, 0)
        index = np.asarray(ptr.index)
        mask = np.broadcast_to(mask, index.shape)
        values = np.broadcast_to(value, index.shape)
        ptr.tensor.a.reshape(-1)[index[mask]] = values[mask]


def extracted(source, name, namespace):
    node = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec('from __future__ import annotations\n' + ast.unparse(node), namespace)
    return namespace[name]


class Tests(unittest.TestCase):
    def test_topk_oracle_accepts_boundary_ties_but_rejects_wrong_results(self):
        smoke = load('smoke_r22_v18')
        logits = np.array([[10.,9.,9.,9.,2.,1.], [1.,2.,7.,7.,8.,9.]])
        lengths = np.array([6,6])
        # Distinct valid selections at the boundary; output order is arbitrary.
        for ids in (np.array([[0,1,2],[3,4,5]]), np.array([[3,0,2],[5,2,4]])):
            smoke.assert_topk_result(ids, np.take_along_axis(logits,ids,axis=1), logits,lengths)
        ids = np.array([[0,1,2],[3,4,5]])
        values = np.take_along_axis(logits,ids,axis=1)
        for bad in (np.array([[0,1,4],[3,4,5]]), np.array([[0,1,1],[3,4,5]]),
                    ids[::-1], np.array([[-1,1,2],[3,4,5]]), np.array([[6,1,2],[3,4,5]])):
            with self.assertRaises(AssertionError):
                smoke.assert_topk_result(bad, np.take_along_axis(logits,np.clip(bad,0,5),axis=1), logits,lengths)
        with self.assertRaises(AssertionError):
            smoke.assert_topk_result(ids, values+1, logits,lengths)
        with self.assertRaises(AssertionError):
            smoke.assert_topk_result(ids, values*np.nan, logits,lengths)
        with self.assertRaises(AssertionError):
            smoke.assert_topk_result(ids, values, logits,np.array([2,6]))

    def source(self, name):
        if not BASE.is_dir(): self.skipTest('set GLM53_V18_BASELINE to pinned v17 sources')
        return (BASE/name).read_text(encoding='utf-8')

    def test_preflight_idempotence_and_no_partial_writes(self):
        self.source(p.COMMON)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'source'; shutil.copytree(BASE, root)
            p.patch(root); p.patch(root); p.patch(root, check=True)
            for name, digest in p.OUTPUTS.items():
                self.assertEqual(hashlib.sha256((root/name).read_text(encoding='utf-8').encode()).hexdigest(), digest)
            shutil.copyfile(BASE/p.COMMON, root/p.COMMON)
            (root/p.INDEXER).write_text('# drift', encoding='utf-8')
            before = {f:f.read_bytes() for f in root.rglob('*') if f.is_file()}
            with self.assertRaises(RuntimeError): p.patch(root)
            self.assertEqual(before, {f:f.read_bytes() for f in root.rglob('*') if f.is_file()})

    def splitter(self):
        ns = {}
        extracted(self.source(p.COMMON), 'split_indexer_prefill_chunks', ns)
        return extracted(p.indexer(self.source(p.INDEXER)), '_split_prefill_chunks', ns)

    def test_chunk_coverage_offsets_tails_and_existing_profile_budget(self):
        split = self.splitter()
        owner = NS(_v18_coalesce=True, max_prefill_buffer_size=40*1048576)
        counts = {}
        for context in (8192, 32768, 65536, 131072, 1048576):
            lens = Tensor([101, context, context + 3])
            queries = Tensor([4096, 2177])
            for enabled in (False, True):
                owner._v18_coalesce = enabled
                chunks = split(owner, lens, queries, 1, 256*1024*1024)
                for request, rows in ((1,4096), (2,2177)):
                    covered = [i for req, q in chunks if req.start == request for i in range(q.start,q.stop)]
                    self.assertEqual(covered, list(range(rows)))
                    self.assertTrue(all(req.stop == req.start + 1 for req, _ in chunks))
                counts[context, enabled] = sum(req.start == 1 for req, _ in chunks)
                if enabled and context >= 32768:
                    self.assertTrue(all(q.stop-q.start <= 2048 for _,q in chunks))
            np.testing.assert_array_equal(lens.a, [101,context,context+3])
        self.assertEqual([counts[c,False] for c in (8192,32768,65536,131072,1048576)], [1,2,4,8,64])
        self.assertEqual([counts[c,True] for c in (8192,32768,65536,131072,1048576)], [1,2,2,2,2])

    def test_compact_metadata_executes_actual_kernel_without_device_item(self):
        common_source = self.source(p.COMMON)
        ns = {'dataclass':dataclass}
        node = next(n for n in ast.walk(ast.parse(common_source)) if isinstance(n, ast.ClassDef) and n.name == 'DeepseekV32IndexerPrefillChunkMetadata')
        exec('from __future__ import annotations\n'+ast.unparse(node),ns)
        lang = Language()
        def jit(fn):
            class Kernel:
                def __getitem__(self, grid):
                    def run(*args):
                        for block in range(grid[0]):
                            lang.block = block; fn(*args)
                    return run
            return Kernel()
        fake_torch = NS(Tensor=Tensor, int32=np.int32,
            empty=lambda shape,dtype,device:Tensor(np.empty(shape,dtype=dtype),device),
            empty_like=lambda t:Tensor(np.empty_like(t.a),t.device))
        fallback_calls = []
        common = NS(DeepseekV32IndexerPrefillChunkMetadata=ns[node.name],
                    build_prefill_chunk_metadata=lambda *a,**kw:fallback_calls.append((a,kw)))
        with patch.dict(sys.modules, {'torch':fake_torch,
            'vllm.triton_utils':NS(tl=lang,triton=NS(jit=jit,cdiv=lambda x,y:(x+y-1)//y)),
            'vllm.v1.attention.backends.mla.indexer':common}):
            m = load('gb10_indexer_prefill')
        for world in (1,2,4):
            for interleave in (1,16,64):
                for rank in range(world):
                    for total, upper in ((1027,1027),(8195,8208)):
                        qcpu = Tensor([0,4,517])
                        q = Tensor(qcpu.a,'cuda')
                        seq = Tensor([20,total],'cuda')
                        seqcpu = Tensor([20,upper])
                        table = Tensor(np.arange(4096).reshape(2,-1),'cuda')
                        args = (1,2,q,qcpu,seq,seq,seqcpu,table,1)
                        result = m.build_paged_chunk(*args,query_slice=slice(127,513),dcp_rank=rank,
                            dcp_world_size=world,cp_kv_cache_interleave_size=interleave)
                        owner_counts = np.cumsum((np.arange(total)//interleave)%world == rank)
                        np.testing.assert_array_equal(result.b12x_seq_lens.a, owner_counts[total-513+127:total])
                        np.testing.assert_array_equal(result.cu_seqlen_ks.a,np.zeros(386))
                        np.testing.assert_array_equal(result.local_cu_seq_lens.a,[0,owner_counts[-1]])
                        np.testing.assert_array_equal(result.cu_seq_lens.a,[0,total])
                        self.assertEqual(result.token_to_seq.a.size,0)
                        self.assertEqual((result.token_start,result.token_end),(131,517))
                        self.assertIs(result.b12x_seq_lens,result.cu_seqlen_ke)
                        self.assertGreaterEqual(result.local_total_seq_lens,owner_counts[-1])
                        pages=(owner_counts[-1]+63)//64
                        np.testing.assert_array_equal(result.block_table.a[0,:pages],table.a[1,:pages])
                        np.testing.assert_array_equal(result.block_table.a[0,pages:],-np.ones(result.block_table.shape[1]-pages))
        m.build_paged_chunk(*args[:-1],2)
        self.assertEqual(len(fallback_calls),1)
        with self.assertRaises(ValueError): m.build_paged_chunk(*args,query_slice=slice(513,513))
        zero_args = (*args[:6],Tensor([20,0]),*args[7:])
        self.assertIsNone(m.build_paged_chunk(*zero_args))

    def test_forward_uses_precomputed_lengths_and_preserves_decode(self):
        source = p.indexer(self.source(p.INDEXER))
        original = self.source(p.INDEXER)
        self.assertEqual(source.split('        if metadata.decode is not None:')[-1],
                         original.split('        if metadata.decode is not None:')[-1])
        self.assertIn('seq_lens = getattr(chunk, "b12x_seq_lens", None)',source)
        # The common path stays generic unless the subclass supplies the hook.
        self.assertIn('self, "_build_prefill_chunk_metadata", build_prefill_chunk_metadata',p.common(self.source(p.COMMON)))
        self.assertIn('== (12, 1)',source)
        self.assertIn('== 32768)',source)

    def test_inherited_smoke_manifest_merges_without_dropping_old_entries(self):
        source = (ROOT/'overlay/smoke_r22_v16.py').read_text(encoding='utf-8')
        node = next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Assign)
                    and any(isinstance(t,ast.Name) and t.id=='hashes' for t in n.targets))
        ns = dict(v11={'old':'keep'},v13={},v14={},v15={},OUTPUTS={p.INDEXER:'v16','other':'keep'},
                  source_overrides=p.OUTPUTS)
        exec(ast.unparse(node),ns)
        self.assertEqual(ns['hashes']['old'],'keep')
        self.assertEqual(ns['hashes']['other'],'keep')
        self.assertEqual(ns['hashes'][p.INDEXER],p.OUTPUTS[p.INDEXER])

    def test_recipe_only_adds_v18_and_builder_inherits_v17(self):
        before=(ROOT/'recipes/glm53-exl3-v17-4x.yaml').read_text(encoding='utf-8')
        after=(ROOT/'recipes/glm53-exl3-v18-4x.yaml').read_text(encoding='utf-8')
        restored=after.replace('r22-v18-mtp','r22-v17-mtp').replace('sm121-v18','sm121-v17')
        restored='\n'.join(line for line in restored.split('\n') if not any(key in line for key in
            ('v18_overlay:','VLLM_GB10_INDEXER_COALESCE:','VLLM_GB10_INDEXER_METADATA:')))
        self.assertEqual(restored,before)
        builder=(ROOT/'scripts/build-r22-v18-image.sh').read_text(encoding='utf-8')
        self.assertIn('for version in 11 12 13 14 15 16 17;',builder)
        self.assertIn('GLM53_R22_SMOKE_SCRIPT=smoke_r22_v18.py',builder)


if __name__ == '__main__': unittest.main()
