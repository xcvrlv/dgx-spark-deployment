"""Execute the actual upstream cache method with recording cache dependencies."""
import ast
import importlib.util
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prefill_hashes', ROOT / 'patches/prefill_hashes.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
SOURCE = Path(os.environ.get('DS41_VLLM_SOURCE', ROOT.parent / 'tmp/jj-audit/local-inference-lab-vllm-5bca5a5/vllm'))


@unittest.skipUnless(SOURCE.is_dir(), 'set DS41_VLLM_SOURCE to the pinned vllm package')
class PrefillHashTests(unittest.TestCase):
    def test_hash_guard_apply_reapply_revert_and_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / p.RELATIVE
            target.parent.mkdir(parents=True)
            original = (SOURCE / p.RELATIVE).read_bytes()
            target.write_bytes(original)
            p.patch(directory)
            p.patch(directory)
            p.patch(directory, check=True)
            p.patch(directory, revert=True)
            self.assertEqual(target.read_bytes(), original)
            target.write_bytes(original + b'\n')
            with self.assertRaisesRegex(RuntimeError, 'Unexpected upstream'):
                p.patch(directory)

    def test_real_cache_method_equivalence_and_copy_work(self):
        text = (SOURCE / p.RELATIVE).read_text()
        utils = ast.parse((SOURCE / 'v1/core/kv_cache_utils.py').read_text())
        view = next(n for n in utils.body if isinstance(n, ast.ClassDef) and n.name == 'BlockHashListWithBlockSize')
        scope = {}
        exec('from __future__ import annotations\nfrom typing import overload\n' + ast.unparse(view), scope)
        view_type = scope['BlockHashListWithBlockSize']

        def run(source, scale, events):
            tree = ast.parse(source)
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BlockPool')
            method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'cache_full_blocks')
            copies = []
            class CountingHashes:
                def __getitem__(self, key):
                    result = values[key]
                    if isinstance(key, slice):
                        copies.append(len(result))
                    return result
            raw = list(range(1536 * scale))
            values = raw if scale == 1 else view_type(raw, 256 // scale, 256)
            hashes = CountingHashes()
            calls = []
            owner = NS(hash_block_size=256 // scale, enable_kv_cache_events=events,
                       _insert_block_hash=lambda h, b, **kw: calls.append(('insert', h, b.id, kw)),
                       _remove_cached_block_hashes=lambda b: [('old', b.id)],
                       _emit_block_removed_events=lambda h: calls.append(('remove', h)),
                       _emit_stored_block_runs=lambda req, ids, *a, **kw: calls.append(('event', ids, a, kw)),
                       _published_full_block_context_start=lambda *a: 0)
            env = {'resolve_block_hashes': lambda *a: hashes,
                   'make_block_hash_with_group_id': lambda h, g: (h, g)}
            exec('from __future__ import annotations\n' + ast.unparse(method), env)
            blocks = [NS(id=i, is_null=i % 17 == 0,
                         block_hash='old' if i % 19 == 0 else None,
                         block_hash_num_tokens=1) for i in range(1536)]
            for start in range(0, 1536, 16):
                end = min(start + 16, 1536)
                mask = None if start % 32 else [i % 3 != 0 for i in range(end-start)]
                env['cache_full_blocks'](owner, NS(block_hashes=raw), blocks, start, end, 256, 2, mask)
            env['cache_full_blocks'](owner, NS(block_hashes=raw), blocks, 1536, 1536, 256, 2)
            return calls, sum(copies)

        for scale in (1, 2, 4):
            for events in (False, True):
                with self.subTest(scale=scale, events=events):
                    before, old_work = run(text, scale, events)
                    after, new_work = run(text.replace(p.OLD.decode(), p.NEW.decode()), scale, events)
                    self.assertEqual(before, after)
                    self.assertEqual(old_work, 74496)
                    self.assertEqual(new_work, 1536)


if __name__ == '__main__':
    unittest.main()
