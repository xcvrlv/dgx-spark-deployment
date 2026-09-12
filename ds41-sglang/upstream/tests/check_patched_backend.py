"""CPU checks of the real patched source, without importing CUDA dependencies."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace


def check(path):
    tree = ast.parse(Path(path).read_text())
    init = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'init_flashmla_related')
    gate = next(n for n in init.body if isinstance(n, ast.Assert))
    code = compile(ast.fix_missing_locations(ast.Module(body=[gate], type_ignores=[])), '<gate>', 'exec')
    for k in (512, 1024, 2048):
        exec(code, {'self': SimpleNamespace(index_topk=k)})
    try:
        exec(code, {'self': SimpleNamespace(index_topk=256)})
    except AssertionError:
        pass
    else:
        raise AssertionError('Gate unexpectedly accepts 256')

    decode = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_low_ratio_index_topk_decode')
    dispatch = next(n for n in decode.body if isinstance(n, ast.If) and 'metadata.use_topk_v2' in ast.unparse(n.test))
    code = compile(ast.fix_missing_locations(ast.Module(body=[dispatch], type_ignores=[])), '<dispatch>', 'exec')
    for filtered in (False, True):
        for needs_raw in (False, True):
            calls = []
            raw, selected, pages = object() if needs_raw else None, object(), object()
            metadata = SimpleNamespace(use_topk_v2=True, c4_seq_lens=object(), page_table=object(), topk_metadata=object())
            def v1(*args):
                raise AssertionError('Fell back to the 1024-limited kernel')
            exec(code, dict(metadata=metadata, raw_indices=raw, selected=selected,
                            page_indices=pages, filter_candidates=filtered,
                            logits=object(), page_size=64,
                            topk_transform_paged=v1,
                            topk_transform_paged_v2=lambda *args: calls.append(args)))
            assert len(calls) == 1
            args = calls[0]
            assert args[2] is (None if filtered else metadata.page_table)
            assert args[3] is (selected if filtered else pages)
            assert args[6] is (None if filtered else raw)
    print('Metadata gate and all four raw/candidate v2 dispatch combinations passed')


if __name__ == '__main__':
    check(sys.argv[1] if len(sys.argv) > 1 else '/sgl-workspace/sglang/python/sglang/srt/layers/attention/deepseek_v4_backend.py')
