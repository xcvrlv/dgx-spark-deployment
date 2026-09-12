"""Expand the known V4.1 metadata gate; refuse unknown kernel implementations."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def patch(root):
    root = Path(root)
    backend = root / 'srt/layers/attention/deepseek_v4_backend.py'
    source = backend.read_text()
    before_hash = hashlib.sha256(source.encode()).hexdigest()
    if before_hash != '486900ed693af0573a941d6b6656ebc01ccb7ff1c44584e43ad3c0fb00a3c3a9':
        raise RuntimeError('Backend differs from audited image revision da64c5c; re-audit before patching')
    old = 'assert self.index_topk in (512, 1024), ('
    new = 'assert self.index_topk in (512, 1024, 2048), ('
    if source.count(old) != 1 or new in source:
        raise RuntimeError('Unknown/already patched V4.1 metadata gate; audit this image first')
    headers = [root / 'kernels/jit/include/sgl_kernel/deepseek_v4/topk_impl.cuh',
               root / 'kernels/jit/csrc/deepseek_v4/topk_v2.cuh']
    combined = '\n'.join(p.read_text() for p in headers)
    limits = re.findall(r'kMaxTopK\s*=\s*(\d+)\s*;', combined)
    if not limits or min(map(int, limits)) < 2048:
        raise RuntimeError('The installed top-k v2 kernel does not declare support for 2048')
    for relative, required in {
        'kernels/ops/attention/dsv4/topk.py': ['def topk_transform_paged_v2(', 'out_raw_indices: Optional[torch.Tensor] = None'],
        'srt/layers/attention/dsa/dsa_topk_backend.py': ['SGLANG_OPT_USE_TOPK_V2'],
        'srt/configs/deepseek_v41.py': ['normalize_deepseek_v41_config', 'values = {**text, **values}'],
    }.items():
        text = (root / relative).read_text()
        if not all(token in text for token in required):
            raise RuntimeError(f'Unrecognized serving path: {relative}')
    receipt = {'metadata_before_sha256': hashlib.sha256(source.encode()).hexdigest(),
               'topk_v2_headers': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in headers},
               'allowed_index_topk': [512, 1024, 2048]}
    source = source.replace(old, new).replace(
        'supported: 512 (small) or 1024 (large)', 'supported: 512, 1024 or 2048 (fleet override)')
    # The pinned preview falls back to v1 whenever raw indices are requested.
    # v2 already supports raw output. Candidate filtering consumes its raw-only
    # output and writes final page/raw outputs in filter_topk_pages below.
    old_dispatch = '''        if metadata.use_topk_v2 and raw_indices is None:
            topk_transform_paged_v2(
                logits,
                metadata.c4_seq_lens,
                None if filter_candidates else metadata.page_table,
                selected if filter_candidates else page_indices,
                page_size,
                metadata.topk_metadata,
            )'''
    new_dispatch = old_dispatch.replace(
        'if metadata.use_topk_v2 and raw_indices is None:', 'if metadata.use_topk_v2:'
    ).replace('                metadata.topk_metadata,\n',
              '                metadata.topk_metadata,\n                None if filter_candidates else raw_indices,\n')
    if source.count(old_dispatch) != 1:
        raise RuntimeError('Unrecognized raw-index fallback; refusing a partial patch')
    source = source.replace(old_dispatch, new_dispatch)
    backend.write_text(source)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/sgl-workspace/sglang/python/sglang')
    parser.add_argument('--receipt', default='/opt/dsv41/topk-patch.json')
    args = parser.parse_args()
    receipt = patch(args.root)
    Path(args.receipt).write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))
