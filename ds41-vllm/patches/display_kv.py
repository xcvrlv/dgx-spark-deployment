#!/usr/bin/env python3
"""Back KV tensors with the GB10 display reserve; checked against vllm 5bca5a5 on 2026-09-19.

allocate_kv_cache already asserts every KV cache tensor shares one backing
allocation, which is what one contiguous display span provides. The import is lazy
so a disabled deployment never loads the allocator. It fails closed: at 0.85
utilization a silent torch.zeros fallback would allocate the credited block count
out of ordinary RAM and OOM later, which is worse than failing at startup.
"""
import argparse
import hashlib
from pathlib import Path

RELATIVE = 'v1/worker/utils.py'
SOURCE_SHA = '066555c3f4a5a9df55d2eaea22db40e3c98523a2279964649251537c0a923a4f'
OLD = b'    buf = torch.zeros(sizes.pop(), dtype=torch.int8, device=device)\n'
NEW = (
    b'    from vllm.v1.worker.ds41_display_kv import backing as _ds41_backing\n'
    b'\n'
    b'    buf = _ds41_backing(sizes.pop(), dtype=torch.int8, device=device)\n'
)


def patch(root, *, check=False, revert=False):
    path = Path(root) / RELATIVE
    data = path.read_bytes()
    original = data.replace(NEW, OLD, 1) if data.count(NEW) == 1 else data
    if hashlib.sha256(original).hexdigest() != SOURCE_SHA or original.count(OLD) != 1:
        raise RuntimeError(f'Unexpected upstream source: {path}; re-audit before patching')
    expected = original if revert else original.replace(OLD, NEW, 1)
    if check and data != expected:
        raise RuntimeError(f'Unexpected patch state: {path}')
    if not check and data != expected:
        path.write_bytes(expected)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('package', type=Path)
    p.add_argument('--check', action='store_true')
    p.add_argument('--revert', action='store_true')
    a = p.parse_args()
    patch(a.package, check=a.check, revert=a.revert)
