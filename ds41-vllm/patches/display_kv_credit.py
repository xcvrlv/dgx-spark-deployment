#!/usr/bin/env python3
"""Credit the display reserve to the profiled KV budget; checked against vllm 5bca5a5 on 2026-09-19.

num_blocks = available_memory // bytes_per_block, so a larger budget is what
produces more usable KV; swapping the backing alone frees ordinary RAM but adds no
capacity. This is the same decoupling upstream documents for
num_gpu_blocks_override, applied to the display reserve instead. Crediting here
means every consumer of the budget sees one effective capacity by construction:
the multimodal IPC reservation, the admission check, auto-fit and the per-worker
config builder. Revert this patch alone to restore the ordinary budget exactly.
"""
import argparse
import hashlib
from pathlib import Path

RELATIVE = 'v1/worker/gpu_worker.py'
SOURCE_SHA = '392133a37ce23bed46e2d898c7752a59be208ed302da43ac902cec68e127d360'
OLD = b'''        self.available_kv_cache_memory_bytes = (
            self.requested_memory
            - profile_result.non_kv_cache_memory
            - late_persistent_memory
            - cudagraph_memory_estimate_applied
        )
'''
NEW = OLD + b'''        # The display reserve is not part of requested_memory or of the profiled
        # ordinary budget, so it is credited here rather than through
        # gpu_memory_utilization. Zero unless the deployment enables it.
        self.available_kv_cache_memory_bytes += _ds41_display_kv_credit()
'''


def patch(root, *, check=False, revert=False):
    path = Path(root) / RELATIVE
    data = path.read_bytes()
    header = b'from vllm.v1.worker.ds41_display_kv import credit as _ds41_display_kv_credit\n'
    original = data.replace(header, b'', 1) if data.count(header) == 1 else data
    original = original.replace(NEW, OLD, 1) if original.count(NEW) == 1 else original
    if hashlib.sha256(original).hexdigest() != SOURCE_SHA or original.count(OLD) != 1:
        raise RuntimeError(f'Unexpected upstream source: {path}; re-audit before patching')
    expected = original if revert else original.replace(OLD, NEW, 1)
    if not revert:
        anchor = expected.index(OLD)  # insert the import at the module's first import
        first = expected.index(b'\nimport ') + 1
        expected = expected[:first] + header + expected[first:anchor] + expected[anchor:]
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
