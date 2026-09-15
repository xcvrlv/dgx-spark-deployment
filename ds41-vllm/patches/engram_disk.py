#!/usr/bin/env python3
"""Expose the DiskTable shard window the pinned vllm engram integration reads; checked against b12x 92cd380 (window unchanged since 3a8b879) on 2026-09-15.

vllm's _ensure_disk_table reads table.shard_start/shard_end/shard_rows, which
the b12x DiskTable keeps on self._cache (its own add_shard and require_complete
read the same); without these properties disk-backed Engram lookup fails with
AttributeError at preparation.
"""
import argparse
import hashlib
from pathlib import Path

RELATIVE = 'sequence/engram/_disk.py'
SOURCE_SHA = '08cbbe58f9d6dc9a52ba6c653a70e6110ecabc552515c23082a0226bb3914c08'
OLD = b'''    @property
    def prefetch_pending(self) -> bool:
        """No disk reads remain pending between synchronous lookups."""
        return False
'''
NEW = OLD + b'''
    @property
    def shard_start(self) -> int:
        """TP-shard row window this table stages, clamped to the table."""
        return self._cache.shard_start

    @property
    def shard_end(self) -> int:
        return self._cache.shard_end

    @property
    def shard_rows(self) -> int:
        return self._cache.shard_rows
'''


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
