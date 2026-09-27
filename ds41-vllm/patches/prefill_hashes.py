#!/usr/bin/env python3
"""Optional GLM v17 bounded hash-copy port; separate from the RoCE image.

JJ b40673c / b12x 9043b44 heads checked 2026-09-13; still absent upstream.
Karmic 1794dcf re-audited 2026-09-27 against KK 953a636, JJ 8e1f1e5,
and b12x d44247b; the bounded copy remains absent. See PREFILL-SPARK-AUDIT.md.
Pass the vllm package directory, not its repository root. --revert rolls back.
"""
import argparse
import hashlib
from pathlib import Path

SOURCE_SHA = 'a0932da63c38487a5fae401c9b43f627e2dcee8e975ec8d6427e00d2796c6239'
SOURCE_SHAS = {SOURCE_SHA,
               '720215b0508dbab462063b2a21d9f7ae1fdbf0bc3041f66ec9e19c6e2dd018d3'}
OLD = b'        new_block_hashes = block_hashes[num_cached_blocks:]\n'
NEW = b'        new_block_hashes = block_hashes[num_cached_blocks:num_full_blocks]\n'
RELATIVE = 'v1/core/block_pool.py'


def patch(root, *, revert=False, check=False):
    path = Path(root) / RELATIVE
    data = path.read_bytes()
    original = data.replace(NEW, OLD, 1) if data.count(NEW) == 1 else data
    if (hashlib.sha256(original).hexdigest() not in SOURCE_SHAS
            or original.count(OLD) != 1):
        raise RuntimeError(f'Unexpected upstream source: {path}; check newer upstream first')
    expected = original if revert else original.replace(OLD, NEW, 1)
    if check and data != expected:
        raise RuntimeError(f'Unexpected patch state: {path}')
    if not check and data != expected:
        path.write_bytes(expected)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package', type=Path)
    parser.add_argument('--revert', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    patch(args.package, revert=args.revert, check=args.check)
