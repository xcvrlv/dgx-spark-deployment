#!/usr/bin/env python3
"""Retain RoCE launcher program metadata; upstream checked 2026-09-14."""
import argparse
import hashlib
from pathlib import Path

SOURCES = {
    '_oneshot_cute.py': 'e796a354c363be8153b9112f5b46b6249c7762e0293c2e3d8541ba036bcf3802',
    '_allgather_cute.py': '1453250675988ffbaf009b1cf8ff97b960e5667abc9791d42d92a7ede294428a',
}
OLD = b'    return run\n'
NEW = b'    from b12x._lib.compile_plan import attach_programs\n    return attach_programs(run, raw)\n'


def patch(root, *, check=False, revert=False):
    changes = []
    for name, sha in SOURCES.items():
        path = Path(root) / 'comm/roce' / name
        data = path.read_bytes()
        original = data.replace(NEW, OLD, 1) if data.count(NEW) == 1 else data
        if hashlib.sha256(original).hexdigest() != sha or original.count(OLD) != 1:
            raise RuntimeError(f'Unexpected upstream source: {path}; re-audit before patching')
        expected = original if revert else original.replace(OLD, NEW, 1)
        if check and data != expected:
            raise RuntimeError(f'Unexpected patch state: {path}')
        changes.append((path, data, expected))
    for path, data, expected in changes:
        if not check and data != expected:
            path.write_bytes(expected)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('package', type=Path)
    p.add_argument('--check', action='store_true')
    p.add_argument('--revert', action='store_true')
    a = p.parse_args()
    patch(a.package, check=a.check, revert=a.revert)
