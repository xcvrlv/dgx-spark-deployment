#!/usr/bin/env python3
"""Reduce serving tuning budget to rounds=1, samples=4; checked against vllm c9dc4e5 on 2026-09-15."""
import argparse
import hashlib
from pathlib import Path

RELATIVE = 'model_executor/warmup/b12x_prepare.py'
SOURCE_SHA = '2213eb87da148aba3508547dcb25585b69b2d6d28da211befa0bc9c5487eecaf'
OLD = b'        compile_workers=16,\n    )'
NEW = b'        compile_workers=16,\n        rounds=1, samples=4,\n    )'


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
