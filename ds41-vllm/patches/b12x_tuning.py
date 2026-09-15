#!/usr/bin/env python3
"""Reduce serving tuning budget: compile_workers=8, rounds=1, samples=4, race_batch=8, race_budget=4GiB; checked against vllm 5bca5a5 on 2026-09-15.

compile_workers=8 halves the compile pool's anonymous host RAM peak (the
compile processes are the main MemAvailable consumer during compilation);
race_budget caps the resident candidate memory per race batch (b12x default is
half of free GPU memory); race_batch bounds candidates per batch (default 32).
"""
import argparse
import hashlib
from pathlib import Path

RELATIVE = 'model_executor/warmup/b12x_prepare.py'
SOURCE_SHA = '9521a4f855cb1a1b7c752008774aa8ee7c01c91a57ae5f4c2efaedd2d71a85fb'
OLD = b'        compile_workers=16,\n    )'
NEW = b'        compile_workers=8,\n        rounds=1, samples=4, race_batch=8, race_budget=4 * (1 << 30),\n    )'
PRIOR = (
    b'        compile_workers=16,\n        rounds=1, samples=4, race_batch=8, race_budget=4 * (1 << 30),\n    )',
    b'        compile_workers=16,\n        rounds=1, samples=4,\n    )',
)


def patch(root, *, check=False, revert=False):
    path = Path(root) / RELATIVE
    data = path.read_bytes()
    original = data
    for applied in (NEW,) + PRIOR:
        if original.count(applied) == 1:
            original = original.replace(applied, OLD, 1)
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
