#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Hash-guarded native b12x barrier-fill optimization inspired by knapcio."""
import argparse
import hashlib
from pathlib import Path

RELATIVE = 'moe/fused_moe/_impl.py'
HELPER = 'moe/fused_moe/ds41_barriers.py'
SOURCE_SHA256 = '41bf14ea26f8f736670ade8df536d663b3b439870dc87a6268c50403c04afdc1'
IMPORT = b'import torch.nn.functional as F\n'
IMPORTED = IMPORT + b'from .ds41_barriers import clear_barriers as _ds41_clear_barriers\n'
OLD = b'        barrier_count.zero_()\n        barrier_epoch.zero_()\n'
NEW = b'        _ds41_clear_barriers(barrier_count, barrier_epoch)\n'


def transform(data, reverse=False):
    for old, new, count in ((IMPORT, IMPORTED, 1), (OLD, NEW, 2)):
        before, after = (new, old) if reverse else (old, new)
        if data.count(before) != count:
            raise RuntimeError('Native MoE barrier anchors drifted; re-audit upstream')
        data = data.replace(before, after)
    return data


def patch(b12x, *, check=False, revert=False):
    root = Path(b12x)
    path, helper = root/RELATIVE, root/HELPER
    runtime = Path(__file__).with_name('knapcio_decode_runtime.py').read_bytes().replace(b'\r\n', b'\n')
    if helper.exists() and helper.read_bytes() != runtime:
        raise RuntimeError('Unexpected native barrier helper')
    data = path.read_bytes()
    original = data if hashlib.sha256(data).hexdigest() == SOURCE_SHA256 else transform(data, True)
    if hashlib.sha256(original).hexdigest() != SOURCE_SHA256:
        raise RuntimeError('Unexpected native MoE source; re-audit upstream')
    expected = original if revert else transform(original)
    compile(expected, str(path), 'exec')
    if check:
        if data != expected or helper.exists() == revert:
            raise RuntimeError('Unexpected native MoE patch state')
    else:
        path.write_bytes(expected)
        if revert:
            helper.unlink(missing_ok=True)
        else:
            helper.write_bytes(runtime)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('b12x', type=Path)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--revert', action='store_true')
    args = parser.parse_args()
    patch(args.b12x, check=args.check, revert=args.revert)
