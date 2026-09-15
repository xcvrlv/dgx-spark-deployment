#!/usr/bin/env python3
"""Fix RoCE preparation dtype names; checked against b12x 92cd380 (target unchanged since 9e90d60f) on 2026-09-15."""
import argparse
import hashlib
from pathlib import Path

RELATIVE='comm/roce/_preparation.py'
SOURCE_SHA='97b60edfae507b1e168e83c13c4178c3bfef9a8c7fbe1a1bb85f03d9435727b9'
OLD=b'programs = {dtype: get_launcher(dtype, *common) for dtype in _dtypes(query)}'
NEW=b'programs = {dtype: get_launcher(str(dtype).removeprefix("torch."), *common) for dtype in _dtypes(query)}'


def patch(root, *, check=False, revert=False):
    path=Path(root)/RELATIVE
    data=path.read_bytes()
    original=data.replace(NEW,OLD,1) if data.count(NEW)==1 else data
    if hashlib.sha256(original).hexdigest()!=SOURCE_SHA or original.count(OLD)!=1:
        raise RuntimeError(f'Unexpected upstream source: {path}; re-audit before patching')
    expected=original if revert else original.replace(OLD,NEW,1)
    if check and data!=expected:
        raise RuntimeError(f'Unexpected patch state: {path}')
    if not check and data!=expected:
        path.write_bytes(expected)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('package',type=Path)
    p.add_argument('--check',action='store_true')
    p.add_argument('--revert',action='store_true')
    a=p.parse_args()
    patch(a.package,check=a.check,revert=a.revert)
