#!/usr/bin/env python3
"""Support mixed-K M16 through the existing M8 FC2 kernel; expose GB10 grouping."""
import argparse
import hashlib
from pathlib import Path

VERSION = 'glm53-r22-v19-1'
MIXED = 'moe/_shared/kernels/w4a16/mixed_trellis.py'
INPUTS = {MIXED: '80927912114bba31e03656fcc53815157bb41774be386bf613c75e49ecea14d1'}
OUTPUTS = {MIXED: 'e2e645f569c26b626900e1e15ee672becce82cd1587c7edb0d5e9b1cb20cce41'}

HELPER = '''def _gb10_fc2_schedule(moe_block_size: int) -> tuple[int, int]:
    # M16 needs the same supported wide-N M8 FC2 specialization as M32/M64.
    # The parent packed block still controls FC1, rotations and route packing.
    block = int(moe_block_size)
    if block not in (16, 32, 64):
        return block, 1
    factor = 2
    if torch.cuda.get_device_capability() == (12, 1):
        factor = int(os.environ.get("VLLM_GB10_EXL3_FC2_GROUP", "2"))
        if factor not in (1, 2, 4):
            raise ValueError("VLLM_GB10_EXL3_FC2_GROUP must be 1, 2 or 4")
    # A job may never cross its packed block's expert boundary. M16 has
    # only two M8 subtiles, so requesting four saturates at two.
    return 8, min(factor, block // 8)


'''


def transform(source):
    anchor = 'def compile_mixed_trellis(\n'
    if source.count(anchor) != 1:
        raise RuntimeError('unexpected mixed compiler anchor')
    source = source.replace(anchor, HELPER + anchor)
    replacements = (
        ('    grouped_m8_fc2 = int(moe_block_size) in (32, 64)\n',
         '    fc2_block, fc2_group = _gb10_fc2_schedule(moe_block_size)\n'),
        ('fc2_moe_block_size=(8 if grouped_m8_fc2 else moe_block_size),',
         'fc2_moe_block_size=fc2_block,'),
        ('fc2_schedule_route_block_factor=(2 if grouped_m8_fc2 else 1),',
         'fc2_schedule_route_block_factor=fc2_group,'),
    )
    for before, after in replacements:
        if source.count(before) != 2:
            raise RuntimeError(f'unexpected mixed compiler source: {before}')
        source = source.replace(before, after)
    return source


def patch(root, check=False):
    path = Path(root) / MIXED
    source = path.read_text(encoding='utf-8')
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest != OUTPUTS[MIXED]:
        if check or digest != INPUTS[MIXED]:
            raise RuntimeError(f'unexpected v19 source: {path} ({digest})')
        result = transform(source)
        compile(result, str(path), 'exec')
        if hashlib.sha256(result.encode()).hexdigest() != OUTPUTS[MIXED]:
            raise RuntimeError(f'v19 output hash mismatch: {path}')
        path.write_text(result, encoding='utf-8', newline='\n')
    print(f'{VERSION}: verified {root}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    patch(args.root, args.check)
