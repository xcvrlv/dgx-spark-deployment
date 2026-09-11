"""Exact E2M1 -> E4M3 expansion of batch-staged Engram rows, low nibble first.

The original E8M0 scale plane remains unchanged and is applied by B12x lookup.
No full table allocation or second quantization is involved.
"""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["count"])
def _expand(packed, expanded, count, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(packed + i, i < count, other=0).to(tl.uint32)
    lo, hi = value & 15, value >> 4
    lm, hm = lo & 7, hi & 7
    # E4M3 encodings of [0,.5,1,1.5,2,3,4,6], including signed zero.
    lb = tl.where(lm == 0, 0, tl.where(lm == 1, 48, 48 + 4 * lm))
    hb = tl.where(hm == 0, 0, tl.where(hm == 1, 48, 48 + 4 * hm))
    tl.store(expanded + 2 * i, lb | ((lo & 8) << 4), i < count)
    tl.store(expanded + 2 * i + 1, hb | ((hi & 8) << 4), i < count)


def expand_rows(packed, expanded, rows):
    if rows:
        _expand[(triton.cdiv(rows * 128, 256),)](
            packed, expanded.view(torch.uint8), rows * 128, BLOCK=256
        )
