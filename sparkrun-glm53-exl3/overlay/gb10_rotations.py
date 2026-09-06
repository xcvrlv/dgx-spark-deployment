"""SM121 EXL3 H128 rotations with the original FP16 rounding boundaries.

Arithmetic follows exllamav3 704aefd's had_hf_r_128_inner (Apache-2.0).
Batch four independent warps per CTA when enough slabs cover the GPU.
"""
import os
from functools import lru_cache
import torch
import triton
import triton.language as tl

ENABLED = os.getenv("VLLM_GB10_DENSE_ROTATIONS", "0") == "1"


@lru_cache(None)
def _sm_count(device):
    props = torch.cuda.get_device_properties(device)
    return props.multi_processor_count if (props.major, props.minor) == (12, 1) else 0


def enabled(x):
    return ENABLED and x.is_cuda and _sm_count(x.device) > 0


def launch_slabs(total, sms):
    # Preserve SM coverage for tiny decode/MTP matrices. For larger work,
    # four warps per block avoid the original one-warp CTA occupancy limit.
    return 4 if total >= 4 * sms else 1


@triton.jit
def _rotate(X, S, Y, ROWS, COLS: tl.constexpr, PRE: tl.constexpr, SLABS: tl.constexpr):
    lane = tl.arange(0, 32)
    slab = tl.program_id(0) * SLABS + tl.arange(0, SLABS)
    valid = slab < ROWS * (COLS // 128)
    offset = slab[:, None] * 128 + lane[None, :] * 4
    scale_offset = (slab[:, None] % (COLS // 128)) * 128 + lane[None, :] * 4
    # Present four adjacent half values together for vectorized/coalesced I/O.
    component = tl.arange(0, 4)
    values = tl.load(X + offset[:, :, None] + component[None, None, :],
                     valid[:, None, None], 0).to(tl.float16).to(tl.float32)
    scales = tl.load(S + scale_offset[:, :, None] + component[None, None, :],
                     valid[:, None, None], 0).to(tl.float32)
    even, odd = tl.split(tl.reshape(values, (SLABS, 32, 2, 2)))
    v0, v2 = tl.split(even)
    v1, v3 = tl.split(odd)
    even, odd = tl.split(tl.reshape(scales, (SLABS, 32, 2, 2)))
    s0, s2 = tl.split(even)
    s1, s3 = tl.split(odd)
    if PRE:
        v0 = (v0 * s0).to(tl.float16).to(tl.float32)
        v1 = (v1 * s1).to(tl.float16).to(tl.float32)
        v2 = (v2 * s2).to(tl.float16).to(tl.float32)
        v3 = (v3 * s3).to(tl.float16).to(tl.float32)
    a0, a1, a2, a3 = v0 + v1, v0 - v1, v2 + v3, v2 - v3
    h0, h1, h2, h3 = a0 + a2, a1 + a3, a0 - a2, a1 - a3
    for step in tl.static_range(5):
        distance = 1 << step
        partner = tl.broadcast_to((lane ^ distance)[None, :], (SLABS, 32))
        p0 = tl.gather(h0, partner, 1)
        p1 = tl.gather(h1, partner, 1)
        p2 = tl.gather(h2, partner, 1)
        p3 = tl.gather(h3, partner, 1)
        negative = ((lane & distance) != 0)[None, :]
        h0 = tl.where(negative, -h0, h0) + p0
        h1 = tl.where(negative, -h1, h1) + p1
        h2 = tl.where(negative, -h2, h2) + p2
        h3 = tl.where(negative, -h3, h3) + p3
    h0 = (h0 * 0.088388347648).to(tl.float16).to(tl.float32)
    h1 = (h1 * 0.088388347648).to(tl.float16).to(tl.float32)
    h2 = (h2 * 0.088388347648).to(tl.float16).to(tl.float32)
    h3 = (h3 * 0.088388347648).to(tl.float16).to(tl.float32)
    if not PRE:
        h0 = (h0 * s0).to(tl.float16).to(tl.float32)
        h1 = (h1 * s1).to(tl.float16).to(tl.float32)
        h2 = (h2 * s2).to(tl.float16).to(tl.float32)
        h3 = (h3 * s3).to(tl.float16).to(tl.float32)
    values = tl.reshape(tl.join(tl.join(h0, h2), tl.join(h1, h3)), (SLABS, 32, 4))
    tl.store(Y + offset[:, :, None] + component[None, None, :], values, valid[:, None, None])


def rotate(x, scales, out, *, pre):
    if (x.ndim != 2 or x.shape != out.shape or x.shape[1] % 128
            or x.dtype not in (torch.float16, torch.bfloat16)
            or out.dtype not in (torch.float16, torch.bfloat16)
            or scales.dtype != torch.float16 or tuple(scales.shape) != (x.shape[1],)
            or not all(t.is_cuda and t.is_contiguous() and t.device == x.device for t in (x, scales, out))):
        raise ValueError("invalid GB10 H128 tensors")
    sms = _sm_count(x.device)
    if not sms:
        raise ValueError("GB10 rotations require SM121")
    rows, cols = x.shape
    total = rows * (cols // 128)
    if total:
        slabs = launch_slabs(total, sms)
        _rotate[(triton.cdiv(total, slabs),)](x, scales, out, rows, cols, pre, slabs,
                                            num_warps=slabs, enable_fp_fusion=False)
