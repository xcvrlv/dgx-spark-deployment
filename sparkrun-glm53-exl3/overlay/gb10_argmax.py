"""SM121 split-vocabulary greedy reduction; no changes to the LM-head GEMM."""
import os
from functools import lru_cache

import torch
import triton
import triton.language as tl

ENABLED = os.getenv("VLLM_GB10_DRAFT_ARGMAX", "0") == "1"


@lru_cache(None)
def _sm121(device):
    return torch.cuda.get_device_capability(device) == (12, 1)


def enabled(logits):
    return (ENABLED and logits.is_cuda and logits.ndim == 2
            and logits.is_contiguous() and logits.shape[0] > 0
            and logits.shape[1] > 0
            and logits.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and _sm121(logits.device))


@triton.jit
def _winner(values, indices, valid):
    # Match torch.max/argmax: first NaN wins, otherwise first maximum wins.
    nan = valid & (values != values)
    first_nan = tl.min(tl.where(nan, indices, 2147483647), 0)
    maximum = tl.max(tl.where(valid & ~nan, values, -float("inf")), 0)
    first_max = tl.min(tl.where(valid & (values == maximum), indices, 2147483647), 0)
    index = tl.where(first_nan != 2147483647, first_nan, first_max)
    value = tl.where(first_nan != 2147483647, float("nan"), maximum)
    return value, index


@triton.jit
def _partial(X, P, N: tl.constexpr, VALID: tl.constexpr,
             CHUNKS: tl.constexpr, BLOCK: tl.constexpr):
    row, chunk = tl.program_id(0), tl.program_id(1)
    cols = chunk * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(X + row * N + cols, cols < VALID, -float("inf")).to(tl.float32)
    value, index = _winner(values, cols, cols < VALID)
    offset = (row * CHUNKS + chunk) * 2
    tl.store(P + offset, value)
    tl.store(P + offset + 1, index.to(tl.float32))


@triton.jit
def _local_finish(P, Y, CHUNKS: tl.constexpr, START: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    values = tl.load(P + (row * CHUNKS + c) * 2, c < CHUNKS, -float("inf"))
    # Empty chunks carry INT_MAX rounded to 2**31 in float32. Exclude them
    # before converting back to int32, whose overflow is not portable.
    raw = tl.load(P + (row * CHUNKS + c) * 2 + 1, c < CHUNKS, 2147483648.0)
    valid = (c < CHUNKS) & (raw < 2147483648.0)
    indices = tl.where(valid, raw, 0).to(tl.int32)
    value, index = _winner(values, indices, valid)
    index = tl.where(index == 2147483647, 0, index)
    # One 16-byte packet per row uses RoCEnante's direct gather layout.
    tl.store(Y + row * 4, value)
    tl.store(Y + row * 4 + 1, (index + START).to(tl.float32))
    tl.store(Y + row * 4 + 2, 0.0)
    tl.store(Y + row * 4 + 3, 0.0)


@triton.jit
def _global_finish(P, Y, TP: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    ranks = tl.arange(0, BLOCK)
    values = tl.load(P + row * TP * 4 + ranks * 4, ranks < TP, -float("inf"))
    _, winner = _winner(values, ranks, ranks < TP)
    token = tl.load(P + row * TP * 4 + winner * 4 + 1).to(tl.int64)
    tl.store(Y + row, token)


def local_pair(logits, valid, start):
    rows, cols = logits.shape
    if not 0 <= valid <= cols or not 0 <= start or start + cols >= 2**24:
        raise ValueError("argmax shard dimensions exceed exact FP32 token-ID range")
    block = 512 if rows == 1 else 1024
    chunks = triton.cdiv(cols, block)
    partial = torch.empty((rows, chunks, 2), device=logits.device, dtype=torch.float32)
    pair = torch.empty((rows, 4), device=logits.device, dtype=torch.float32)
    _partial[(rows, chunks)](logits, partial, cols, valid, chunks, block, num_warps=4)
    _local_finish[(rows,)](partial, pair, chunks, start, triton.next_power_of_2(chunks), num_warps=4)
    return pair


def global_tokens(pairs, tp):
    if pairs.ndim != 2 or pairs.shape[1] != 4 * tp or not pairs.is_contiguous():
        raise ValueError("expected contiguous [rows, 4 * TP] gathered argmax packets")
    out = torch.empty(pairs.shape[0], device=pairs.device, dtype=torch.int64)
    _global_finish[(pairs.shape[0],)](pairs, out, tp, triton.next_power_of_2(tp), num_warps=4)
    return out
