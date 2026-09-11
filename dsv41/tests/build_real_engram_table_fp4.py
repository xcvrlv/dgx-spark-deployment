# SPDX-License-Identifier: Apache-2.0
"""Build one TP rank's real FP4 Engram row file from the MXFP4+FP4-Engram hybrid.

The fp4 counterpart of `build_real_engram_table.py`. The hybrid checkpoint
stores the Engram table as `uint8[rows, 128]` packed E2M1 (low nibble first)
plus the same-shaped block-32 E8M0 scale plane, so a row is 136 bytes rather
than the fp8 checkpoint's 264. At TP=4 a rank owns `cdiv(24, 4) = 6` of the 24
hash-head buckets per Engram layer, so its share is about 12.2 GiB per layer
and 24.3 GiB for both.

This builder streams both slices in chunks rather than loading tens of GiB of
tensor, validates the checkpoint's row count against the config-derived head
layout, and can verify the written rows against the checkpoint bytes bitwise
at computed offsets. The serve builds its own row files at weight load
(`_stage_disk_shard` -> `build_row_file`); this one exists for the cold-read
harness, for rank-offset verification before a serve, and for the bitwise
file check.

Layer 1's Engram lives entirely in model-00047-of-00048 and layer 14's in
model-00048-of-00048, so each layer reads exactly one file.

`head_sizes` imports `EngramLayout` from DeepSeek's reference inference tree:
run this from a directory holding `inference/engram.py` (as
`engram-selftest.sh` stages one), or with PYTHONPATH pointing at one.

    python3 build_real_engram_table_fp4.py --out /big/engram --layer 1 --rank 0
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
for path in (HERE, os.path.join(HERE, "..", "patch")):
    if os.path.isdir(path):
        sys.path.insert(0, path)

from engram_disk import FP8_BLOCK_SIZE  # noqa: E402  the standalone twin

# The released config. The toy self-test reproduces `engram_num_embeddings`
# exactly from these, so the prime search is confirmed against the real thing.
VOCAB = 16_000_000
LAYERS = (1, 14)
NGRAM, HEADS, HEAD_DIM = 4, 8, 256
SHARD = {1: "model-00047-of-00048.safetensors", 14: "model-00048-of-00048.safetensors"}
WEIGHT_BYTES = HEAD_DIM // 2
STRIDE = WEIGHT_BYTES + HEAD_DIM // FP8_BLOCK_SIZE


def head_sizes(layer: int) -> list[int]:
    """The 24 prime bucket sizes for one layer, in checkpoint (ngram, head) order."""
    from engram import EngramLayout

    class _Args:
        engram_layer_ids = LAYERS
        engram_max_ngram_size = NGRAM
        engram_n_heads = HEADS
        engram_head_dim = HEAD_DIM
        engram_vocab_size = VOCAB
        engram_num_embeddings = ()

    layout = EngramLayout.from_args(_Args())
    per_ngram = layout.primes[LAYERS.index(layer)]
    return [p for order in per_ngram for p in order]


def seam_rows(sizes: list[int]) -> set[int]:
    """First and last row of each owned bucket, in file-relative row ids.

    Random interior rows do not catch an offset error; these do. The README's
    18-row check was 14 seams per shard for exactly this reason. The range's
    own ends are the first bucket's first row and the last bucket's last.
    """
    rows: set[int] = set()
    offset = 0
    for size in sizes:
        rows.add(offset)
        rows.add(offset + size - 1)
        offset += size
    return rows


def verify_rows(
    shard: str, layer: int, path: str, start: int, sizes: list[int], count: int
) -> int:
    """Read `count` rows of the written file back and compare against the
    checkpoint, bitwise, at computed offsets.

    Seam rows are forced in; a stride sample fills the rest of the count. The
    file's row `p` is checkpoint row `start + p`.
    """
    rows = sum(sizes)
    picks = set(seam_rows(sizes))
    if len(picks) < count:
        step = max(rows // count, 1)
        picks |= {i * step for i in range(count)}
    picks = sorted(p for p in picks if 0 <= p < rows)[:count]

    failures = 0
    with safe_open(shard, framework="pt") as f, open(path, "rb") as out:
        w = f.get_slice(f"layers.{layer}.engram.embed.weight")
        s = f.get_slice(f"layers.{layer}.engram.embed.scale")
        for p in picks:
            out.seek(p * STRIDE)
            row = out.read(STRIDE)
            if len(row) != STRIDE:
                print(f"  row {p:,}: short read, {len(row)} of {STRIDE} bytes")
                failures += 1
                continue
            wt = torch.frombuffer(row[:WEIGHT_BYTES], dtype=torch.uint8)
            st = torch.frombuffer(row[WEIGHT_BYTES:], dtype=torch.uint8)
            ref_w = w[start + p : start + p + 1]
            ref_s = s[start + p : start + p + 1]
            if ref_w.dtype != torch.uint8:
                ref_w = ref_w.view(torch.uint8)
            if ref_s.dtype != torch.uint8:
                ref_s = ref_s.view(torch.uint8)
            if not torch.equal(wt, ref_w.reshape(-1)) or not torch.equal(
                st, ref_s.reshape(-1)
            ):
                print(f"  row {p:,}: differs from the checkpoint")
                failures += 1
    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--snapshot",
        default=os.path.expanduser(
            "~/.cache/huggingface/hub/DeepSeek-V4.1-Flash-MXFP4-FP4-Engram"
        ),
        help="the hybrid checkpoint directory (plain path, 48-shard layout)",
    )
    ap.add_argument("--out", required=True, help="directory for the row file")
    ap.add_argument("--layer", type=int, default=1, choices=LAYERS)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=1 << 20, help="rows per read")
    ap.add_argument(
        "--verify",
        type=int,
        default=0,
        help="bitwise-check N rows of the file against the checkpoint",
    )
    a = ap.parse_args()

    shard = os.path.join(a.snapshot, SHARD[a.layer])
    if not os.path.exists(shard):
        raise SystemExit(f"missing checkpoint shard: {shard}")
    print(f"[src] {shard}")

    sizes = head_sizes(a.layer)
    per_rank_cols = -(-len(sizes) // a.tp)
    lo_head = a.rank * per_rank_cols
    hi_head = min(lo_head + per_rank_cols, len(sizes))
    start = sum(sizes[:lo_head])
    rows = sum(sizes[lo_head:hi_head])
    print(
        f"[shard] layer {a.layer} rank {a.rank}/{a.tp}: heads {lo_head}..{hi_head - 1}, "
        f"rows {start:,}..{start + rows:,} ({rows:,} rows, {rows * STRIDE / 2**30:.1f} GiB)"
    )

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"engram_fp4_L{a.layer}_r{a.rank}of{a.tp}.bin")
    n_scales = HEAD_DIM // FP8_BLOCK_SIZE

    t0 = time.time()
    written = 0
    with safe_open(shard, framework="pt") as f:
        w = f.get_slice(f"layers.{a.layer}.engram.embed.weight")
        s = f.get_slice(f"layers.{a.layer}.engram.embed.scale")
        total = w.get_shape()[0]
        # The hybrid's packed FP4 plane: uint8 rows with head_dim // 2 bytes.
        if w.get_shape()[1] != WEIGHT_BYTES:
            raise SystemExit(
                f"expected the packed FP4 Engram plane ({WEIGHT_BYTES} columns), "
                f"got {w.get_shape()}"
            )
        if s.get_shape()[1] != n_scales:
            raise SystemExit(
                f"expected the block-32 E8M0 scale plane ({n_scales} columns), "
                f"got {s.get_shape()}"
            )
        # Insurance against a different row structure: the reader addresses
        # rows globally, so a checkpoint whose rows did not match the
        # config-derived layout would silently serve another rank's rows.
        if total != sum(sizes):
            raise SystemExit(
                f"checkpoint row count {total:,} differs from the head layout "
                f"{sum(sizes):,}; the hash ids and the row file would disagree"
            )
        assert start + rows <= total, (start, rows, total)
        with open(path, "wb", buffering=1 << 22) as out:
            for lo in range(start, start + rows, a.chunk):
                hi = min(lo + a.chunk, start + rows)
                wb = w[lo:hi]
                if wb.dtype != torch.uint8:
                    wb = wb.view(torch.uint8)
                sb = s[lo:hi].view(torch.uint8)
                assert wb.shape[1] == WEIGHT_BYTES, wb.shape
                assert sb.shape[1] == n_scales, sb.shape
                out.write(torch.cat([wb, sb], dim=1).contiguous().numpy().tobytes())
                written += hi - lo
                if written % (32 * a.chunk) == 0 or hi == start + rows:
                    el = time.time() - t0
                    gib = written * STRIDE / 2**30
                    print(
                        f"  {written:,}/{rows:,} rows  {gib:.1f} GiB  "
                        f"{gib / max(el, 1e-9):.2f} GiB/s",
                        flush=True,
                    )

    size = os.path.getsize(path)
    assert size == rows * STRIDE, (size, rows * STRIDE)
    print(f"[done] {path} {size / 2**30:.1f} GiB in {time.time() - t0:.0f}s")
    print(
        f"[use]  DiskEngramTable(path, {HEAD_DIM}, row_start={start}, "
        f"row_count={rows}, packed_fp4=True)"
    )

    if a.verify:
        failures = verify_rows(
            shard, a.layer, path, start, sizes[lo_head:hi_head], a.verify
        )
        if failures:
            print(f"[verify] {failures} rows differ from the checkpoint")
            return 1
        print(f"[verify] {a.verify} rows bitwise-equal to the checkpoint")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
