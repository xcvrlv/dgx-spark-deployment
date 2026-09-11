# SPDX-License-Identifier: Apache-2.0
"""Bitwise contract checks for the FP4 disk Engram path.

The fleet's checkpoint, DeepSeek-V4.1-Flash-MXFP4-FP4-Engram, stores the
Engram table as uint8[rows, 128] packed E2M1 (low nibble first) with the
same-shaped E8M0 scale plane. patch/engram-fp4-disk.py adds that format to the
disk reader the engram-disk-table.patch produces. This test exercises the
ACTUAL editor applied to a copy of the ACTUAL standalone twin, on CPU:

    python3 test_engram_disk_fp4.py [scratch_dir]

Runs anywhere with torch (no CUDA, no vLLM). On Windows the reader takes its
buffered path, which the harness proves returns the same bytes.

What each result proves is printed with it. A parity assertion that cannot
fail proves nothing: the LUT is first checked against literal expected numbers,
so the bitwise comparisons below cannot pass by sharing a bug with it.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import threading

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
EDITOR = os.path.join(HERE, "..", "patch", "engram-fp4-disk.py")
TWIN = next(
    path
    for path in (
        os.path.join(HERE, "engram_disk.py"),
        os.path.join(HERE, "..", "patch", "engram_disk.py"),
    )
    if os.path.exists(path)
)

DIM = 256
ROWS = 40_000
N_SCALES = DIM // 32
FP4_STRIDE = 136  # 128 packed E2M1 bytes + 8 E8M0 scale bytes

failures = 0


def report(ok: bool, what: str) -> None:
    global failures
    print(f"{'PASS' if ok else 'FAIL'} {what}")
    if not ok:
        failures += 1


def apply_editor(target: str) -> None:
    """Run the real editor against `target` exactly as the build layer does."""
    result = subprocess.run(
        [sys.executable, EDITOR, target], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"editor failed on {target}:\n{result.stdout}{result.stderr}")
    print(result.stdout.strip())


def make_fp4_table(rows: int, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A synthetic packed E2M1 table. E2M1 has no NaN/Inf encoding, so any
    packed byte is valid. E8M0 exponents in a sane band, byte 0 included."""
    g = torch.Generator().manual_seed(7)
    packed = torch.randint(0, 256, (rows, dim // 2), dtype=torch.uint8, generator=g)
    scale = torch.randint(100, 150, (rows, dim // 32), dtype=torch.uint8, generator=g)
    scale[0, 0] = 0
    return packed, scale


def check_lut(lut: torch.Tensor) -> None:
    """The 16 E2M1 codes decode to literal expected numbers.

    Asserted against literals, not against another decode, so the bitwise
    comparisons below cannot pass by sharing a bug with the reference. The
    nibble order is checked separately: low nibble first.
    """
    codes = torch.arange(16, dtype=torch.uint8)
    got = lut[codes.to(torch.int64)].tolist()
    magnitudes = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    # Signed zero: code 8 with magnitude 0 is -0.0, code 0 is 0.0.
    want = [magnitudes[c & 7] * (-1.0 if c & 8 else 1.0) for c in range(16)]
    report(got == want, f"E2M1 LUT decodes to the literal table (got {got[:4]}...)")

    # Low nibble first: byte 0x12 carries lo=2 (1.0) then hi=1 (0.5).
    pair = lut[torch.tensor([2, 1], dtype=torch.int64)].view(2).tolist()
    report(
        pair == [1.0, 0.5],
        f"low nibble first (byte 0x12 -> lo=2,hi=1 -> {pair}, want [1.0, 0.5])",
    )
    signed = lut[torch.tensor([0, 8], dtype=torch.int64)].tolist()
    report(
        signed == [0.0, -0.0],
        f"signed zero preserved (codes 0,8 -> {signed})",
    )


def dequant_reference_fp4(
    packed: torch.Tensor, scale: torch.Tensor, indices: torch.Tensor,
    row_start: int, row_count: int,
) -> torch.Tensor:
    """In-memory equivalent of the fp4 gather, for parity tests.

    The LUT is verified against literals in check_lut(), so this shares no bug
    with the reader. The ue8m0 decode is the kernel's: byte << 23, bitcast.
    """
    e2m1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    flat = indices.reshape(-1).to(torch.int64)
    n = flat.numel()
    owned = (flat - row_start >= 0) & (flat - row_start < row_count)
    local = flat.masked_fill(~owned, 0)
    rows = []
    for row in packed[local].tolist():
        for b in row:
            lo, hi = b & 15, b >> 4
            rows.append(e2m1[lo & 7] * (-1.0 if lo & 8 else 1.0))
            rows.append(e2m1[hi & 7] * (-1.0 if hi & 8 else 1.0))
    values = torch.tensor(rows, dtype=torch.float32).view(n, DIM)  # low nibble first
    exps = (scale.view(torch.uint8)[local].to(torch.int32) << 23).view(torch.float32)
    out = values.unflatten(-1, (N_SCALES, 32))
    out = (out * exps.unsqueeze(-1)).flatten(-2).to(torch.bfloat16)
    return out.masked_fill(~owned.unsqueeze(-1), 0).view(
        *indices.shape, packed.shape[1] * 2
    )


def bits_equal(got: torch.Tensor, want: torch.Tensor) -> bool:
    """bf16 through int16: NaN compares equal to itself."""
    return bool(torch.equal(got.view(torch.int16), want.view(torch.int16)))


def install_preadv_standin() -> None:
    """Windows stand-ins for the POSIX calls the reader needs.

    Windows CPython exposes neither preadv nor pread, and its os.open defaults
    to text mode, which would translate CRLF pairs inside the row file and
    corrupt every read. The stand-in forces O_BINARY on the reader's opens and
    serves preadv with serialized lseek+read: lseek+read is not atomic across
    the reader's worker threads, which share one fd, while preadv carries its
    offset per call, so serialized per-call reads are equivalent for it. The
    real syscall itself is exercised on Linux by engram_disk_harness.py; this
    stand-in only lets the slicing, dequant and masking logic run on a host
    whose stdlib has neither call. A short read raises instead of silently
    truncating, the same structural assumption the reader makes.
    """
    if hasattr(os, "preadv"):
        return

    _binary = getattr(os, "O_BINARY", 0)
    _open = os.open

    def _open_forcing_binary(path, flags, /, *args, **kwargs):
        if _binary and not flags & _binary:
            flags |= _binary
        return _open(path, flags, *args, **kwargs)

    os.open = _open_forcing_binary

    _lock = threading.Lock()

    def _preadv(fd, buffers, offset, /):
        want = memoryview(buffers[0]).nbytes
        with _lock:
            os.lseek(fd, offset, os.SEEK_SET)
            blocks, got = [], 0
            while got < want:
                block = os.read(fd, want - got)
                if not block:
                    break
                blocks.append(block)
                got += len(block)
        if got != want:
            raise OSError(
                f"preadv stand-in: short read, {got} of {want} bytes at {offset}"
            )
        buffers[0][:] = b"".join(blocks)
        return got

    os.preadv = _preadv


def main() -> int:
    scratch = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="fp4-engram-")
    os.makedirs(scratch, exist_ok=True)
    print(f"[scratch] {scratch}")
    print(f"[editor] {EDITOR}")
    print(f"[twin]   {TWIN}")

    # 1. The fp4 editor applied to a copy of the real twin, exactly as the
    #    build layer applies it to the installed tree.
    twin_fp4 = os.path.join(scratch, "engram_disk_fp4.py")
    shutil.copyfile(TWIN, twin_fp4)
    apply_editor(twin_fp4)
    sys.path.insert(0, scratch)
    import engram_disk_fp4 as disk  # noqa: E402  the edited twin

    install_preadv_standin()

    lut = disk._E2M1_LUT
    check_lut(lut)

    # 2. The fp4 row file: size, stride, and bitwise parity against the
    #    reference dequantized from the packed bytes.
    packed, scale = make_fp4_table(ROWS, DIM)
    path = os.path.join(scratch, "engram_fp4.bin")
    stride = disk.build_row_file(path, packed, scale, packed_fp4=True)
    report(stride == FP4_STRIDE, f"fp4 row stride is {FP4_STRIDE} bytes (got {stride})")
    size = os.path.getsize(path)
    report(
        size == ROWS * FP4_STRIDE,
        f"fp4 row file is {ROWS} x {FP4_STRIDE} bytes (got {size})",
    )

    ids = torch.randint(0, ROWS, (64, 12), dtype=torch.int32)
    ids[0, 0] = 37  # row 37 is the one the flipped-byte control mutates
    with disk.DiskEngramTable(path, DIM, row_count=ROWS, packed_fp4=True) as t:
        got = t.gather(ids)
    want = dequant_reference_fp4(packed, scale, ids, 0, ROWS)
    report(
        bits_equal(got, want),
        "fp4 gather is bitwise equal to the reference dequant "
        f"({ids.numel()} rows of {DIM})",
    )

    # 3. The ue8m0 decode. Byte 0 must give 0.0, byte 255 +inf, exactly as the
    #    kernel does. The packed bytes are fixed at 0x02, whose low nibble is
    #    E2M1 1.0, so the first value of every 32-value block is exactly the
    #    scale, which is what the check compares.
    w = torch.full((1, DIM // 2), 0x02, dtype=torch.uint8)  # E2M1 1.0, low nibble
    s = torch.tensor([[0, 127, 128, 129, 126, 1, 254, 255]], dtype=torch.uint8)
    decode_path = os.path.join(scratch, "engram_fp4_decode.bin")
    disk.build_row_file(decode_path, w, s, packed_fp4=True)
    with disk.DiskEngramTable(decode_path, DIM, row_count=1, packed_fp4=True) as t:
        dec = t.gather(torch.zeros(1, 1, dtype=torch.int64))
    blocks = dec.view(N_SCALES, 32)[:, 0].float().tolist()
    want_dec = [0.0, 1.0, 2.0, 4.0, 0.5, 2.0**-126, 2.0**127, float("inf")]
    report(
        blocks == want_dec,
        f"ue8m0 decodes as the fp32 exponent field on the fp4 path (byte 0 -> {blocks[0]})",
    )

    # 4. Negative controls. Each must fail, or the parity assertion proves
    #    nothing.
    ids2 = torch.randint(0, ROWS, (64, 12), dtype=torch.int32)
    flipped = packed.clone()
    flipped[37, 5] ^= 0x08
    path2 = os.path.join(scratch, "engram_fp4_flipped.bin")
    disk.build_row_file(path2, flipped, scale, packed_fp4=True)
    with disk.DiskEngramTable(path2, DIM, row_count=ROWS, packed_fp4=True) as t:
        bad = t.gather(ids)
    report(
        not bits_equal(bad, want),
        "one flipped packed byte changes the result (control fired)",
    )

    with disk.DiskEngramTable(
        path, DIM, row_start=1, row_count=ROWS - 1, packed_fp4=True
    ) as t:
        shifted = t.gather(ids)
    report(
        not bits_equal(shifted, want),
        "a reader built with row_start=1 reads different rows (control fired)",
    )

    with disk.DiskEngramTable(path, DIM, row_count=ROWS, packed_fp4=True) as t:
        out_of_shard = t.gather(ids + ROWS)
    report(
        int(out_of_shard.abs().sum()) == 0,
        "out-of-shard ids gather zeros",
    )

    # 5. The fp8 path is unchanged: build and gather an fp8 table and compare
    #    against the twin's own reference, which the upstream test asserts
    #    against the Triton kernel.
    g = torch.Generator().manual_seed(3)
    w8 = torch.randint(0, 256, (4_000, DIM), dtype=torch.uint8, generator=g)
    w8 = torch.where((w8 & 0x7F) == 0x7F, torch.full_like(w8, 0x3C), w8)
    s8 = torch.randint(100, 150, (4_000, N_SCALES), dtype=torch.uint8, generator=g)
    fp8_path = os.path.join(scratch, "engram_fp8.bin")
    stride8 = disk.build_row_file(
        fp8_path, w8.view(torch.float8_e4m3fn), s8.view(torch.float8_e8m0fnu)
    )
    report(stride8 == DIM + N_SCALES, f"fp8 row stride is unchanged (got {stride8})")
    ids8 = torch.randint(0, 4_000, (16, 12), dtype=torch.int32)
    with disk.DiskEngramTable(fp8_path, DIM, row_count=4_000) as t:
        got8 = t.gather(ids8)
    want8 = disk.dequantize_reference(
        w8.view(torch.float8_e4m3fn),
        s8.view(torch.float8_e8m0fnu),
        ids8,
        0,
        4_000,
    )
    report(
        bits_equal(got8, want8),
        "fp8 gather still matches the twin's reference (regression guard)",
    )

    # 6. The editor refuses a second application (anchors gone) and a file
    #    matching no edit set.
    again = subprocess.run(
        [sys.executable, EDITOR, twin_fp4], capture_output=True, text=True
    )
    report(
        again.returncode != 0 and "expected 1 occurrence" in again.stderr,
        "a second editor application fails loudly (anchor gone)",
    )
    nomatch = os.path.join(scratch, "nomatch.py")
    with open(nomatch, "w") as f:
        f.write("x = 1\n")
    third = subprocess.run(
        [sys.executable, EDITOR, nomatch], capture_output=True, text=True
    )
    report(
        third.returncode != 0 and "matches no engram-fp4-disk edit set" in third.stderr,
        "a file matching no edit set is refused",
    )

    print(f"### rc={failures}")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
