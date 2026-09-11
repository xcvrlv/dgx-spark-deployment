#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""engram-fp4-disk: packed E2M1 (FP4) Engram rows for the MXFP4+FP4-Engram hybrid.

vLLM's disk-backed Engram table (``EngramConfig.table_path``, added by
``patch/engram-disk-table.patch``) stages fp8_e4m3fn[256] + ue8m0[8] rows —
264 bytes — and dequantizes float8_e4m3fn on the CPU. The fleet's checkpoint,
``DeepSeek-V4.1-Flash-MXFP4-FP4-Engram``, stores the same table as
``uint8[rows, 128]`` packed E2M1 (low nibble first) with the same-shaped E8M0
scale plane. Staged unchanged, the serve dies at weight load, in
``_stage_disk_shard`` -> ``build_row_file``:

    AssertionError: (torch.Size([96000564, 8]), 96000564, 128)

because ``build_row_file`` asserts ``scale.shape == (n, dim // block_size)``
against a 128-byte weight, where ``dim // 32`` is 4 and the scale plane is 8.
Even bypassed, the reader would decode packed nibbles as fp8 bytes and return
wrong rows with no exception.

This editor adds a ``packed_fp4`` path to the two files the disk-table patch
produces:

  build_row_file      accepts ``[n, dim // 2]`` uint8 and writes 136-byte rows:
                      128 packed weight bytes + 8 E8M0 scale bytes. One row is
                      still one contiguous read.
  DiskEngramTable     computes the fp4 stride and decodes the E2M1 nibbles
                      directly in ``gather``, through a 16-entry LUT. Every E2M1
                      value is exact in fp32 and every E8M0 scale is a power of
                      two, so value * scale is exact before the bf16 cast. The
                      decode is bitwise equal to expanding to E4M3 first — the
                      ds4.1 adapter's Triton expansion — and to the local
                      lookup kernel's inline decode. Values are preserved, not
                      requantized.
  _stage_disk_shard   detects fp4 from the staged weight plane (uint8 with
                      ``shape[1] * 2 == dim``) and passes the format to both.
                      It also logs ``[DS41_FP4_DISK] format=...`` so a cluster
                      boot records the format and row bytes per rank. The
                      resident loader keeps its loud shape assertion, so fp4
                      without ``table_path`` still fails visibly.

The row file layout, the global row addressing, the O_DIRECT path, the worker
pool and the ue8m0 decode are unchanged. The detection is per tensor, which is
exactly the hybrid's signature: U8 weight with ``shape[1] * 2 == head_dim``
plus an F8_E8M0 scale plane, block size 32.

Applies to an installed vLLM tree; pass the two file paths:

    python3 engram-fp4-disk.py \\
        <vllm>/models/deepseek_v4_1/common/engram_disk.py \\
        <vllm>/models/deepseek_v4_1/common/engram.py

The same disk edits apply to the standalone twin ``patch/engram_disk.py``, so
``tests/test_engram_disk_fp4.py`` can exercise the fp4 read path without a
device. Every substitution asserts its occurrence count, so a version drift
fails loudly. A second application fails the same way: the anchors are gone
after the first one.
"""
import sys


def sub(path, old, new):
    s = open(path).read()
    n = s.count(old)
    assert n == 1, f"{path}: expected 1 occurrence of {old.splitlines()[0]!r}, found {n}"
    open(path, "w").write(s.replace(old, new))
    print(f"  {path}: patched at {old.strip().splitlines()[0]!r}")


DISK = "models/deepseek_v4_1/common/engram_disk.py"
ENGRAM = "models/deepseek_v4_1/common/engram.py"

# The disk edits apply to the in-tree engram_disk.py and to the standalone
# twin patch/engram_disk.py, which are identical in these regions.
DISK_EDITS = [
    # 1. The 16-entry E2M1 LUT, after the module constants.
    (
        """FP8_BLOCK_SIZE = 32
SECTOR = 512
PAGE = 4096
_O_DIRECT = getattr(os, "O_DIRECT", 0)
""",
        """FP8_BLOCK_SIZE = 32
SECTOR = 512
PAGE = 4096
_O_DIRECT = getattr(os, "O_DIRECT", 0)


# ds41-fp4: the 16 E2M1 codes decode to [0,.5,1,1.5,2,3,4,6], signed. All
# exact in fp32, and every E8M0 scale is a power of two, so value * scale is
# exact before the bf16 cast. Bitwise equal to expanding to E4M3 first — the
# low nibble first, signed zero — and to the lookup kernel's inline decode.
def _e2m1_lut() -> torch.Tensor:
    codes = torch.arange(16, dtype=torch.uint8)
    mag = (codes & 7).to(torch.float32)
    parity = (codes & 1).to(torch.float32)
    vals = torch.where(
        mag < 4,
        mag * 0.5,
        (2.0 + parity) * torch.where(mag < 6, 1.0, 2.0),
    )
    return torch.where((codes & 8) != 0, -vals, vals)


_E2M1_LUT = _e2m1_lut()
""",
    ),
    # 2. build_row_file accepts packed E2M1 rows.
    (
        """def build_row_file(
    path: str,
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_size: int = FP8_BLOCK_SIZE,
    chunk_rows: int = 1 << 16,
) -> int:
    \"\"\"Write `weight`/`scale` to `path` as interleaved fixed-stride rows.

    `weight` is [n, dim] float8_e4m3fn, `scale` is [n, dim // block_size]
    float8_e8m0fnu. Returns the row stride in bytes.
    \"\"\"
    n, dim = weight.shape
    assert scale.shape == (n, dim // block_size), (scale.shape, n, dim)
    stride = row_stride(dim, block_size)
    w = weight.cpu().view(torch.uint8)
    s = scale.cpu().view(torch.uint8)
""",
        """def build_row_file(
    path: str,
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_size: int = FP8_BLOCK_SIZE,
    chunk_rows: int = 1 << 16,
    packed_fp4: bool = False,
) -> int:
    \"\"\"Write `weight`/`scale` to `path` as interleaved fixed-stride rows.

    `weight` is [n, dim] float8_e4m3fn, `scale` is [n, dim // block_size]
    float8_e8m0fnu. Returns the row stride in bytes.

    ds41-fp4: with `packed_fp4`, `weight` is [n, dim // 2] uint8 carrying two
    E2M1 nibbles per byte, low nibble first, and the row carries `dim // 2`
    weight bytes plus the same `dim // block_size` scales. This is not another
    quantization step: the packed bytes decode to exactly the values of the
    downloaded FP4 checkpoint.
    \"\"\"
    n, dim = weight.shape
    if packed_fp4 and weight.dtype != torch.uint8:
        raise ValueError(
            f"FP4 Engram needs packed uint8 rows, got {weight.dtype}"
        )
    if packed_fp4:
        dim = dim * 2
    assert scale.shape == (n, dim // block_size), (scale.shape, n, dim)
    stride = (
        dim // 2 + dim // block_size if packed_fp4 else row_stride(dim, block_size)
    )
    w = weight.cpu().view(torch.uint8)
    s = scale.cpu().view(torch.uint8)
""",
    ),
    # 3. DiskEngramTable carries the format and computes the fp4 stride.
    (
        """        row_count: int | None = None,
        block_size: int = FP8_BLOCK_SIZE,
        threads: int = 64,
        use_odirect: bool = True,
    ) -> None:
        self.dim = dim
        self.block_size = block_size
        self.stride = row_stride(dim, block_size)
        self.row_start = row_start
        self.n_scales = dim // block_size
""",
        """        row_count: int | None = None,
        block_size: int = FP8_BLOCK_SIZE,
        threads: int = 64,
        use_odirect: bool = True,
        packed_fp4: bool = False,
    ) -> None:
        self.dim = dim
        self.block_size = block_size
        self.packed_fp4 = packed_fp4
        self.stride = (
            dim // 2 + dim // block_size
            if packed_fp4
            else row_stride(dim, block_size)
        )
        self.row_start = row_start
        self.n_scales = dim // block_size
""",
    ),
    # 4. The fp4 decode in gather. Same unflatten/scale/reduction structure
    #    as the fp8 path; only the value decode differs.
    (
        """        buf = torch.frombuffer(dst, dtype=torch.uint8).view(n, stride)
        values = buf[:, :self.dim].view(torch.float8_e4m3fn).float()
        # ue8m0 is a power of two, so its byte IS the fp32 exponent field.
        # Decoded the way the kernel does it, including byte 0, which this
        # gives 0.0 and torch's float8_e8m0fnu cast gives 2**-127.
        exps = (buf[:, self.dim:].to(torch.int32) << 23).view(torch.float32)
        out = values.unflatten(-1, (self.n_scales, self.block_size))
        out = (out * exps.unsqueeze(-1)).flatten(-2).to(torch.bfloat16)
        return out.masked_fill(~owned.unsqueeze(-1), 0).view(*indices.shape, self.dim)
""",
        """        buf = torch.frombuffer(dst, dtype=torch.uint8).view(n, stride)
        # ue8m0 is a power of two, so its byte IS the fp32 exponent field.
        # Decoded the way the kernel does it, including byte 0, which this
        # gives 0.0 and torch's float8_e8m0fnu cast gives 2**-127.
        if self.packed_fp4:
            # ds41-fp4: two E2M1 nibbles per byte, low nibble first: dim col c
            # is packed byte c // 2, its low nibble when c is even and its
            # high nibble when c is odd. Decoded through the 16-entry LUT.
            # Every value is exact in fp32 and every scale is a power of two,
            # so value * scale is exact before the bf16 cast. Bitwise equal to
            # expanding to E4M3 first, which is what the ds4.1 adapter's
            # Triton kernel does on the GPU.
            packed = buf[:, : self.dim // 2].long()
            values = torch.stack(
                [_E2M1_LUT[packed & 15], _E2M1_LUT[packed >> 4]], dim=-1
            ).flatten(-2)
            exps = (buf[:, self.dim // 2 :].to(torch.int32) << 23).view(
                torch.float32
            )
        else:
            values = buf[:, :self.dim].view(torch.float8_e4m3fn).float()
            exps = (buf[:, self.dim:].to(torch.int32) << 23).view(torch.float32)
        out = values.unflatten(-1, (self.n_scales, self.block_size))
        out = (out * exps.unsqueeze(-1)).flatten(-2).to(torch.bfloat16)
        return out.masked_fill(~owned.unsqueeze(-1), 0).view(*indices.shape, self.dim)
""",
    ),
]

# The loader edit. Only the vLLM engram.py carries _stage_disk_shard; the twin
# does not, and no anchor below appears in it.
ENGRAM_EDITS = [
    (
        """        assert self.table_path is not None
        self._staged[field] = shard
        if len(self._staged) < 2:
            return
        build_row_file(
            self._row_file,
            self._staged["weight"],
            self._staged["scale"],
            self.block_size,
        )
        self._staged.clear()
        self._table = DiskEngramTable(
            self._row_file,
            self.dim,
            row_start=self.vocab_start_idx,
            row_count=self.part_num_embeddings,
            block_size=self.block_size,
            threads=self._disk_read_threads,
            use_odirect=self._disk_direct_io,
        )
""",
        """        assert self.table_path is not None
        # ds41-fp4: the MXFP4+FP4-Engram hybrid stores the weight plane as
        # uint8[rows, dim // 2] packed E2M1 (low nibble first) with the same
        # E8M0 scale plane. Detected here from the staged weight; the row file
        # then carries 136-byte rows and the reader decodes the nibbles
        # directly. Values are preserved, not requantized.
        weight = self._staged["weight"]
        packed_fp4 = (
            weight.dtype == torch.uint8
            and weight.dim() == 2
            and weight.shape[1] * 2 == self.dim
        )
        if packed_fp4 and self.dim % self.block_size:
            raise ValueError("FP4 Engram requires a block size dividing the head dim")
        build_row_file(
            self._row_file,
            weight,
            self._staged["scale"],
            self.block_size,
            packed_fp4=packed_fp4,
        )
        logger.info(
            "[DS41_FP4_DISK] format=%s weight_row_bytes=%d row_stride=%d "
            "rows=%d file=%s",
            "fp4" if packed_fp4 else "fp8",
            self.dim // 2 if packed_fp4 else self.dim,
            (self.dim // 2 + self.dim // self.block_size)
            if packed_fp4
            else (self.dim + self.dim // self.block_size),
            self.part_num_embeddings,
            self._row_file,
        )
        self._staged.clear()
        self._table = DiskEngramTable(
            self._row_file,
            self.dim,
            row_start=self.vocab_start_idx,
            row_count=self.part_num_embeddings,
            block_size=self.block_size,
            threads=self._disk_read_threads,
            use_odirect=self._disk_direct_io,
            packed_fp4=packed_fp4,
        )
""",
    ),
]


def main() -> int:
    files = sys.argv[1:]
    if not files:
        root = "/usr/local/lib/python3.12/dist-packages/vllm"
        files = [f"{root}/{DISK}", f"{root}/{ENGRAM}"]
    print("engram-fp4-disk: patching", *files)
    for path in files:
        s = open(path).read()
        if "_stage_disk_shard" in s:
            edits = ENGRAM_EDITS
            what = "loader"
        elif "class DiskEngramTable" in s:
            edits = DISK_EDITS
            what = "disk reader"
        else:
            raise SystemExit(
                f"{path}: matches no engram-fp4-disk edit set. Pass the two "
                "files the engram-disk-table.patch produced, or the standalone "
                "twin patch/engram_disk.py."
            )
        print(f"  {path}: {what} edits")
        for old, new in edits:
            sub(path, old, new)
    print("engram-fp4-disk: done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
