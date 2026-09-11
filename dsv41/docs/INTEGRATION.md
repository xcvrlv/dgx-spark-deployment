# Fix 8: the disk reader does not accept the fleet's FP4 Engrams

## The failure

No arithmetic survives the first weight load. Serving the fleet's
`DeepSeek-V4.1-Flash-MXFP4-FP4-Engram` checkpoint on the upstream stack, the
disk-backed Engram loader (`_stage_disk_shard` -> `build_row_file`) dies in
its shape assertion:

```
AssertionError: (torch.Size([96000564, 8]), 96000564, 128)
```

`build_row_file` asserts `scale.shape == (n, dim // block_size)` against a
128-byte weight, where `dim // 32` is 4 and the scale plane is 8. The row
count is the rank-0 share of layer 1's table: 96,000,564 rows, derived from
the deterministic prime search (confirmed against the released config by the
toy self-test).

## The cause

The upstream stack's disk reader stages fp8_e4m3fn[256] + ue8m0[8] rows —
264 bytes. The fleet's checkpoint stores the same table as `uint8[rows, 128]`
packed E2M1 (low nibble first) with the same-shaped E8M0 scale plane, the
exact signature `engram_dtype=fp4`, block size 32, `ue8m0` from the
checkpoint's `quantization_config`. Setting `table_path` alone fails the
source-shape/dtype checks.

The hybrid's tensor signature, from `scripts/prepare-deepseek-v41-hybrid.py`:

| Tensor | dtype | shape[-1] |
|---|---|--:|
| `layers.{1,14}.engram.embed.weight` | `U8` (packed E2M1) | 128 |
| `layers.{1,14}.engram.embed.scale` | `F8_E8M0` | 8 |

The ds4.1 stack's own adapter (`ds4.1/patches/fp4_disk.py`) handles this
signature on the GPU: a Triton kernel expands packed E2M1 nibbles to E4M3
bytes, low nibble first, and the B12x lookup kernel applies the E8M0 scales.
The dsv41 stack has no GPU-side expand kernel — its disk reader dequantizes
on the CPU in `DiskEngramTable.gather` — so the equivalent here is a CPU-side
E2M1 decode.

## The change

`patch/engram-fp4-disk.py`, applied on top of the installed tree after the
engram-disk-table layer. It adds a `packed_fp4` path to the two files the
disk-table patch produces. Three pieces:

1. `build_row_file` accepts `[n, dim // 2]` uint8 and writes 136-byte rows:
   128 packed weight bytes + 8 E8M0 scale bytes. One row is still one
   contiguous read. The row layout, the global row addressing, the O_DIRECT
   path and the worker pool are unchanged.
2. `DiskEngramTable` computes the fp4 stride and decodes the E2M1 nibbles
   directly in `gather`, through a 16-entry LUT. Every E2M1 value is exact in
   fp32 and every E8M0 scale is a power of two, so value * scale is exact
   before the bf16 cast. The decode is bitwise equal to expanding to E4M3
   first — the ds4.1 adapter's Triton expansion — and to the local lookup
   kernel's inline decode (`(values.to(tl.float32) * scale).to(tl.bfloat16)`).
   Values are preserved, not requantized.
3. `_stage_disk_shard` detects fp4 from the staged weight plane (uint8 with
   `shape[1] * 2 == dim`) and passes the format to both. It also logs
   `[DS41_FP4_DISK] format=... weight_row_bytes=... row_stride=...`, so a
   cluster boot records the format and row bytes per rank. The resident
   loader keeps its loud shape assertion, so fp4 without `table_path` still
   fails visibly (`engram shard does not fit param`).

The detection is per tensor, which is exactly the hybrid's signature: U8
weight with `shape[1] * 2 == head_dim` plus an F8_E8M0 scale plane, block
size 32. This is not another quantization step; it preserves the values of
the downloaded FP4 checkpoint.

## The E2M1 nibble

Low nibble first. Each packed byte carries two E2M1 codes: the low nibble at
dim column `2k`, the high nibble at `2k + 1`. An earlier version of the
adapter decoded through `torch.cat([LUT[lo], LUT[hi]], dim=1)`, which puts
every low nibble in columns 0-127 — a different interleaving than the
checkpoint's. The bitwise gather comparison caught it: 768 of 768 rows
differed. The adapter now decodes with `torch.stack([lo, hi], dim=-1)
.flatten(-2)`, which interleaves per byte, matching the ds4.1 kernel's
`tl.store(expanded + 2 * i, ...)` / `+ 2 * i + 1, hb`.

## What the tests cover

`tests/test_engram_disk_fp4.py` applies the editor to a copy of the
standalone twin and runs on CPU. 14 checks, all passing:

| Check | Result |
|---|---|
| The E2M1 LUT against literal expected numbers | PASS |
| Low nibble first (byte 0x12 -> lo=2, hi=1) | PASS |
| Signed zero preserved (codes 0, 8) | PASS |
| fp4 row stride is 136 bytes | PASS |
| fp4 row file size | PASS |
| fp4 gather bitwise-equal to a reference dequant of the packed bytes | PASS |
| ue8m0 decodes as the fp32 exponent field on the fp4 path (byte 0 -> 0.0) | PASS |
| One flipped packed byte changes the result | PASS (control fired) |
| A reader built with `row_start=1` reads different rows | PASS (control fired) |
| Out-of-shard ids gather zeros | PASS |
| fp8 row stride is unchanged | PASS |
| fp8 gather still matches the twin's reference | PASS (regression guard) |
| A second editor application fails loudly | PASS (anchor gone) |
| A file matching no edit set is refused | PASS |

The editor's anchors were also verified against the real patched vLLM tree:
a fresh copy of the PR #56214 tree at `e47aa780`, `engram-disk-table.patch`
applied, then `patch/engram-fp4-disk.py` against the two produced files. All
five substitutions matched, the patched tree compiles, and an idempotent
reapply fails loudly.

## Not proven

State of the work. Nothing below was measured.

- **Full-model or multi-host correctness.** The CPU contract test and the
  anchor verification do not establish that the serve boots and answers.
  GPU kernels cannot be exercised on the Windows development host.
- **The Engram hash ids.** The reader was verified against the checkpoint,
  and the row files were verified against the checkpoint bytes. Neither
  checks that the model computes the right row ids. See
  [silent-corruption.md](silent-corruption.md).
- **Bitwise equality against the local ds4.1 GPU path.** The fp4 decode is
  bitwise equal to the expansion-plus-scale arithmetic by construction (all
  E2M1 values exact, all scales powers of two, fp32 multiply, bf16 cast),
  but no run compares the two paths on the same serve.
- **Cluster evidence.** A boot should record
  `[DS41_FP4_DISK] format=fp4 weight_row_bytes=128 row_stride=136` in all
  four ranks' logs; `VL41_ENGRAM_PRESTAGE_VERIFY=N` gives the bitwise
  staged-vs-in-forward proof. Neither has been run.
