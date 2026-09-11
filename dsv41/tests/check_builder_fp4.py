# SPDX-License-Identifier: Apache-2.0
"""End-to-end check of build_real_engram_table_fp4.py on a synthetic hybrid shard.

Creates a tiny safetensors shard with the hybrid's tensor signature (U8 packed
E2M1 weight plane + F8_E8M0 scale plane under the checkpoint's tensor names),
replaces head_sizes with a simple layout, runs the builder with --verify, and
checks the row file, the bitwise verify, and the exit codes. A checkpoint with
a different row count must fail the builder's row-count insurance.
"""
import importlib.util
import os
import sys
import tempfile

import torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
BUILDER = os.path.join(HERE, "build_real_engram_table_fp4.py")

spec = importlib.util.spec_from_file_location("builder", BUILDER)
builder = importlib.util.module_from_spec(spec)
sys.modules["builder"] = builder
spec.loader.exec_module(builder)

# head_sizes is replaced after exec_module: the module body defines it, and
# only main() calls it. A simple 12-bucket layout keeps the test small.
SIZES = [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
builder.head_sizes = lambda layer: list(SIZES)

ROWS = sum(SIZES)


def make_shard(directory: str, rows: int) -> str:
    g = torch.Generator().manual_seed(5)
    # Packed E2M1: every byte is a valid code.
    w = torch.randint(0, 256, (rows, 128), dtype=torch.uint8, generator=g)
    # E8M0 scales in a sane band.
    s = torch.randint(120, 134, (rows, 8), dtype=torch.uint8, generator=g)
    path = os.path.join(directory, "model-00047-of-00048.safetensors")
    save_file(
        {
            "layers.1.engram.embed.weight": w,
            "layers.1.engram.embed.scale": s.view(torch.float8_e8m0fnu),
        },
        path,
    )
    return path


def run(argv: list[str]) -> int:
    sys.argv = argv
    try:
        return builder.main()
    except SystemExit as err:
        return err.code if isinstance(err.code, int) else 1


def main() -> int:
    failures = 0
    scratch = tempfile.mkdtemp(prefix="fp4-builder-")
    shard = make_shard(scratch, ROWS)

    rc = run(
        [
            BUILDER,
            "--snapshot",
            scratch,
            "--out",
            os.path.join(scratch, "rows"),
            "--layer",
            "1",
            "--rank",
            "0",
            "--tp",
            "4",
            "--verify",
            "16",
        ]
    )
    path = os.path.join(scratch, "rows", "engram_fp4_L1_r0of4.bin")
    owned = builder.head_sizes(1)[: -(-len(builder.head_sizes(1)) // 4)]
    ok = os.path.exists(path) and os.path.getsize(path) == sum(owned) * builder.STRIDE
    print(
        f"{'PASS' if ok else 'FAIL'} row file is {sum(owned):,} x {builder.STRIDE} bytes"
        f" (got {os.path.getsize(path) if os.path.exists(path) else 0})"
    )
    failures += int(not ok)
    ok = rc == 0
    print(f"{'PASS' if ok else 'FAIL'} builder verified rows bitwise (rc={rc})")
    failures += int(not ok)

    # A checkpoint with a different row count must fail the row-count
    # insurance: the reader addresses rows globally, so a mismatch would
    # silently serve another rank's rows.
    bad = tempfile.mkdtemp(prefix="fp4-bad-")
    make_shard(bad, ROWS + 7)
    rc2 = run(
        [
            BUILDER,
            "--snapshot",
            bad,
            "--out",
            os.path.join(bad, "rows"),
            "--layer",
            "1",
            "--tp",
            "4",
        ]
    )
    ok2 = rc2 != 0
    print(
        f"{'PASS' if ok2 else 'FAIL'} a checkpoint with a different row count"
        f" is refused (rc={rc2})"
    )
    failures += int(not ok2)

    print(f"### rc={failures}")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
