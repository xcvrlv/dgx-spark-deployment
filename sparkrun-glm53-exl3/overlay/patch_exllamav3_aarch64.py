#!/usr/bin/env python3
"""Make ExLlamaV3's x86 CPU capability probes build on AArch64."""

from __future__ import annotations

import argparse
from pathlib import Path


MARKER = "// GLM53_AARCH64_CPU_PROBE_STUB"
PROBES = {
    "avx2_target.cpp": (
        "bool is_avx2_supported()",
        '__builtin_cpu_supports("avx2")',
    ),
    "avx512_target.cpp": (
        "bool is_avx512_supported()",
        '__builtin_cpu_supports("avx512f")',
    ),
}

PARALLEL_STUBS = {
    "all_reduce_cpu_avx2.cpp": """#include <cstddef>
#include "all_reduce_cpu_avx2.h"
#include <stdexcept>

namespace {
[[noreturn]] void unavailable_on_aarch64()
{
    throw std::runtime_error("ExLlamaV3 AVX2 CPU all-reduce is unavailable on AArch64");
}
}

void enable_fast_fp() {}
void enable_fast_fp_avx2() {}

void perform_cpu_reduce
(
    PGContext*, size_t, uint32_t, uint8_t*, size_t
)
{
    unavailable_on_aarch64();
}

void perform_cpu_reduce_avx2
(
    PGContext*, size_t, uint32_t, uint8_t*, size_t
)
{
    unavailable_on_aarch64();
}
""",
    "all_reduce_cpu_avx512.cpp": """#include "all_reduce_cpu_avx512.h"
#include <stdexcept>

namespace {
[[noreturn]] void unavailable_on_aarch64()
{
    throw std::runtime_error("ExLlamaV3 AVX-512 CPU all-reduce is unavailable on AArch64");
}
}

void enable_fast_fp_avx512() {}

void bf16_add_inplace_avx512
(
    uint16_t*, const uint16_t*, size_t
)
{
    unavailable_on_aarch64();
}

void perform_cpu_reduce_avx512
(
    PGContext*, size_t, uint32_t, uint8_t*, size_t
)
{
    unavailable_on_aarch64();
}
""",
}


def patched_source(source: str, signature: str) -> str:
    return (
        f"{MARKER}\n"
        "#if defined(__aarch64__)\n"
        f"{signature} {{ return false; }}\n"
        "#else\n"
        f"{source.rstrip()}\n"
        "#endif\n"
    )


def patched_parallel_source(source: str, stub: str) -> str:
    return (
        f"{MARKER}\n"
        "#if defined(__aarch64__)\n"
        f"{stub.rstrip()}\n"
        "#else\n"
        f"{source.rstrip()}\n"
        "#endif\n"
    )


def patch(root: Path, *, check: bool) -> None:
    source_dir = root / "exllamav3" / "exllamav3_ext"
    for name, (signature, builtin) in PROBES.items():
        path = source_dir / name
        source = path.read_text(encoding="utf-8")
        if MARKER in source:
            assert source.count(MARKER) == 1, path
            assert f"{signature} {{ return false; }}" in source, path
            assert source.count(builtin) >= 1, path
            continue
        if check:
            raise AssertionError(f"AArch64 probe patch is absent: {path}")
        assert source.count(signature) == 1, path
        assert source.count(builtin) >= 1, path
        path.write_text(patched_source(source, signature), encoding="utf-8")

    parallel_dir = source_dir / "parallel"
    for name, stub in PARALLEL_STUBS.items():
        path = parallel_dir / name
        source = path.read_text(encoding="utf-8")
        if MARKER in source:
            assert source.count(MARKER) == 1, path
            assert "unavailable_on_aarch64" in source, path
            assert source.count("#include <immintrin.h>") == 1, path
            continue
        if check:
            raise AssertionError(f"AArch64 all-reduce patch is absent: {path}")
        assert source.count("#include <immintrin.h>") == 1, path
        assert source.count("perform_cpu_reduce") >= 1, path
        path.write_text(patched_parallel_source(source, stub), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    patch(args.root, check=args.check)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
