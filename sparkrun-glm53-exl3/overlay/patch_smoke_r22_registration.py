#!/usr/bin/env python3
"""Correct the R22 EXL3 registration assertion in the image smoke check."""

from __future__ import annotations

import argparse
from pathlib import Path


OLD = '    assert get_quantization_config("exl3").get_name() == "exl3"\n'
NEW = (
    '    exl3_config_cls = get_quantization_config("exl3")\n'
    '    assert exl3_config_cls.__name__ == "Exl3Config"\n'
    '    assert exl3_config_cls().get_name() == "exl3"\n'
)


def patch(path: Path, *, check: bool) -> None:
    source = path.read_text(encoding="utf-8")
    if NEW in source:
        assert OLD not in source, path
        return
    if check:
        raise AssertionError(f"R22 EXL3 registration smoke patch is absent: {path}")
    assert source.count(OLD) == 1, path
    path.write_text(source.replace(OLD, NEW), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    patch(args.path, check=args.check)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
