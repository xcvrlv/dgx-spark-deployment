#!/usr/bin/env bash
# Fetch the upstream allocator used by the GB10 display-reserve technique.
#
# Why fetch instead of vendor: display_kv.c and libds41_display_kv.so are
# AGPL-3.0-only (c) the upstream author. They are kept as separate, separately
# licensed artifacts rather than copied into this repository. The probe in this
# directory is our own code and links against them.
#
# Both files are pinned by commit and verified by SHA-256 before use.
set -euo pipefail

REPO=https://github.com/coolbho3k/DeepSeek-v4.1-Flash-2x-DGX-Spark.git
COMMIT=878e0eecd893fadc69ad2d58b2df0fabb0fae2ee

SRC_REMOTE=release/runtime/sources/display_kv.c
SRC_SHA256=c689efeab383d33662b9b524e4d34bea5067f0232f85f0df4aceebd1b06a96a4

SO_REMOTE=release/runtime/serving/libds41_display_kv.so
SO_SHA256=bf60fcdf13126ed74363d2a7f0eff208a7318667d75cde323ba11b6b07f8556c

# Reference only: the runtime's PyTorch integration, useful to read before
# porting this into another serving stack. Not used by the probe build.
PY_REMOTE=release/runtime/serving/ds41/display_kv.py
PY_SHA256=81ae6b42cb400f5986b853e200eab101323ab992296f43659fc062db985d0c4f

HERE=$(cd "$(dirname "$0")" && pwd)
OUT="$HERE/upstream"
mkdir -p "$OUT"

raw() { printf 'https://raw.githubusercontent.com/coolbho3k/DeepSeek-v4.1-Flash-2x-DGX-Spark/%s/%s' "$COMMIT" "$1"; }

fetch_verified() {
    local remote=$1 out=$2 want=$3
    local got
    if command -v sha256sum >/dev/null 2>&1; then
        got=$(sha256sum "$out" 2>/dev/null | cut -d' ' -f1)
    else
        got=$(shasum -a 256 "$out" 2>/dev/null | cut -d' ' -f1)
    fi
    if [ "$got" = "$want" ]; then
        echo "  ok      $out  sha256=$got"
        return 0
    fi
    echo "  MISMATCH $out" >&2
    echo "    want $want" >&2
    echo "    got  ${got:-<none>}" >&2
    return 1
}

echo "Fetching pinned upstream artifacts (commit $COMMIT)"
echo "  license: AGPL-3.0-only — not vendored into this repository"

if [ ! -f "$OUT/display_kv.c" ] || ! fetch_verified "$SRC_REMOTE" "$OUT/display_kv.c" "$SRC_SHA256" 2>/dev/null; then
    rm -f "$OUT/display_kv.c"
    curl -fsSL "$(raw "$SRC_REMOTE")" -o "$OUT/display_kv.c"
    fetch_verified "$SRC_REMOTE" "$OUT/display_kv.c" "$SRC_SHA256"
fi

if [ ! -f "$OUT/libds41_display_kv.so" ] || ! fetch_verified "$SO_REMOTE" "$OUT/libds41_display_kv.so" "$SO_SHA256" 2>/dev/null; then
    rm -f "$OUT/libds41_display_kv.so"
    curl -fsSL "$(raw "$SO_REMOTE")" -o "$OUT/libds41_display_kv.so"
    fetch_verified "$SO_REMOTE" "$OUT/libds41_display_kv.so" "$SO_SHA256"
fi

if [ ! -f "$OUT/display_kv.py" ] || ! fetch_verified "$PY_REMOTE" "$OUT/display_kv.py" "$PY_SHA256" 2>/dev/null; then
    rm -f "$OUT/display_kv.py"
    curl -fsSL "$(raw "$PY_REMOTE")" -o "$OUT/display_kv.py"
    fetch_verified "$PY_REMOTE" "$OUT/display_kv.py" "$PY_SHA256"
fi

echo
echo "Prebuilt .so check (must be aarch64 ELF, since the Sparks are arm64):"
if command -v file >/dev/null 2>&1; then
    file "$OUT/libds41_display_kv.so"
else
    head -c 20 "$OUT/libds41_display_kv.so" | od -An -tx1 | head -2
fi

echo
echo "Wrote:"
ls -l "$OUT"
echo
echo "Read $OUT/display_kv.py and $OUT/display_kv.c before running anything."
