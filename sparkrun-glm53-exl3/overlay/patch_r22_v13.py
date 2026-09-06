#!/usr/bin/env python3
"""Pinned v13 GB10 dense-rotation overlay for the v12 composition."""
import argparse
import hashlib
from pathlib import Path

VERSION = "glm53-r22-v13-1"
KERNEL = "moe/_shared/kernels/w4a16/kernel.py"
HELPER = "moe/_shared/kernels/w4a16/gb10_rotations.py"
EXL3 = "model_executor/layers/quantization/exl3.py"
INPUTS = {'moe/_shared/kernels/w4a16/kernel.py': '21ff268a1db2d56360727073ae4d9d28a61add0813de5ebdf27581991f8a4ea3', 'model_executor/layers/quantization/exl3.py': '084d03052e0b6a6598ab2b49eb09e3eecad4c34d2338bb18b28a87cc47098cb7'}
OUTPUTS = {'moe/_shared/kernels/w4a16/kernel.py': '4da593709d5d18b10e73773b5b2cd28bee33bbf33b92d0bb2fc90844d7344894', 'model_executor/layers/quantization/exl3.py': '4cb328921d0a0da25bde46f41d10c4a879d175167465fa518d3d4e7e6b1251dc', 'moe/_shared/kernels/w4a16/gb10_rotations.py': 'b659c3693895bafe5546f00ec4295bd889a27b892c80b849e1cef7b4fa8f3b2d'}


def replace(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"v13 source anchor mismatch: {old[:100]!r}")
    return text.replace(old, new, 1)


def kernel(text):
    start = text.index("def _run_trellis256_dense_current_device(")
    end = text.index("\n\ndef run_trellis256_dense(", start)
    part = text[start:end]
    part = replace(part, "    hadamard_128 = _resolve_exl3_hadamard_128(hadamard_128)\n",
                   "    from . import gb10_rotations\n"
                   "    fused_rotation = (hadamard_128 is None and trellis_bits == 6\n"
                   "                      and trellis_codebook == 'mcg' and gb10_rotations.enabled(x))\n"
                   "    hadamard_128 = _resolve_exl3_hadamard_128(hadamard_128)\n")
    begin = part.index("    if x.dtype == torch.float16:\n")
    finish = part.index("    props = torch.cuda.get_device_properties(x.device)\n", begin)
    legacy = part[begin:finish]
    part = part[:begin] + (
        "    if fused_rotation:\n"
        "        rotated_compute = _trellis_dense_buffer(\n"
        "            'rotated_compute', rotated_f16 if compute_dtype == torch.float16 else rotated_compute,\n"
        "            shape=(m, size_k), dtype=compute_dtype, device=x.device,\n"
        "        )\n"
        "        gb10_rotations.rotate(x, prepared_dense.suh, rotated_compute, pre=True)\n"
        "    else:\n" + "".join("    " + line if line.strip() else line for line in legacy.splitlines(keepends=True))
    ) + part[finish:]
    part = replace(part, "    if compute_dtype == torch.float16:\n        gemm_f16 = gemm_output\n",
                   "    if fused_rotation:\n"
                   "        gb10_rotations.rotate(gemm_output, prepared_dense.svh, output, pre=False)\n"
                   "        return output\n\n"
                   "    if compute_dtype == torch.float16:\n        gemm_f16 = gemm_output\n")
    return text[:start] + part + text[end:]


def exl3(text):
    text = replace(text, '    weight = _b12x_trellis_weight(trellis, suh, svh, x.dtype)\n',
                   "    from b12x.moe._shared.kernels.w4a16.gb10_rotations import enabled\n"
                   "    # The established online K6 matmul is FP16 even with BF16 model activations.\n"
                   "    weight = _b12x_trellis_weight(trellis, suh, svh,\n"
                   "                                  torch.float16 if enabled(x) else x.dtype)\n")
    start = text.index("def _b12x_trellis_linear(\n")
    end = text.index("\n\nclass Exl3Config", start)
    part = text[start:end]
    part = replace(part, "    gemm_output = torch.empty_like(output)\n",
                   "    from b12x.moe._shared.kernels.w4a16.gb10_rotations import enabled\n"
                   "    compute_dtype = torch.float16 if enabled(x) else x.dtype\n"
                   "    gemm_output = torch.empty_like(output, dtype=compute_dtype)\n")
    part = replace(part, "    rotated_f16 = torch.empty_like(x)\n",
                   "    rotated_f16 = torch.empty_like(x, dtype=compute_dtype)\n")
    text = text[:start] + part + text[end:]
    anchor = "        x_2d = x.reshape(-1, x.shape[-1]).to(torch.float16).contiguous()\n        output = _b12x_trellis_linear(\n"
    text = replace(text, anchor,
                   "        from b12x.moe._shared.kernels.w4a16.gb10_rotations import enabled\n"
                   "        if bias is None and enabled(x):\n"
                   "            x_2d = x.reshape(-1, x.shape[-1]).contiguous()\n"
                   '            logger.info_once("v13: GB10 fused online K6 rotations active; FP16 MMA retained")\n'
                   "        else:\n"
                   "            x_2d = x.reshape(-1, x.shape[-1]).to(torch.float16).contiguous()\n"
                   "        output = _b12x_trellis_linear(\n")
    return text


def patch(b12x_root, vllm_root, check=False):
    helper = Path(__file__).with_name("gb10_rotations.py").read_text(encoding="utf-8")
    if hashlib.sha256(helper.encode()).hexdigest() != OUTPUTS[HELPER]:
        raise RuntimeError("v13 helper hash mismatch")
    pending = []
    for root, name, transform in ((b12x_root, KERNEL, kernel), (vllm_root, EXL3, exl3), (b12x_root, HELPER, None)):
        path = Path(root) / name
        source = path.read_text(encoding="utf-8") if path.exists() else None
        digest = hashlib.sha256(source.encode()).hexdigest() if source is not None else None
        if digest == OUTPUTS[name]:
            continue
        if check or digest != INPUTS.get(name):
            raise RuntimeError(f"unexpected v13 source: {path} ({digest})")
        result = transform(source) if transform else helper
        compile(result, str(path), "exec")
        if hashlib.sha256(result.encode()).hexdigest() != OUTPUTS[name]:
            raise RuntimeError(f"v13 output mismatch: {path}")
        pending.append((path, result))
    for path, result in pending:
        path.write_text(result, encoding="utf-8", newline="\n")
    print(f"{VERSION}: verified {b12x_root} and {vllm_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("b12x_root", type=Path)
    parser.add_argument("vllm_root", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    patch(args.b12x_root, args.vllm_root, args.check)
