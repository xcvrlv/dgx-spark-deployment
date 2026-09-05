#!/usr/bin/env python3
"""Pinned, late B12X overlay: mixed-K output fusion and RoCEnante PR315.

The MoE arithmetic and packed weights are unchanged. Only the final store
narrows to the activation dtype; the router reduction still accumulates FP32.
"""
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

VERSION = "glm53-r22-v11-1"
INPUT_HASHES = {'moe/_shared/kernels/w4a16/kernel.py': '591d06f211229703fc465f745db87241675d4b70fbcbb7af96c3d203642567c8',
 'moe/_shared/kernels/w4a16/mixed_trellis.py': '4d5140de84e3dde5875ff75ad95c9a62b6041b5dde2e69f0fb96562319aabd78',
 'comm/roce/_allgather_cute.py': 'a6dd608f01d3bf837e1bcb517f7e925b19cda533590263d17bb22feccd1572a4',
 'comm/roce/roce_oneshot.py': 'a555e83c0ae6ac99d27c14706f92966d4f5fbd213d682cbd3819cd174926288e'}
OUTPUT_HASHES = {'moe/_shared/kernels/w4a16/kernel.py': '21ff268a1db2d56360727073ae4d9d28a61add0813de5ebdf27581991f8a4ea3',
 'moe/_shared/kernels/w4a16/mixed_trellis.py': '80927912114bba31e03656fcc53815157bb41774be386bf613c75e49ecea14d1',
 'comm/roce/_allgather_cute.py': '1453250675988ffbaf009b1cf8ff97b960e5667abc9791d42d92a7ede294428a',
 'comm/roce/roce_oneshot.py': '92a32ed7ad570f54228998da10dc8dd463a042458c2bbd6beec521d1668344d4'}


def replace(text, old, new, count=1):
    if text.count(old) != count:
        raise RuntimeError(f"source anchor mismatch: {old[:100]!r}")
    return text.replace(old, new)


def kernel(text):
    start = text.index("class W4A16TopKSumCompileResult:")
    end = text.index("\n\n@dataclass", start)
    text = text[:end] + '\n    output_element_dtype: str = "fp32"\n' + text[end:]
    start = text.index("def compile_w4a16_topk_sum(")
    end = text.index("\n\ndef ", start)
    part = text[start:end]
    part = replace(part, '    element_dtype: str = "bf16",\n',
                   '    element_dtype: str = "bf16",\n    output_element_dtype: str | None = None,\n')
    part = replace(part, "    cutlass_dtype = _cutlass_element_dtype(element_dtype)\n",
                   '    output_element_dtype = output_element_dtype or ("fp32" if full_rotation else element_dtype)\n'
                   '    if output_element_dtype not in {"fp32", "fp16", "bf16"}:\n'
                   '        raise ValueError("invalid top-k output dtype")\n'
                   '    if not full_rotation and output_element_dtype != element_dtype:\n'
                   '        raise ValueError("output override requires full rotation")\n'
                   '    cutlass_dtype = _cutlass_element_dtype(element_dtype)\n')
    part = replace(part, '        "w4a16_topk_sum",\n', '        "w4a16_topk_sum",\n        output_element_dtype,\n')
    part = replace(part, "    output_dtype = cutlass.Float32 if full_rotation else cutlass_dtype\n",
                   '    output_dtype = (cutlass.Float32 if output_element_dtype == "fp32"\n'
                   '                    else _cutlass_element_dtype(output_element_dtype))\n')
    part = replace(part, '            "moe.w4a16.topk_sum",\n            3,',
                   '            "moe.w4a16.topk_sum",\n            4,')
    part = replace(part, "    result = W4A16TopKSumCompileResult(\n",
                   "    result = W4A16TopKSumCompileResult(\n        output_element_dtype=output_element_dtype,\n")
    text = text[:start] + part + text[end:]
    start = text.index("class W4A16TopKSumKernel:")
    end = text.index("\n_CACHE:", start)
    part = text[start:end]
    # Make conversion explicit at each full-rotation store, after all FP32 math.
    part, count = re.subn(r"(output_flat\[out_base[^\n]+\] = )((?:acc|o)\d)\n",
                          r"\1\2.to(output_flat.element_type)\n", part)
    if count != 12:
        raise RuntimeError(f"expected 12 final stores, got {count}")
    return text[:start] + part + text[end:]


def mixed(text):
    text = replace(text, "from dataclasses import dataclass, replace\n",
                   "from dataclasses import dataclass, replace\nimport os\n")
    text = replace(text, '        element_dtype="fp16",\n        full_rotation=True,\n',
                   '        element_dtype="fp16",\n'
                   '        output_element_dtype=(rotation_input_dtype\n'
                   '            if os.environ.get("VLLM_EXL3_MIXED_FUSED_OUTPUT", "0") == "1"\n'
                   '            else "fp32"),\n'
                   '        full_rotation=True,\n', 2)
    text = replace(text, "(launch.size_m, launch.hidden_size), dtype=torch.float32, device=device",
                   '(launch.size_m, launch.hidden_size),\n'
                   '            dtype={"fp32": torch.float32, "fp16": torch.float16,\n'
                   '                   "bf16": torch.bfloat16}[launch.topk_sum.output_element_dtype],\n'
                   '            device=device')
    text = replace(text, "            cutlass.Float32,\n            buffers.output.data_ptr(),",
                   '            (cutlass.Float32 if launch.topk_sum.output_element_dtype == "fp32"\n'
                   '             else _cutlass_element_dtype(launch.topk_sum.output_element_dtype)),\n'
                   '            buffers.output.data_ptr(),', 2)
    return text


def gather(text):
    text = replace(text, "the NIC-written slots with system-scope loads.\n",
                   "the NIC-written slots with system-scope loads.\n\n"
                   "Launch grid and message size are runtime scalars.  The host picks a\n"
                   "power-of-two grid from the shard size and hands the kernel that grid's own\n"
                   "staging and tail counters, so small gathers do not pay for the full grid and\n"
                   "gathers may interleave with reductions of any size inside one CUDA graph.\n")
    text = replace(text, "            fence_sc_sys()\n            cute.arch.sync_threads()\n\n"
                   "            # 2. the last block to finish staging rings the proxy doorbell\n"
                   "            if Int32(tidx) == Int32(0):\n",
                   "            cute.arch.sync_threads()\n\n"
                   "            # 2. the last block to finish staging rings the proxy doorbell.  One\n"
                   "            # system fence per block after the barrier (cumulative over the\n"
                   "            # staging stores the barrier ordered) replaces one per thread.\n"
                   "            if Int32(tidx) == Int32(0):\n"
                   "                fence_sc_sys()\n")
    return replace(text, '"comm.roce.allgather", 3, cache_key', '"comm.roce.allgather", 4, cache_key')


def roce(text):
    start = text.index("    def _launch_gather(")
    end = text.index("\n    def ", start + 1)
    part = text[start:end]
    part = replace(part, "        stage_counter, tail_counter = self._counter_addresses(self._blocks)\n",
                   "        # Same size-aware geometry as the all-reduce: a small shard (top-k\n"
                   "        # values and ids, MTP logits) launches a few blocks instead of the\n"
                   "        # full grid, and each power-of-two grid has its own arrival counters\n"
                   "        # so gathers and reductions of any size may interleave in one graph.\n"
                   "        grid_blocks = _grid_blocks(nbytes // PACK_BYTES, self._threads, self._blocks)\n"
                   "        stage_counter, tail_counter = self._counter_addresses(grid_blocks)\n")
    part = replace(part, "            self._blocks,\n", "            grid_blocks,\n")
    return text[:start] + part + text[end:]


TRANSFORMS = {
    "moe/_shared/kernels/w4a16/kernel.py": kernel,
    "moe/_shared/kernels/w4a16/mixed_trellis.py": mixed,
    "comm/roce/_allgather_cute.py": gather,
    "comm/roce/roce_oneshot.py": roce,
}


def patch(root: Path, check=False):
    pending = []
    for name, transform in TRANSFORMS.items():
        path = root / name
        source = path.read_text(encoding="utf-8")
        digest = hashlib.sha256(source.encode()).hexdigest()
        if digest == OUTPUT_HASHES[name]:
            continue
        if check or digest != INPUT_HASHES[name]:
            raise RuntimeError(f"unexpected v11 source: {path} ({digest})")
        result = transform(source)
        compile(result, str(path), "exec")
        if hashlib.sha256(result.encode()).hexdigest() != OUTPUT_HASHES[name]:
            raise RuntimeError(f"v11 output hash mismatch: {path}")
        pending.append((path, result))
    for path, result in pending:
        path.write_text(result, encoding="utf-8", newline="\n")
    print(f"{VERSION}: verified {root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    patch(args.root, args.check)
