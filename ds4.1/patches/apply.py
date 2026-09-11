#!/usr/bin/env python3
"""Apply the FP4 disk adapter to the pinned vLLM/B12x sources. Fail on drift."""
import argparse
import hashlib
import json
from pathlib import Path


def replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one patch anchor: {old[:90]!r}")
    return text.replace(old, new)


def patch_vllm(text):
    text = replace(text, '        self.table_memory = (', '''        qc = getattr(config, "quantization_config", None) or {}
        self.weight_format = qc.get("engram_dtype", "fp8")
        if self.weight_format not in ("fp8", "fp4"):
            raise ValueError("Unsupported Engram weight format")
        if self.weight_format == "fp4" and (
            qc.get("engram_block_size") != 32
            or qc.get("engram_scale_fmt") != "ue8m0"
        ):
            raise ValueError("FP4 Engram requires block32/E8M0 scales")
        self.table_memory = (''')
    text = replace(text, '    def __init__(self, plan, table_memory="device"):',
                   '    def __init__(self, plan, table_memory="device", weight_format="fp8"):')
    text = replace(text,
                   '        self.disk_table = native.DiskTable(plan) if table_memory == "disk" else None',
                   '''        if weight_format == "fp4" and table_memory != "disk":
            raise ValueError("This FP4 Engram adapter requires table_memory=disk")
        self.weight_format = weight_format
        self.disk_table = (
            native.DiskTable(plan, weight_format=weight_format)
            if table_memory == "disk" else None
        )''')
    text = replace(text,
                   '                width = self.plan.scale_shape[1] if scale else self.plan.weight_shape[1]',
                   '''                width = (self.plan.scale_shape[1] if scale else
                         (128 if self.weight_format == "fp4" else 256))''')
    text = replace(text, '                    else (torch.float8_e4m3fn,)',
                   '                    else ((torch.uint8,) if self.weight_format == "fp4" else (torch.float8_e4m3fn,))')
    text = replace(text,
                   '                self.disk_table.add_shard(0, source.path, source.offset, scale=scale)',
                   '''                self.disk_table.add_shard(0, source.path, source.offset, scale=scale)
                if not scale:
                    print(f"[DS41_FP4_DISK] format={self.weight_format} "
                          f"stored_row_bytes={width} global_rows={self.plan.table_rows} "
                          f"rank={self.plan.caps.tp_rank}", flush=True)''')
    return replace(text,
                   '        self.embed_tokens = ParallelEngramEmbedding(plan, layout.table_memory)',
                   '''        self.embed_tokens = ParallelEngramEmbedding(
            plan, layout.table_memory, layout.weight_format
        )''')


def patch_b12x_v1(text):
    text = replace(text,
                   '        self, plan: Plan, shard_rows: int | None = None, queue_depth: int = 64',
                   '        self, plan: Plan, shard_rows: int | None = None, queue_depth: int = 64, *, weight_format: str = "fp8"')
    text = replace(text, '        self.plan = plan\n        self._cache = DiskRowCache(',
                   '''        if weight_format not in ("fp8", "fp4"):
            raise ValueError("Unsupported disk Engram weight_format")
        self.packed_fp4 = weight_format == "fp4"
        self.plan = plan
        self._cache = DiskRowCache(''')
    text = replace(text, '            weight_row_bytes=256,',
                   '            weight_row_bytes=128 if self.packed_fp4 else 256,')
    text = replace(text, '        self.weight = self._cache.weight.view(torch.float8_e4m3fn)',
                   '''        if self.packed_fp4:
            from ._fp4_disk import expand_rows
            self.weight = torch.empty(
                (plan.caps.max_tokens * 24, 256),
                dtype=torch.float8_e4m3fn, device=plan.caps.device,
            )
            # Compile before serving and before graph capture/JIT monitoring.
            expand_rows(self._cache.weight, self.weight, 1)
        else:
            self.weight = self._cache.weight.view(torch.float8_e4m3fn)''')
    return replace(text, '            cache.read_rows(b.hash_ids, prepared * 24)\n            lookup_op(',
                   '''            cache.read_rows(b.hash_ids, prepared * 24)
            if b.disk_table.packed_fp4:
                from ._fp4_disk import expand_rows
                expand_rows(cache.weight, b.weight, prepared * 24)
            lookup_op(''')


def upgrade_b12x_v1(text):
    text = replace(text, '''        if self.packed_fp4:
            from ._fp4_disk import expand_rows
            self.weight = torch.empty(
                (plan.caps.max_tokens * 24, 256),
                dtype=torch.float8_e4m3fn, device=plan.caps.device,
            )
            # Compile before serving and before graph capture/JIT monitoring.
            expand_rows(self._cache.weight, self.weight, 1)
        else:
            self.weight = self._cache.weight.view(torch.float8_e4m3fn)''',
                   '''        self.weight = (self._cache.weight if self.packed_fp4 else
                       self._cache.weight.view(torch.float8_e4m3fn))
        if self.packed_fp4:
            # Warm the exact fused variant before serving/JIT monitoring.
            # Only one token of temporary output; all row reads are masked.
            from ._kernels import _lookup
            _lookup[(1, 24)](
                self.weight, self._cache.scale,
                torch.zeros((1, 24), dtype=torch.int64, device=plan.caps.device),
                torch.zeros((1,), dtype=torch.int32, device=plan.caps.device),
                torch.empty((1, 6144), dtype=torch.bfloat16, device=plan.caps.device),
                0, plan.caps.max_tokens, plan.table_rows, plan.shard_start,
                plan.shard_end, True, True, num_warps=4,
            )''')
    text = replace(text, '    c = plan.caps\n    if disk_table is not None:',
                   '    c = plan.caps\n    packed_fp4 = disk_table is not None and disk_table.packed_fp4\n    if disk_table is not None:')
    text = replace(text,
                   '        weight_shape, scale_shape = (c.max_tokens * 24, 256), (c.max_tokens * 24, 8)',
                   '        weight_shape, scale_shape = (c.max_tokens * 24, 128 if packed_fp4 else 256), (c.max_tokens * 24, 8)')
    text = replace(text, '        ("weight", weight, weight_shape, torch.float8_e4m3fn),',
                   '        ("weight", weight, weight_shape, torch.uint8 if packed_fp4 else torch.float8_e4m3fn),')
    text = replace(text, '''            if b.disk_table.packed_fp4:
                from ._fp4_disk import expand_rows
                expand_rows(cache.weight, b.weight, prepared * 24)
''', '')
    return replace(text, '                compact_rows=True,\n                prepared_tokens=prepared,',
                   '                compact_rows=True,\n                prepared_tokens=prepared,\n                packed_fp4=b.disk_table.packed_fp4,')


def patch_b12x(text):
    return upgrade_b12x_v1(patch_b12x_v1(text))


def patch_kernels(text):
    text = replace(text, '    COMPACT: tl.constexpr,\n):',
                   '    COMPACT: tl.constexpr,\n    PACKED_FP4: tl.constexpr = False,\n):')
    text = replace(text, '''    quant = tl.load(weight + local_row * 256 + col.to(tl.int64), local, 0.0).to(
        tl.float32
    )''', '''    if PACKED_FP4:
        packed = tl.load(
            weight + local_row * 128 + (col // 2).to(tl.int64), local, 0
        ).to(tl.uint32)
        code = (packed >> ((col % 2) * 4)) & 15
        mag = code & 7
        # E2M1 magnitudes: 0,.5,1,1.5,2,3,4,6. All exact in FP32.
        magnitude = tl.where(mag < 4, mag.to(tl.float32) * 0.5,
                             (2 + (mag & 1)).to(tl.float32) * tl.where(mag < 6, 1., 2.))
        quant = tl.where((code & 8) != 0, -magnitude, magnitude)
    else:
        quant = tl.load(weight + local_row * 256 + col.to(tl.int64), local, 0.0).to(
            tl.float32
        )''')
    text = replace(text, '    prepared_tokens: int = -1,\n) -> None:',
                   '    prepared_tokens: int = -1,\n    packed_fp4: bool = False,\n) -> None:')
    text = replace(text, '        compact_rows,\n        num_warps=4,',
                   '        compact_rows,\n        packed_fp4,\n        num_warps=4,')
    return replace(text, '    prepared_tokens=-1,\n):',
                   '    prepared_tokens=-1,\n    packed_fp4=False,\n):')


def canonical(text):
    return text.rstrip() + "\n"


def sha(text):
    return hashlib.sha256(canonical(text).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--b12x-root", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    manifest = json.loads((here / "source-hashes.json").read_text())
    writes = []
    for root, name, transform in (
        (args.vllm_root, "vllm/models/deepseek_v4_1/common/engram.py", patch_vllm),
        (args.b12x_root, "b12x/sequence/engram/api.py", patch_b12x),
        (args.b12x_root, "b12x/sequence/engram/_kernels.py", patch_kernels),
    ):
        path = root / name
        original = canonical(path.read_text())
        current = sha(original)
        if current == manifest[name]["output"]:
            continue
        if current == manifest[name].get("v1_output"):
            patched = canonical(upgrade_b12x_v1(original))
        elif current == manifest[name]["input"]:
            patched = canonical(transform(original))
        else:
            raise ValueError(f"Unrecognized source: {path} SHA256={current}")
        assert sha(patched) == manifest[name]["output"]
        compile(patched, str(path), "exec")
        writes.append((path, patched))
    target = args.b12x_root / "b12x/sequence/engram/_fp4_disk.py"
    content = (here / "fp4_disk.py").read_text()
    if target.exists() and target.read_text() != content:
        raise ValueError(f"Unexpected existing helper: {target}")
    # All gates pass before writing either source.
    for path, patched in writes:
        path.write_text(patched)
    target.write_text(content)
    print("FP4 Engram SSD adapter applied/verified (ds41-fp4-disk-v2)")


if __name__ == "__main__":
    main()
