#!/usr/bin/env python3
"""GB10 FC1 whole-tile tail split: stripe only the ragged final M8 decode wave."""
import argparse
import hashlib
from pathlib import Path

VERSION = 'glm53-r22-v20-1'
KERNEL = 'moe/_shared/kernels/w4a16/kernel.py'
INPUTS = {KERNEL: '88b864c62ac8b9f9337435ff57cc5dd7788da7f32af6b3ca72c713291de6a16a'}
OUTPUTS = {KERNEL: 'f37db321a01d178b1799ffdba0d6d7eca018163299cfa76915b23704c8e436f6'}

HELPER = '''def _gb10_fc1_tail_splitk_enabled() -> bool:
    """GB10-only FC1 whole-tile tail split for route-packed M8 decode plans.

    The whole-tile schedule idles every SM outside a ragged final FC1 wave
    while that wave's whole-K tiles finish. On a 48-SM GB10, one decode token
    routed to eight distinct experts produces 64 FC1 mn-tiles: one full wave
    plus a 16-tile remainder that otherwise occupies only 16 SMs. MTP3
    verification batches land on the same remainder pattern. When enabled,
    complete waves stay whole-K and ONLY the ragged remainder is striped
    across every CTA along K, reusing the existing split-K slice, lock and
    fc1_scratch finalize (three slices per remainder tile at 64 mn-tiles;
    the pair-rate and staging paths already take absolute K-tile indices
    under sliced jobs). For this geometry the remainder is always a multiple
    of n_tiles between 8 and 40 tiles, so the per-tile lock slots stay far
    inside the sms*4 workspace and the doubled M8 fp32 scratch already
    covers every decode shape. Exact grid fills and single-wave batches keep
    the stock idle-CTA handoff: a split-K finalize costs more than it
    recovers for phases that small. Read during GEMM compilation and part of
    every affected cache key; restart all workers after changing it. The
    global stripe switch above rewrites both phases and FC2 grouping, so the
    two are mutually exclusive; this schedule preserves FC2 grouping and
    every prefill (M16/M32/M64) plan unchanged.
    """

    if torch.cuda.get_device_capability() != (12, 1):
        return False
    value = os.environ.get("VLLM_GB10_EXL3_FC1_TAILSPLIT", "0")
    if value not in ("0", "1"):
        raise ValueError("VLLM_GB10_EXL3_FC1_TAILSPLIT must be 0 or 1")
    return value == "1"


'''


def replace(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f'v20 source anchor mismatch: {old[:100]!r}')
    return text.replace(old, new, 1)


def transform(text):
    # 1) Env helper next to the existing schedule switch.
    text = replace(text,
        '    return os.environ.get("B12X_W4A16_SMALL_M_SPLITK", "0") == "1"\n'
        '\n'
        '\n'
        'def _sqg_xor_cheb_t12_smem_enabled() -> bool:\n',
        '    return os.environ.get("B12X_W4A16_SMALL_M_SPLITK", "0") == "1"\n'
        '\n'
        '\n'
        + HELPER +
        'def _sqg_xor_cheb_t12_smem_enabled() -> bool:\n')
    # 2) GEMM constructor parameter.
    text = replace(text,
        '        schedule_whole_tiles: bool = False,\n'
        '        dynamic_num_experts: bool = False,\n'
        '        schedule_route_block_factor: int = 1,\n',
        '        schedule_whole_tiles: bool = False,\n'
        '        dynamic_num_experts: bool = False,\n'
        '        schedule_route_block_factor: int = 1,\n'
        '        whole_tile_tail_splitk: bool = False,\n')
    # 3) GEMM constructor storage and fail-closed validation.
    text = replace(text,
        '        if (\n'
        '            self.schedule_whole_tiles\n'
        '            and not self.direct_topk_routes\n'
        '            and not self.weight_layout_trellis256\n'
        '        ):\n'
        '            raise ValueError(\n'
        '                "schedule_whole_tiles requires direct_topk_routes '
        'or trellis_t256"\n'
        '            )\n',
        '        if (\n'
        '            self.schedule_whole_tiles\n'
        '            and not self.direct_topk_routes\n'
        '            and not self.weight_layout_trellis256\n'
        '        ):\n'
        '            raise ValueError(\n'
        '                "schedule_whole_tiles requires direct_topk_routes '
        'or trellis_t256"\n'
        '            )\n'
        '        # GB10 FC1 whole-tile tail split (v20). The switch may only\n'
        '        # ever reach route-packed M8 GEMMs under the whole-tile\n'
        '        # schedule; any other geometry must reject it instead of\n'
        '        # silently reinterpreting the scheduler below.\n'
        '        self.whole_tile_tail_splitk = bool(whole_tile_tail_splitk)\n'
        '        if self.whole_tile_tail_splitk and (\n'
        '            not self.schedule_whole_tiles\n'
        '            or int(moe_block_size) != 8\n'
        '            or self.direct_topk_routes\n'
        '            or self.dense_route_fast_path\n'
        '        ):\n'
        '            raise ValueError(\n'
        '                "whole_tile_tail_splitk requires the route-packed M8 "\n'
        '                "whole-tile schedule"\n'
        '            )\n')
    # 4) Compiled-kernel cache key.
    text = replace(text,
        '            self.schedule_whole_tiles,\n'
        '            self.schedule_route_block_factor,\n'
        '            self.sqg_xor_cheb_t12_smem,\n'
        '            self.small_m_splitk,\n'
        '        )\n',
        '            self.schedule_whole_tiles,\n'
        '            self.schedule_route_block_factor,\n'
        '            self.sqg_xor_cheb_t12_smem,\n'
        '            self.small_m_splitk,\n'
        '            self.whole_tile_tail_splitk,\n'
        '        )\n')
    # 5) Scheduler: keep complete whole-K waves, stripe only the ragged
    #    remainder along K through the existing tail/slice machinery.
    text = replace(text,
        '        if cutlass.const_expr(self.schedule_whole_tiles):\n'
        '            # Whole-tile waves: one CTA computes each mn-tile over '
        'the full K,\n'
        '            # task = cta + wave * grid_x, ragged last wave skipped '
        'through the\n'
        '            # route_block_idx bound below. No split-K tail, no lock '
        'traffic.\n'
        '            tail_mn_tiles = Int32(0)\n'
        '            full_grid_mn_iters = (global_mn_tiles + grid_x - '
        'Int32(1)) // grid_x\n',
        '        if cutlass.const_expr(self.schedule_whole_tiles):\n'
        '            # Whole-tile waves: one CTA computes each mn-tile over '
        'the full K,\n'
        '            # task = cta + wave * grid_x, ragged last wave skipped '
        'through the\n'
        '            # route_block_idx bound below. No split-K tail, no lock '
        'traffic.\n'
        '            tail_mn_tiles = Int32(0)\n'
        '            full_grid_mn_iters = (global_mn_tiles + grid_x - '
        'Int32(1)) // grid_x\n'
        '            if cutlass.const_expr(self.whole_tile_tail_splitk):\n'
        '                # GB10 FC1 tail split (v20): the ragged remainder '
        'always has\n'
        '                # fewer than grid_x mn-tiles, and for this FC1 '
        'geometry it is\n'
        '                # a multiple of n_tiles. Keep completed waves '
        'whole-K and\n'
        '                # stripe only that remainder across every CTA along '
        'K through\n'
        '                # the existing slice/lock finalize below. At 64 '
        'mn-tiles on a\n'
        '                # 48-CTA grid the 16-tile remainder covers 48 k-'
        'cells per CTA\n'
        '                # in three lock-ordered fp32 slices per tile '
        'instead of\n'
        '                # leaving 32 CTAs idle. Exact fills and single-wave '
        'batches\n'
        '                # keep the stock schedule above: a split-K finalize '
        'is not\n'
        '                # worth it for phases that small.\n'
        '                full_waves = global_mn_tiles // grid_x\n'
        '                ragged_tiles = global_mn_tiles - full_waves * '
        'grid_x\n'
        '                if full_waves > Int32(0) and ragged_tiles > '
        'Int32(0):\n'
        '                    full_grid_mn_iters = full_waves\n'
        '                    tail_mn_tiles = ragged_tiles\n')
    # 6) Fused kernel resolves the switch for FC1 only, guarded by the
    #    schedule it actually picked.
    text = replace(text,
        '        self.schedule_whole_tiles = bool(\n'
        '            (schedule_whole_tiles or weight_layout == "trellis_t256")\n'
        '            and not self.small_m_splitk\n'
        '        )\n',
        '        self.schedule_whole_tiles = bool(\n'
        '            (schedule_whole_tiles or weight_layout == "trellis_t256")\n'
        '            and not self.small_m_splitk\n'
        '        )\n'
        '        # v20 GB10 FC1 whole-tile tail split: route-packed M8 plans '
        'may\n'
        '        # stripe only the ragged final whole-tile wave along K. '
        'Prefill\n'
        '        # (M16/M32/M64), dense direct-topk decode and FC2 keep the '
        'v19\n'
        '        # schedule; the global stripe switch above rewrites both '
        'phases\n'
        '        # and stays exclusive with this one.\n'
        '        self.fc1_tail_splitk = (\n'
        '            _gb10_fc1_tail_splitk_enabled()\n'
        '            and not self.small_m_splitk\n'
        '            and self.schedule_whole_tiles\n'
        '            and self.moe_block_size == 8\n'
        '            and not self.direct_topk_routes\n'
        '        )\n')
    # 7) Only the FC1 GEMM constructor receives the switch.
    text = replace(text,
        '            schedule_whole_tiles=self.schedule_whole_tiles,\n'
        '            dynamic_num_experts=self.dynamic_num_experts,\n'
        '        )\n'
        '        self.fc2 = W4A16GemmKernel(\n',
        '            schedule_whole_tiles=self.schedule_whole_tiles,\n'
        '            dynamic_num_experts=self.dynamic_num_experts,\n'
        '            whole_tile_tail_splitk=self.fc1_tail_splitk,\n'
        '        )\n'
        '        self.fc2 = W4A16GemmKernel(\n')
    return text


def patch(root, check=False):
    path = Path(root) / KERNEL
    source = path.read_text(encoding='utf-8')
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest != OUTPUTS[KERNEL]:
        if check or digest != INPUTS[KERNEL]:
            raise RuntimeError(f'unexpected v20 source: {path} ({digest})')
        result = transform(source)
        compile(result, str(path), 'exec')
        if hashlib.sha256(result.encode()).hexdigest() != OUTPUTS[KERNEL]:
            raise RuntimeError(f'v20 output hash mismatch: {path}')
        path.write_text(result, encoding='utf-8', newline='\n')
    print(f'{VERSION}: verified {root}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    patch(args.root, args.check)
