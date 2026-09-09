#!/usr/bin/env python3
"""Archived R7 paired FC2 M8 weight reuse: one B stream for two M8 subtiles."""
import argparse
import hashlib
from pathlib import Path

VERSION = 'glm53-r22-v21-1'
KERNEL = 'moe/_shared/kernels/w4a16/kernel.py'
MIXED = 'moe/_shared/kernels/w4a16/mixed_trellis.py'
# Both inputs are the complete v20 image state: kernel.py is patch_r22_v20's
# output and mixed_trellis.py is patch_r22_v19's output. The v20
# instanttensor-r1 layer changes only loader policy, so both hashes describe
# that image exactly.
INPUTS = {
    KERNEL: 'f37db321a01d178b1799ffdba0d6d7eca018163299cfa76915b23704c8e436f6',
    MIXED: 'e2e645f569c26b626900e1e15ee672becce82cd1587c7edb0d5e9b1cb20cce41',
}
OUTPUTS = {
    KERNEL: '680e52351266717150abc618d19d80dfbb0449b4222753a45874efd1593e62a1',
    MIXED: '7c38f9b33d40515b60d28785f15b7ff1510866da8631e9bec1d336f5247fbea6',
}

HELPER = '''def _gb10_fc2_m8_pair_enabled() -> bool:
    """GB10-only archived R7 paired FC2 for grouped M8 prefill plans.

    The archived R7 kernel decodes one B/scale bundle and applies the decoded
    weight fragment to two independent M8 accumulator sets. R22's grouped
    schedule instead repeats the complete B staging and trellis LUT
    dequantization for every M8 subtile of a route block. Pairing halves
    those repetitions at equal work; it does not imply half the DRAM traffic
    or twice the model throughput, and FC1, attention and collectives remain.
    Read during GEMM compilation and part of every affected cache key;
    restart all workers after changing it. M8 decode plans keep factor 1 and
    therefore never execute the pair path: the pair requires an even grouping
    factor of 2 or 4, which only exists for M16/M32/M64 route blocks. The
    group2 comparison isolates weight reuse from grouping; the production
    group4 dispatch becomes two pair calls under the same switch.
    """

    if torch.cuda.get_device_capability() != (12, 1):
        return False
    value = os.environ.get("VLLM_GB10_EXL3_FC2_M8_PAIR", "0")
    if value not in ("0", "1"):
        raise ValueError("VLLM_GB10_EXL3_FC2_M8_PAIR must be 0 or 1")
    return value == "1"


'''

PAIR_READ_BLOCK = '''    @cute.jit
    def _read_moe_block_data_pair(
        self,
        packed_route_indices: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        route_block_idx: Int32,
        global_scale_f32: cutlass.Float32,
        active_size_m: Int32,
    ):
        """Load two adjacent M8 route blocks into one 16-row metadata slab.

        v21 archived R7 paired FC2: the doubled route/rd-route/top-k regions
        hold both M8 halves. Each half keeps its own padded metadata rows and
        valid count; the valid-count slot stays a single region.
        """
        route_indices_int4_addr = self._int4_addr(
            smem_base, Int32(self.sh_route_off) + tid
        )
        route_indices_gmem = get_ptr_as_int64(
            packed_route_indices,
            route_block_idx * Int32(self.moe_block_size) + tid * Int32(4),
        )
        cp_async4_shared_global_pred(
            route_indices_int4_addr,
            route_indices_gmem,
            (tid < Int32(2 * self.moe_block_size // 4)).to(Int32),
        )
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        if tid >= Int32(self.cta_threads - 32):
            lane = tid - Int32(self.cta_threads - 32)
            valid0 = Int32(0)
            valid1 = Int32(0)
            if lane < Int32(2 * self.moe_block_size):
                idx = ld_shared_i32_relaxed(
                    smem_base + Int32(self.sh_route_off * 16) + lane * Int32(4)
                )
                valid = (idx < active_size_m * Int32(self.top_k)).to(Int32)
                if lane < Int32(self.moe_block_size):
                    valid0 = valid
                else:
                    valid1 = valid
            valid0 = cute.arch.warp_redux_sync(valid0, "add")
            valid1 = cute.arch.warp_redux_sync(valid1, "add")
            if lane == Int32(0):
                valid_addr = smem_base + Int32(self.sh_valid_count_off * 16)
                st_shared_i32(valid_addr, valid0)
                st_shared_i32(valid_addr + Int32(4), valid1)

        if tid < Int32(2 * self.moe_block_size):
            idx = ld_shared_i32_relaxed(
                smem_base + Int32(self.sh_route_off * 16) + tid * Int32(4)
            )
            rd_row = idx // Int32(self.top_k)
            if cutlass.const_expr(self.route_major_a):
                rd_row = idx // Int32(self.input_route_divisor)
            st_shared_i32(
                smem_base + Int32(self.sh_rd_route_off * 16) + tid * Int32(4),
                rd_row,
            )
            if cutlass.const_expr(self.mul_topk_weights):
                safe_idx = idx
                if idx >= active_size_m * Int32(self.top_k):
                    safe_idx = Int32(0)
                topk = (
                    topk_weights_flat[safe_idx].to(cutlass.Float32) * global_scale_f32
                )
                st_shared_u32(
                    smem_base + Int32(self.sh_topk_off * 16) + tid * Int32(4),
                    self._broadcast_f32_to_elem2(topk),
                )

        cute.arch.sync_threads()
        valid_addr = smem_base + Int32(self.sh_valid_count_off * 16)
        block_valid_rows0 = ld_shared_i32_relaxed(valid_addr)
        block_valid_rows1 = ld_shared_i32_relaxed(valid_addr + Int32(4))
        cute.arch.sync_threads()
        return block_valid_rows0, block_valid_rows1

'''

PAIR_PROLOGUE_AND_TILE = '''    @cute.jit
    def _tile_common_prologue_pair(
        self,
        global_scale: cute.Tensor,
        packed_route_indices: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        route_block_idx: Int32,
        expert_idx: Int32,
        output_n_tile: Int32,
        active_size_m: Int32,
    ):
        global_scale_f32 = global_scale[expert_idx].to(cutlass.Float32)
        if cutlass.const_expr(self.scale_format_e8m0_k32):
            if cutlass.const_expr(self.is_fp16):
                global_scale_f32 *= cutlass.Float32(_E8M0_K32_FP16_GLOBAL_COMPENSATION)
            else:
                global_scale_f32 *= cutlass.Float32(_E8M0_K32_BF16_GLOBAL_COMPENSATION)
        block_valid_rows0, block_valid_rows1 = self._read_moe_block_data_pair(
            packed_route_indices,
            topk_weights_flat,
            smem_base,
            tid,
            route_block_idx,
            global_scale_f32,
            active_size_m,
        )
        offsets = self._tile_stream_offsets(tid, expert_idx, output_n_tile)
        return (
            global_scale_f32,
            block_valid_rows0,
            block_valid_rows1,
            *offsets,
        )

    @cute.jit
    def _run_tile_m8_pair(
        self,
        a_bf16_flat: cute.Tensor,
        a_alt_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        c_bf16_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        global_scale: cute.Tensor,
        packed_route_indices: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
        trellis_lut_addr: Int64,
        smem_base: Int32,
        tid: Int32,
        route_block_idx: Int32,
        expert_idx: Int32,
        output_n_tile: Int32,
        reduce_k_tile: Int32,
        reduce_tile_count: Int32,
        reduce_slice_count: Int32,
        reduce_slice_idx: Int32,
        lock_slot: Int32,
        active_size_m: Int32,
    ):
        """v21 archived R7 paired FC2: one B stream, two independent M8
        accumulator sets. Each M8 half has its own padded 16-row shared-memory
        slab and output metadata rows, and both halves keep the current ABI,
        output fusion and rotations. The decoded weight fragment applies to
        both accumulator sets, so B staging and trellis LUT dequantization run
        once per pair instead of once per M8 subtile."""
        (
            global_scale_f32,
            block_valid_rows0,
            block_valid_rows1,
            a_gl_stride,
            b_gl_stride,
            s_gl_stride,
            scales_expert_off,
            b_gl_rd_base,
            a_gl_rd_row,
            a_gl_rd_col0,
            a_sh_wr,
            a_rows_per_iter,
            b_sh_rd,
            s_sh_rd,
        ) = self._tile_common_prologue_pair(
            global_scale,
            packed_route_indices,
            topk_weights_flat,
            smem_base,
            tid,
            route_block_idx,
            expert_idx,
            output_n_tile,
            active_size_m,
        )
        a0_sh_rd = self._a_shared_read_offset(tid, 8)
        a1_sh_rd = a0_sh_rd + Int32(self.a_sh_rd_delta_i)

        acc0 = [
            cute.make_rmem_tensor((_SCALAR_ACC_FRAGMENT_WIDTH,), cutlass.Float32)
            for _ in range(16 // _SCALAR_ACC_FRAGMENT_WIDTH)
        ]
        acc1 = [
            cute.make_rmem_tensor((_SCALAR_ACC_FRAGMENT_WIDTH,), cutlass.Float32)
            for _ in range(16 // _SCALAR_ACC_FRAGMENT_WIDTH)
        ]
        for frag in cutlass.range_constexpr(16 // _SCALAR_ACC_FRAGMENT_WIDTH):
            acc0[frag].fill(0.0)
            acc1[frag].fill(0.0)

        k_tiles = reduce_tile_count
        self._prefetch_initial_tiles(
            a_bf16_flat,
            a_alt_bf16_flat,
            b_i32_flat,
            scales_i32_flat,
            smem_base,
            tid,
            k_tiles,
            reduce_k_tile,
            block_valid_rows0,
            block_valid_rows1,
            True,
            a_gl_stride,
            b_gl_stride,
            s_gl_stride,
            scales_expert_off,
            b_gl_rd_base,
            a_gl_rd_row,
            a_gl_rd_col0,
            a_sh_wr,
            a_rows_per_iter,
            output_n_tile,
            expert_idx,
            -1,
        )

        b_scale_cur = cute.make_rmem_tensor((2, 4), Uint32)
        b_scale_next = cute.make_rmem_tensor((2, 4), Uint32)
        self._load_b_scale_register_bundle(
            b_scale_cur,
            smem_base,
            tid,
            b_sh_rd,
            s_sh_rd,
            Int32(0),
            Int32(0),
            reduce_k_tile,
            -1,
        )
        a0_regs_cur = cute.make_rmem_tensor((2,), Uint32)
        a0_regs_next = cute.make_rmem_tensor((2,), Uint32)
        a1_regs_cur = cute.make_rmem_tensor((2,), Uint32)
        a1_regs_next = cute.make_rmem_tensor((2,), Uint32)
        self._load_a_registers_m8_bundle(
            a0_regs_cur, smem_base, a0_sh_rd, Int32(0), Int32(0)
        )
        self._load_a_registers_m8_bundle(
            a1_regs_cur, smem_base, a1_sh_rd, Int32(0), Int32(0)
        )
        self._run_mma_pipeline_m8_pair(
            a_bf16_flat,
            a_alt_bf16_flat,
            b_i32_flat,
            scales_i32_flat,
            trellis_lut_addr,
            smem_base,
            tid,
            acc0,
            acc1,
            b_scale_cur,
            b_scale_next,
            a0_regs_cur,
            a0_regs_next,
            a1_regs_cur,
            a1_regs_next,
            b_sh_rd,
            s_sh_rd,
            a0_sh_rd,
            a1_sh_rd,
            k_tiles,
            reduce_k_tile,
            block_valid_rows0,
            block_valid_rows1,
            a_gl_stride,
            b_gl_stride,
            s_gl_stride,
            scales_expert_off,
            b_gl_rd_base,
            a_gl_rd_row,
            a_gl_rd_col0,
            a_sh_wr,
            a_rows_per_iter,
            output_n_tile,
            expert_idx,
        )

        self._finish_tile(
            acc0,
            acc0,
            acc0,
            acc0,
            c_bf16_flat,
            c_tmp_f32_flat,
            locks_i32_flat,
            smem_base,
            tid,
            output_n_tile,
            block_valid_rows0,
            Int32(0),
            global_scale_f32,
            reduce_slice_count,
            reduce_slice_idx,
            lock_slot,
            True,
        )
        self._finish_tile(
            acc1,
            acc1,
            acc1,
            acc1,
            c_bf16_flat,
            c_tmp_f32_flat,
            locks_i32_flat,
            smem_base,
            tid,
            output_n_tile,
            block_valid_rows1,
            Int32(self.moe_block_size),
            global_scale_f32,
            reduce_slice_count,
            reduce_slice_idx,
            lock_slot + Int32(1),
            True,
        )

    @cute.jit
    def _run_mma_pipeline_m8_pair(
        self,
        a_bf16_flat: cute.Tensor,
        a_alt_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        trellis_lut_addr: Int64,
        smem_base: Int32,
        tid: Int32,
        acc0,
        acc1,
        b_scale_cur: cute.Tensor,
        b_scale_next: cute.Tensor,
        a0_regs_cur: cute.Tensor,
        a0_regs_next: cute.Tensor,
        a1_regs_cur: cute.Tensor,
        a1_regs_next: cute.Tensor,
        b_sh_rd: Int32,
        s_sh_rd: Int32,
        a0_sh_rd: Int32,
        a1_sh_rd: Int32,
        k_tiles: Int32,
        reduce_k_tile: Int32,
        block_valid_rows0: Int32,
        block_valid_rows1: Int32,
        a_gl_stride: Int32,
        b_gl_stride: Int32,
        s_gl_stride: Int32,
        scales_expert_off: Int32,
        b_gl_rd_base: Int32,
        a_gl_rd_row: Int32,
        a_gl_rd_col0: Int32,
        a_sh_wr: Int32,
        a_rows_per_iter: Int32,
        output_n_tile: Int32,
        expert_idx: Int32,
    ):
        """Archived R7 paired M8 MMA pipeline: one decoded weight fragment
        applies to both accumulator sets with independent A operands. Pair
        staging takes absolute K-tile indices, so sliced jobs and graph replay
        stay clean."""
        b_frag = cute.make_rmem_tensor((2, 2), Uint32)
        tile_idx = Int32(0)
        while tile_idx < k_tiles:
            for pipe in cutlass.range_constexpr(_STAGES):
                if tile_idx < k_tiles:
                    for kk in cutlass.range_constexpr(self.b_sh_wr_iters):
                        self._load_next_fragment_bundle_m8_pair(
                            b_scale_next,
                            a0_regs_next,
                            a1_regs_next,
                            smem_base,
                            tid,
                            b_sh_rd,
                            s_sh_rd,
                            a0_sh_rd,
                            a1_sh_rd,
                            pipe,
                            kk,
                            tile_idx,
                            k_tiles,
                            reduce_k_tile,
                        )

                        self._prefetch_pipeline_step(
                            a_bf16_flat,
                            a_alt_bf16_flat,
                            b_i32_flat,
                            scales_i32_flat,
                            smem_base,
                            tid,
                            pipe,
                            kk,
                            tile_idx,
                            k_tiles,
                            reduce_k_tile,
                            block_valid_rows0,
                            block_valid_rows1,
                            True,
                            a_gl_stride,
                            b_gl_stride,
                            s_gl_stride,
                            scales_expert_off,
                            b_gl_rd_base,
                            a_gl_rd_row,
                            a_gl_rd_col0,
                            a_sh_wr,
                            a_rows_per_iter,
                            output_n_tile,
                            expert_idx,
                            -1,
                        )

                        for jj in cutlass.range_constexpr(4):
                            if cutlass.const_expr(self.weight_layout_trellis256):
                                self._scaled_dequant_b_fragment_trellis256(
                                    b_frag,
                                    b_scale_cur[0, jj],
                                    b_scale_cur[1, jj],
                                    trellis_lut_addr,
                                )
                            else:
                                q, s = self._select_b_scale_register(jj, b_scale_cur)
                                self._scaled_dequant_b_fragment(b_frag, q, s)
                            self._mma_accumulate_m8(
                                acc0,
                                jj,
                                a0_regs_cur,
                                b_frag,
                            )
                            self._mma_accumulate_m8(
                                acc1,
                                jj,
                                a1_regs_cur,
                                b_frag,
                            )

                        self._copy_a_register_bundle_m8(a0_regs_cur, a0_regs_next)
                        self._copy_a_register_bundle_m8(a1_regs_cur, a1_regs_next)
                        self._copy_b_scale_register_bundle(b_scale_cur, b_scale_next)
                    tile_idx += Int32(1)
            cute.arch.sync_threads()
            if tile_idx < k_tiles:
                self._load_b_scale_register_bundle(
                    b_scale_cur,
                    smem_base,
                    tid,
                    b_sh_rd,
                    s_sh_rd,
                    Int32(0),
                    Int32(0),
                    reduce_k_tile + tile_idx,
                    -1,
                )
                self._load_a_registers_m8_bundle(
                    a0_regs_cur,
                    smem_base,
                    a0_sh_rd,
                    Int32(0),
                    Int32(0),
                )
                self._load_a_registers_m8_bundle(
                    a1_regs_cur,
                    smem_base,
                    a1_sh_rd,
                    Int32(0),
                    Int32(0),
                )

'''

PAIR_BUNDLE_LOADER = '''    @cute.jit
    def _load_next_fragment_bundle_m8_pair(
        self,
        b_scale_next: cute.Tensor,
        a0_regs_next: cute.Tensor,
        a1_regs_next: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        b_sh_rd: Int32,
        s_sh_rd: Int32,
        a0_sh_rd: Int32,
        a1_sh_rd: Int32,
        pipe: cutlass.Constexpr[int],
        kk: cutlass.Constexpr[int],
        tile_idx: Int32,
        k_tiles: Int32,
        reduce_k_tile: Int32,
    ):
        self._clear_b_scale_register_bundle(b_scale_next)
        self._clear_a_register_bundle_m8(a0_regs_next)
        self._clear_a_register_bundle_m8(a1_regs_next)

        if cutlass.const_expr(kk + 1 < self.b_sh_wr_iters):
            if tile_idx < k_tiles:
                self._load_b_scale_register_bundle(
                    b_scale_next,
                    smem_base,
                    tid,
                    b_sh_rd,
                    s_sh_rd,
                    Int32(pipe),
                    Int32(kk + 1),
                    reduce_k_tile + tile_idx,
                    -1,
                )
                self._load_a_registers_m8_bundle(
                    a0_regs_next,
                    smem_base,
                    a0_sh_rd,
                    Int32(pipe),
                    Int32(kk + 1),
                )
                self._load_a_registers_m8_bundle(
                    a1_regs_next,
                    smem_base,
                    a1_sh_rd,
                    Int32(pipe),
                    Int32(kk + 1),
                )
        else:
            next_tile = tile_idx + Int32(1)
            if next_tile < k_tiles:
                next_pipe = Int32((pipe + 1) % _STAGES)
                self._load_b_scale_register_bundle(
                    b_scale_next,
                    smem_base,
                    tid,
                    b_sh_rd,
                    s_sh_rd,
                    next_pipe,
                    Int32(0),
                    reduce_k_tile + next_tile,
                    -1,
                )
                self._load_a_registers_m8_bundle(
                    a0_regs_next,
                    smem_base,
                    a0_sh_rd,
                    next_pipe,
                    Int32(0),
                )
                self._load_a_registers_m8_bundle(
                    a1_regs_next,
                    smem_base,
                    a1_sh_rd,
                    next_pipe,
                    Int32(0),
                )

'''


def replace(text, old, new, count=1):
    if text.count(old) != count:
        raise RuntimeError(f'v21 source anchor mismatch: {old[:100]!r}')
    return text.replace(old, new, count)


def transform_kernel(text):
    # 1) Env helper next to the v20 FC1 tail-split switch.
    text = replace(text,
        '    value = os.environ.get("VLLM_GB10_EXL3_FC1_TAILSPLIT", "0")\n'
        '    if value not in ("0", "1"):\n'
        '        raise ValueError("VLLM_GB10_EXL3_FC1_TAILSPLIT must be 0 or 1")\n'
        '    return value == "1"\n'
        '\n'
        '\n'
        'def _sqg_xor_cheb_t12_smem_enabled() -> bool:\n',
        '    value = os.environ.get("VLLM_GB10_EXL3_FC1_TAILSPLIT", "0")\n'
        '    if value not in ("0", "1"):\n'
        '        raise ValueError("VLLM_GB10_EXL3_FC1_TAILSPLIT must be 0 or 1")\n'
        '    return value == "1"\n'
        '\n'
        '\n'
        + HELPER +
        'def _sqg_xor_cheb_t12_smem_enabled() -> bool:\n')
    # 2) GEMM constructor parameter.
    text = replace(text,
        '        schedule_route_block_factor: int = 1,\n'
        '        whole_tile_tail_splitk: bool = False,\n'
        '    ):\n',
        '        schedule_route_block_factor: int = 1,\n'
        '        whole_tile_tail_splitk: bool = False,\n'
        '        paired_m8_routes: bool = False,\n'
        '    ):\n')
    # 3) GEMM constructor storage and fail-closed validation.
    text = replace(text,
        '        self.cta_m_blocks = int(_covering_count(moe_block_size, 16))\n'
        '        self.uses_m_block_8 = moe_block_size == 8\n'
        '        self.max_m_blocks = int(max_m_blocks)\n',
        '        self.cta_m_blocks = int(_covering_count(moe_block_size, 16))\n'
        '        self.uses_m_block_8 = moe_block_size == 8\n'
        '        # v21 archived R7 paired FC2. The switch may only ever reach\n'
        '        # route-packed M8 GEMMs under the whole-tile schedule with an\n'
        '        # even grouping factor; M8 decode keeps factor 1 and any other\n'
        '        # geometry must reject the switch instead of silently\n'
        '        # reinterpreting the scheduler below.\n'
        '        self.paired_m8_routes = bool(paired_m8_routes)\n'
        '        if self.paired_m8_routes and (\n'
        '            not self.uses_m_block_8\n'
        '            or not self.schedule_whole_tiles\n'
        '            or self.schedule_route_block_factor not in (2, 4)\n'
        '        ):\n'
        '            raise ValueError(\n'
        '                "paired_m8_routes requires M8 whole-tile scheduling "\n'
        '                "with an even schedule_route_block_factor of 2 or 4"\n'
        '            )\n'
        '        self.max_m_blocks = int(max_m_blocks)\n')
    # 4) The paired tile needs two padded 16-row A slabs (archived R7).
    text = replace(text,
        '        self.a_sh_stage = self.a_sh_stride * (16 * self.cta_m_blocks)\n'
        '        self.a_gl_rd_delta_o = 16 * self.cta_k_blocks // 8\n',
        '        self.a_sh_stage = self.a_sh_stride * (16 * self.cta_m_blocks)\n'
        '        if self.paired_m8_routes:\n'
        '            # The M8 ldmatrix mapping consumes a padded 16-row slab:\n'
        '            # rows 8-15 must remain zero.  A paired tile therefore\n'
        '            # needs two independent 16-row slabs even though only\n'
        '            # eight rows in each slab are live.\n'
        '            self.a_sh_stage *= 2\n'
        '        self.a_gl_rd_delta_o = 16 * self.cta_k_blocks // 8\n')
    # 5) Two adjacent M8 route blocks share one 16-row metadata slab.
    text = replace(text,
        '        sh_block_route_indices = self.moe_block_size // 4\n'
        '        sh_rd_block_route_indices = self.moe_block_size // 4\n'
        '        sh_block_topk_weights = self.moe_block_size // 2\n',
        '        # v21 archived R7 paired FC2: the route/rd-route/top-k regions\n'
        '        # double when the pair path is compiled; the valid-count slot\n'
        '        # stays a single region shared by both halves.\n'
        '        route_metadata_rows = self.moe_block_size * (\n'
        '            2 if self.paired_m8_routes else 1\n'
        '        )\n'
        '        sh_block_route_indices = route_metadata_rows // 4\n'
        '        sh_rd_block_route_indices = route_metadata_rows // 4\n'
        '        sh_block_topk_weights = route_metadata_rows // 2\n')
    # 6) Compiled-kernel cache key: the pair changes the A slab, the metadata
    #    regions and therefore the compiled identity and residency target.
    text = replace(text,
        '            self.schedule_route_block_factor,\n'
        '            self.sqg_xor_cheb_t12_smem,\n'
        '            self.small_m_splitk,\n'
        '            self.whole_tile_tail_splitk,\n'
        '        )\n',
        '            self.schedule_route_block_factor,\n'
        '            self.sqg_xor_cheb_t12_smem,\n'
        '            self.small_m_splitk,\n'
        '            self.whole_tile_tail_splitk,\n'
        '            self.paired_m8_routes,\n'
        '        )\n')
    # 7) Staging signature: reintroduce the archived R7 pair operands.
    text = replace(text,
        '        pipe: Int32,\n'
        '        tile_idx: Int32,\n'
        '        block_valid_rows: Int32,\n'
        '        a_gl_stride: Int32,\n',
        '        pipe: Int32,\n'
        '        tile_idx: Int32,\n'
        '        block_valid_rows: Int32,\n'
        '        block_valid_rows1: Int32,\n'
        '        paired_m8: cutlass.Constexpr[bool],\n'
        '        a_gl_stride: Int32,\n')
    # 8) A staging route index: rows 0-7 stage subtile zero, rows 16-23
    #    subtile one, the padded rows stay zero, and each half resolves its
    #    route index from its own metadata rows.
    text = replace(text,
        '        for i in cutlass.range_constexpr(self.a_sh_wr_iters):\n'
        '            row = a_rows_per_iter * Int32(i) + a_gl_rd_row\n'
        '            route_index = Int32(0)\n'
        '            if row < Int32(self.moe_block_size):\n'
        '                route_index = ld_shared_i32_relaxed(\n'
        '                    smem_base\n'
        '                    + Int32(self.sh_rd_route_off * 16)\n'
        '                    + row * Int32(4)\n'
        '                )\n',
        '        for i in cutlass.range_constexpr(self.a_sh_wr_iters):\n'
        '            row = a_rows_per_iter * Int32(i) + a_gl_rd_row\n'
        '            metadata_row = row\n'
        '            route_rows = Int32(self.moe_block_size)\n'
        '            if cutlass.const_expr(paired_m8):\n'
        '                # v21 archived R7 paired FC2: the doubled M8 A slab\n'
        '                # stages rows 0-7 for subtile zero and rows 16-23 for\n'
        '                # subtile one; the padded rows 8-15 and 24-31 must\n'
        '                # stay zero. Each half resolves its route index from\n'
        '                # its own metadata rows.\n'
        '                route_rows = Int32(2 * self.moe_block_size)\n'
        '                metadata_row = Int32(-1)\n'
        '                if row < Int32(self.moe_block_size):\n'
        '                    metadata_row = row\n'
        '                elif row >= Int32(2 * self.moe_block_size) and row < '
        'Int32(\n'
        '                    3 * self.moe_block_size\n'
        '                ):\n'
        '                    metadata_row = row - Int32(self.moe_block_size)\n'
        '            route_index = Int32(0)\n'
        '            if metadata_row >= Int32(0) and metadata_row < route_rows:\n'
        '                route_index = ld_shared_i32_relaxed(\n'
        '                    smem_base\n'
        '                    + Int32(self.sh_rd_route_off * 16)\n'
        '                    + metadata_row * Int32(4)\n'
        '                )\n')
    # 9) A staging valid rows: per-half bounds under the pair.
    text = replace(text,
        '            if cutlass.const_expr(self.has_k_tile_tail):\n'
        '                a_k_int4 = tile_idx * Int32(self.a_gl_rd_delta_o) + '
        'a_gl_rd_col0\n'
        '                if row < block_valid_rows and a_k_int4 < a_gl_stride:\n',
        '            row_valid = row < block_valid_rows\n'
        '            if cutlass.const_expr(paired_m8):\n'
        '                if row < Int32(self.moe_block_size):\n'
        '                    row_valid = row < block_valid_rows\n'
        '                elif row >= Int32(2 * self.moe_block_size) and row < '
        'Int32(\n'
        '                    3 * self.moe_block_size\n'
        '                ):\n'
        '                    row_valid = (\n'
        '                        row - Int32(2 * self.moe_block_size) < '
        'block_valid_rows1\n'
        '                    )\n'
        '                else:\n'
        '                    row_valid = row < Int32(0)\n'
        '            if cutlass.const_expr(self.has_k_tile_tail):\n'
        '                a_k_int4 = tile_idx * Int32(self.a_gl_rd_delta_o) + '
        'a_gl_rd_col0\n'
        '                if row_valid and a_k_int4 < a_gl_stride:\n')
    text = replace(text,
        '            else:\n'
        '                cp_async4_shared_global_pred(\n'
        '                    a_dst,\n'
        '                    a_src,\n'
        '                    (row < block_valid_rows).to(Int32),\n'
        '                )\n',
        '            else:\n'
        '                cp_async4_shared_global_pred(\n'
        '                    a_dst,\n'
        '                    a_src,\n'
        '                    row_valid.to(Int32),\n'
        '                )\n')
    # 10) Prefetch helper signatures.
    text = replace(text,
        '        pipe: cutlass.Constexpr[int],\n'
        '        kk: cutlass.Constexpr[int],\n'
        '        tile_idx: Int32,\n'
        '        k_tiles: Int32,\n'
        '        reduce_k_tile: Int32,\n'
        '        block_valid_rows: Int32,\n'
        '        a_gl_stride: Int32,\n',
        '        pipe: cutlass.Constexpr[int],\n'
        '        kk: cutlass.Constexpr[int],\n'
        '        tile_idx: Int32,\n'
        '        k_tiles: Int32,\n'
        '        reduce_k_tile: Int32,\n'
        '        block_valid_rows: Int32,\n'
        '        block_valid_rows1: Int32,\n'
        '        paired_m8: cutlass.Constexpr[bool],\n'
        '        a_gl_stride: Int32,\n')
    text = replace(text,
        '        tid: Int32,\n'
        '        k_tiles: Int32,\n'
        '        reduce_k_tile: Int32,\n'
        '        block_valid_rows: Int32,\n'
        '        a_gl_stride: Int32,\n',
        '        tid: Int32,\n'
        '        k_tiles: Int32,\n'
        '        reduce_k_tile: Int32,\n'
        '        block_valid_rows: Int32,\n'
        '        block_valid_rows1: Int32,\n'
        '        paired_m8: cutlass.Constexpr[bool],\n'
        '        a_gl_stride: Int32,\n')
    text = replace(text,
        '        pipe: cutlass.Constexpr[int],\n'
        '        tile_idx: Int32,\n'
        '        k_tiles: Int32,\n'
        '        reduce_k_tile: Int32,\n'
        '        block_valid_rows: Int32,\n'
        '        a_gl_stride: Int32,\n',
        '        pipe: cutlass.Constexpr[int],\n'
        '        tile_idx: Int32,\n'
        '        k_tiles: Int32,\n'
        '        reduce_k_tile: Int32,\n'
        '        block_valid_rows: Int32,\n'
        '        block_valid_rows1: Int32,\n'
        '        paired_m8: cutlass.Constexpr[bool],\n'
        '        a_gl_stride: Int32,\n')
    # 11) Prefetch pass-through call sites (one per helper).
    text = replace(text,
        '                pipe,\n'
        '                tile_idx,\n'
        '                k_tiles,\n'
        '                reduce_k_tile,\n'
        '                block_valid_rows,\n'
        '                a_gl_stride,\n',
        '                pipe,\n'
        '                tile_idx,\n'
        '                k_tiles,\n'
        '                reduce_k_tile,\n'
        '                block_valid_rows,\n'
        '                block_valid_rows1,\n'
        '                paired_m8,\n'
        '                a_gl_stride,\n')
    text = replace(text,
        '                    Int32(pipe),\n'
        '                    reduce_k_tile + Int32(pipe),\n'
        '                    block_valid_rows,\n'
        '                    a_gl_stride,\n',
        '                    Int32(pipe),\n'
        '                    reduce_k_tile + Int32(pipe),\n'
        '                    block_valid_rows,\n'
        '                    block_valid_rows1,\n'
        '                    paired_m8,\n'
        '                    a_gl_stride,\n')
    text = replace(text,
        '                Int32((pipe + _STAGES - 1) % _STAGES),\n'
        '                reduce_k_tile + fetch_tile,\n'
        '                block_valid_rows,\n'
        '                a_gl_stride,\n',
        '                Int32((pipe + _STAGES - 1) % _STAGES),\n'
        '                reduce_k_tile + fetch_tile,\n'
        '                block_valid_rows,\n'
        '                block_valid_rows1,\n'
        '                paired_m8,\n'
        '                a_gl_stride,\n')
    # 12) Finish-tile signature: thread the archived R7 metadata row base
    #     through so a paired second half drains its own metadata rows.
    text = replace(text,
        '        output_n_tile: Int32,\n'
        '        block_valid_rows: Int32,\n'
        '        global_scale_f32: cutlass.Float32,\n'
        '        reduce_slice_count: Int32,\n'
        '        reduce_slice_idx: Int32,\n'
        '        lock_slot: Int32,\n'
        '        uses_m_block_8: cutlass.Constexpr[bool],\n'
        '    ):\n'
        '        if cutlass.const_expr(uses_m_block_8):\n'
        '            self._fold_cta_partials_m8(acc0, smem_base, tid)\n',
        '        output_n_tile: Int32,\n'
        '        block_valid_rows: Int32,\n'
        '        metadata_row_base: Int32,\n'
        '        global_scale_f32: cutlass.Float32,\n'
        '        reduce_slice_count: Int32,\n'
        '        reduce_slice_idx: Int32,\n'
        '        lock_slot: Int32,\n'
        '        uses_m_block_8: cutlass.Constexpr[bool],\n'
        '    ):\n'
        '        if cutlass.const_expr(uses_m_block_8):\n'
        '            self._fold_cta_partials_m8(acc0, smem_base, tid)\n')
    text = replace(text,
        '                self._store_tile_m8(\n'
        '                    acc0,\n'
        '                    c_bf16_flat,\n'
        '                    smem_base,\n'
        '                    tid,\n'
        '                    output_n_tile,\n'
        '                    block_valid_rows,\n'
        '                    global_scale_f32,\n'
        '                )\n',
        '                self._store_tile_m8(\n'
        '                    acc0,\n'
        '                    c_bf16_flat,\n'
        '                    smem_base,\n'
        '                    tid,\n'
        '                    output_n_tile,\n'
        '                    block_valid_rows,\n'
        '                    metadata_row_base,\n'
        '                    global_scale_f32,\n'
        '                )\n')
    # 13) Store-tile signature: only the M8 store gains the metadata row base.
    text = replace(text,
        '        self,\n'
        '        acc,\n'
        '        c_bf16_flat: cute.Tensor,\n'
        '        smem_base: Int32,\n'
        '        tid: Int32,\n'
        '        output_n_tile: Int32,\n'
        '        block_valid_rows: Int32,\n'
        '        global_scale_f32: cutlass.Float32,\n'
        '    ):\n'
        '        if cutlass.const_expr(self.has_n_tile_tail):\n',
        '        self,\n'
        '        acc,\n'
        '        c_bf16_flat: cute.Tensor,\n'
        '        smem_base: Int32,\n'
        '        tid: Int32,\n'
        '        output_n_tile: Int32,\n'
        '        block_valid_rows: Int32,\n'
        '        metadata_row_base: Int32,\n'
        '        global_scale_f32: cutlass.Float32,\n'
        '    ):\n'
        '        if cutlass.const_expr(self.has_n_tile_tail):\n')
    text = replace(text,
        '        store_iters = _covering_count(16, self.cta_threads // '
        '(2 * self.cta_n_blocks))\n'
        '        if cutlass.const_expr(self.has_n_tile_tail):\n'
        '            self._drain_output_smem_tail(\n'
        '                c_bf16_flat,\n'
        '                smem_base,\n'
        '                c_gl_stride,\n'
        '                c_gl_stride_covered,\n'
        '                c_gl_wr,\n'
        '                c_gl_wr_delta,\n'
        '                c_sh_rd,\n'
        '                c_sh_rd_delta,\n'
        '                block_valid_rows,\n'
        '                store_iters,\n'
        '            )\n',
        '        store_iters = _covering_count(16, self.cta_threads // '
        '(2 * self.cta_n_blocks))\n'
        '        if cutlass.const_expr(self.has_n_tile_tail):\n'
        '            self._drain_output_smem_tail(\n'
        '                c_bf16_flat,\n'
        '                smem_base,\n'
        '                c_gl_stride,\n'
        '                c_gl_stride_covered,\n'
        '                c_gl_wr,\n'
        '                c_gl_wr_delta,\n'
        '                c_sh_rd,\n'
        '                c_sh_rd_delta,\n'
        '                block_valid_rows,\n'
        '                metadata_row_base,\n'
        '                store_iters,\n'
        '            )\n')
    text = replace(text,
        '                metadata_row_base,\n'
        '                store_iters,\n'
        '            )\n'
        '        else:\n'
        '            self._drain_output_smem(\n'
        '                c_bf16_flat,\n'
        '                smem_base,\n'
        '                c_gl_stride,\n'
        '                c_gl_wr,\n'
        '                c_gl_wr_delta,\n'
        '                c_sh_rd,\n'
        '                c_sh_rd_delta,\n'
        '                block_valid_rows,\n'
        '                store_iters,\n'
        '            )\n',
        '                metadata_row_base,\n'
        '                store_iters,\n'
        '            )\n'
        '        else:\n'
        '            self._drain_output_smem(\n'
        '                c_bf16_flat,\n'
        '                smem_base,\n'
        '                c_gl_stride,\n'
        '                c_gl_wr,\n'
        '                c_gl_wr_delta,\n'
        '                c_sh_rd,\n'
        '                c_sh_rd_delta,\n'
        '                block_valid_rows,\n'
        '                metadata_row_base,\n'
        '                store_iters,\n'
        '            )\n')
    # 14) Drain signatures and metadata reads (archived R7 semantics).
    text = replace(text,
        '        c_sh_rd: Int32,\n'
        '        c_sh_rd_delta: Int32,\n'
        '        block_valid_rows: Int32,\n'
        '        store_iters: cutlass.Constexpr[int],\n'
        '    ):\n'
        '        for _ in cutlass.range_constexpr(store_iters):\n'
        '            row = c_gl_wr // c_gl_stride_covered\n',
        '        c_sh_rd: Int32,\n'
        '        c_sh_rd_delta: Int32,\n'
        '        block_valid_rows: Int32,\n'
        '        metadata_row_base: Int32,\n'
        '        store_iters: cutlass.Constexpr[int],\n'
        '    ):\n'
        '        for _ in cutlass.range_constexpr(store_iters):\n'
        '            row = c_gl_wr // c_gl_stride_covered\n')
    text = replace(text,
        '        c_sh_rd: Int32,\n'
        '        c_sh_rd_delta: Int32,\n'
        '        block_valid_rows: Int32,\n'
        '        store_iters: cutlass.Constexpr[int],\n'
        '    ):\n'
        '        for _ in cutlass.range_constexpr(store_iters):\n'
        '            row = c_gl_wr // c_gl_stride\n',
        '        c_sh_rd: Int32,\n'
        '        c_sh_rd_delta: Int32,\n'
        '        block_valid_rows: Int32,\n'
        '        metadata_row_base: Int32,\n'
        '        store_iters: cutlass.Constexpr[int],\n'
        '    ):\n'
        '        for _ in cutlass.range_constexpr(store_iters):\n'
        '            row = c_gl_wr // c_gl_stride\n')
    text = replace(text,
        '            if row < block_valid_rows:\n'
        '                route_index = ld_shared_i32_relaxed(\n'
        '                    smem_base + Int32(self.sh_route_off * 16) + row * '
        'Int32(4)\n'
        '                )\n',
        '            if row < block_valid_rows:\n'
        '                metadata_row = metadata_row_base + row\n'
        '                route_index = ld_shared_i32_relaxed(\n'
        '                    smem_base + Int32(self.sh_route_off * 16)\n'
        '                    + metadata_row * Int32(4)\n'
        '                )\n')
    text = replace(text,
        '            if row < block_valid_rows and col_word < c_gl_stride:\n'
        '                route_index = ld_shared_i32_relaxed(\n'
        '                    smem_base + Int32(self.sh_route_off * 16) + row * '
        'Int32(4)\n'
        '                )\n',
        '            if row < block_valid_rows and col_word < c_gl_stride:\n'
        '                metadata_row = metadata_row_base + row\n'
        '                route_index = ld_shared_i32_relaxed(\n'
        '                    smem_base + Int32(self.sh_route_off * 16)\n'
        '                    + metadata_row * Int32(4)\n'
        '                )\n')
    text = replace(text,
        '                if cutlass.const_expr(self.mul_topk_weights):\n'
        '                    scale_bf2 = ld_shared_u32(\n'
        '                        smem_base\n'
        '                        + Int32(self.sh_topk_off * 16)\n'
        '                        + row * Int32(4)\n'
        '                    )\n',
        '                if cutlass.const_expr(self.mul_topk_weights):\n'
        '                    scale_bf2 = ld_shared_u32(\n'
        '                        smem_base\n'
        '                        + Int32(self.sh_topk_off * 16)\n'
        '                        + metadata_row * Int32(4)\n'
        '                    )\n', count=2)
    # 15) Existing finish-tile callers keep the stock metadata row base.
    text = replace(text,
        '        self._finish_tile(\n'
        '            acc,\n'
        '            acc,\n'
        '            acc,\n'
        '            acc,\n'
        '            c_bf16_flat,\n'
        '            c_tmp_f32_flat,\n'
        '            locks_i32_flat,\n'
        '            smem_base,\n'
        '            tid,\n'
        '            output_n_tile,\n'
        '            block_valid_rows,\n'
        '            global_scale_f32,\n',
        '        self._finish_tile(\n'
        '            acc,\n'
        '            acc,\n'
        '            acc,\n'
        '            acc,\n'
        '            c_bf16_flat,\n'
        '            c_tmp_f32_flat,\n'
        '            locks_i32_flat,\n'
        '            smem_base,\n'
        '            tid,\n'
        '            output_n_tile,\n'
        '            block_valid_rows,\n'
        '            Int32(0),\n'
        '            global_scale_f32,\n')
    text = replace(text,
        '        self._finish_tile(\n'
        '            acc0,\n'
        '            acc1,\n'
        '            acc2,\n'
        '            acc3,\n'
        '            c_bf16_flat,\n'
        '            c_tmp_f32_flat,\n'
        '            locks_i32_flat,\n'
        '            smem_base,\n'
        '            tid,\n'
        '            output_n_tile,\n'
        '            block_valid_rows,\n'
        '            global_scale_f32,\n',
        '        self._finish_tile(\n'
        '            acc0,\n'
        '            acc1,\n'
        '            acc2,\n'
        '            acc3,\n'
        '            c_bf16_flat,\n'
        '            c_tmp_f32_flat,\n'
        '            locks_i32_flat,\n'
        '            smem_base,\n'
        '            tid,\n'
        '            output_n_tile,\n'
        '            block_valid_rows,\n'
        '            Int32(0),\n'
        '            global_scale_f32,\n')
    # 16) Archived R7 pair functions: metadata pair read, prologue, tile,
    #     MMA pipeline and fragment bundle loader.
    text = replace(text,
        '        cute.arch.sync_threads()\n'
        '        valid_count = ld_shared_i32_relaxed(\n'
        '            smem_base + Int32(self.sh_valid_count_off * 16)\n'
        '        )\n'
        '        cute.arch.sync_threads()\n'
        '        return valid_count\n'
        '\n'
        '    @cute.jit\n'
        '    def _run_tile(\n',
        '        cute.arch.sync_threads()\n'
        '        valid_count = ld_shared_i32_relaxed(\n'
        '            smem_base + Int32(self.sh_valid_count_off * 16)\n'
        '        )\n'
        '        cute.arch.sync_threads()\n'
        '        return valid_count\n'
        '\n'
        + PAIR_READ_BLOCK +
        '    @cute.jit\n'
        '    def _run_tile(\n')
    text = replace(text,
        '            Int32(0),\n'
        '            global_scale_f32,\n'
        '            reduce_slice_count,\n'
        '            reduce_slice_idx,\n'
        '            lock_slot,\n'
        '            True,\n'
        '        )\n'
        '\n'
        '    @cute.jit\n'
        '    def _run_tile_large_m(\n',
        '            Int32(0),\n'
        '            global_scale_f32,\n'
        '            reduce_slice_count,\n'
        '            reduce_slice_idx,\n'
        '            lock_slot,\n'
        '            True,\n'
        '        )\n'
        '\n'
        + PAIR_PROLOGUE_AND_TILE +
        '    @cute.jit\n'
        '    def _run_tile_large_m(\n')
    text = replace(text,
        '                self._load_a_register_bundle(\n'
        '                    a_regs_next,\n'
        '                    smem_base,\n'
        '                    a_sh_rd,\n'
        '                    Int32((pipe + 1) % _STAGES),\n'
        '                    Int32(0),\n'
        '                    uses_m_block_8,\n'
        '                )\n'
        '\n'
        '    @cute.jit\n'
        '    def _scaled_dequant_b_fragment(self, frag: cute.Tensor, q: Uint32, '
        's: Uint32):\n',
        '                self._load_a_register_bundle(\n'
        '                    a_regs_next,\n'
        '                    smem_base,\n'
        '                    a_sh_rd,\n'
        '                    Int32((pipe + 1) % _STAGES),\n'
        '                    Int32(0),\n'
        '                    uses_m_block_8,\n'
        '                )\n'
        '\n'
        + PAIR_BUNDLE_LOADER +
        '    @cute.jit\n'
        '    def _scaled_dequant_b_fragment(self, frag: cute.Tensor, q: Uint32, '
        's: Uint32):\n')
    # 17) Fused kernel resolves the switch for FC2 only, guarded by the
    #     schedule it actually picked.
    text = replace(text,
        '        self.fc1_tail_splitk = (\n'
        '            _gb10_fc1_tail_splitk_enabled()\n'
        '            and not self.small_m_splitk\n'
        '            and self.schedule_whole_tiles\n'
        '            and self.moe_block_size == 8\n'
        '            and not self.direct_topk_routes\n'
        '        )\n',
        '        self.fc1_tail_splitk = (\n'
        '            _gb10_fc1_tail_splitk_enabled()\n'
        '            and not self.small_m_splitk\n'
        '            and self.schedule_whole_tiles\n'
        '            and self.moe_block_size == 8\n'
        '            and not self.direct_topk_routes\n'
        '        )\n'
        '        # v21 archived R7 paired FC2: grouped M8 prefill plans may\n'
        '        # decode one weight fragment per pair of M8 subtiles. M8\n'
        '        # decode keeps factor 1 and never reaches the pair path; the\n'
        '        # global stripe switch above stays exclusive with this one.\n'
        '        self.fc2_paired_m8_routes = (\n'
        '            _gb10_fc2_m8_pair_enabled()\n'
        '            and not self.small_m_splitk\n'
        '            and self.schedule_whole_tiles\n'
        '            and self.fc2_moe_block_size == 8\n'
        '            and self.fc2_schedule_route_block_factor in (2, 4)\n'
        '            and not self.direct_topk_routes\n'
        '        )\n')
    # 18) Only the FC2 GEMM constructor receives the switch.
    text = replace(text,
        '            schedule_whole_tiles=self.schedule_whole_tiles,\n'
        '            dynamic_num_experts=self.dynamic_num_experts,\n'
        '            schedule_route_block_factor=self.fc2_schedule_route_block_factor,\n'
        '        )\n',
        '            schedule_whole_tiles=self.schedule_whole_tiles,\n'
        '            dynamic_num_experts=self.dynamic_num_experts,\n'
        '            schedule_route_block_factor=self.fc2_schedule_route_block_factor,\n'
        '            paired_m8_routes=self.fc2_paired_m8_routes,\n'
        '        )\n')
    return text


DISPATCH_ANCHOR = '''        factor = gemm.schedule_route_block_factor
        first_route_block = route_block_idx * Int32(factor)
        first_lock_slot = lock_slot * Int32(factor)
        for subtile in cutlass.range_constexpr(factor):
            gemm._run_tile(
'''

DISPATCH_NEW = '''        factor = gemm.schedule_route_block_factor
        first_route_block = route_block_idx * Int32(factor)
        first_lock_slot = lock_slot * Int32(factor)
        # v21 archived R7 paired FC2: one pair call covers two adjacent M8
        # subtiles with a single B stream. Production group4 becomes two pair
        # calls; lock slots stay consecutive so every subtile keeps exactly
        # one slot in both arms. Odd factors cannot arise from the supported
        # grouping values and are rejected at construction.
        if cutlass.const_expr(gemm.paired_m8_routes):
            if cutlass.const_expr(factor == 2):
                gemm._run_tile_m8_pair(
                    a_flat,
                    a_alt_flat,
                    b_flat,
                    c_flat,
                    scales_flat,
                    global_scale,
                    packed_route_indices,
                    topk_weights,
                    c_tmp,
                    locks,
                    trellis_lut_addr,
                    smem_base,
                    tid,
                    first_route_block,
                    local_expert,
                    output_n_tile,
                    reduce_k_tile,
                    reduce_tile_count,
                    reduce_slice_count,
                    reduce_slice_idx,
                    first_lock_slot,
                    active_size_m,
                )
            else:
                for pair_idx in cutlass.range_constexpr(factor // 2):
                    gemm._run_tile_m8_pair(
                        a_flat,
                        a_alt_flat,
                        b_flat,
                        c_flat,
                        scales_flat,
                        global_scale,
                        packed_route_indices,
                        topk_weights,
                        c_tmp,
                        locks,
                        trellis_lut_addr,
                        smem_base,
                        tid,
                        first_route_block + Int32(2 * pair_idx),
                        local_expert,
                        output_n_tile,
                        reduce_k_tile,
                        reduce_tile_count,
                        reduce_slice_count,
                        reduce_slice_idx,
                        first_lock_slot + Int32(2 * pair_idx),
                        active_size_m,
                    )
        else:
            for subtile in cutlass.range_constexpr(factor):
                gemm._run_tile(
'''

PAIR_CONTRACT_ANCHOR = '''                "mixed Trellis FC2 schedule factor must divide one packed "
                f"route block: factor={fc2_factor}, maximum={expected_factor}"
            )
'''

PAIR_CONTRACT_NEW = '''                "mixed Trellis FC2 schedule factor must divide one packed "
                f"route block: factor={fc2_factor}, maximum={expected_factor}"
            )
        expected_pair = fc2_factor in (2, 4) and driver.fc2.moe_block_size == 8
        if bool(driver.fc2.paired_m8_routes) != expected_pair:
            raise ValueError(
                "mixed Trellis FC2 pair contract mismatch: "
                f"factor={fc2_factor}, m={driver.fc2.moe_block_size}, "
                f"paired={driver.fc2.paired_m8_routes}"
            )
'''


def transform_mixed(text):
    text = replace(text, DISPATCH_ANCHOR, DISPATCH_NEW)
    text = replace(text, PAIR_CONTRACT_ANCHOR, PAIR_CONTRACT_NEW, count=2)
    return text


def patch(root, check=False):
    results = {}
    for rel, transform_fn in ((KERNEL, transform_kernel), (MIXED, transform_mixed)):
        path = Path(root) / rel
        source = path.read_text(encoding='utf-8')
        digest = hashlib.sha256(source.encode()).hexdigest()
        if digest != OUTPUTS[rel]:
            if check or digest != INPUTS[rel]:
                raise RuntimeError(f'unexpected v21 source: {path} ({digest})')
            result = transform_fn(source)
            compile(result, str(path), 'exec')
            results[rel] = hashlib.sha256(result.encode()).hexdigest()
            path.write_text(result, encoding='utf-8', newline='\n')
    print(f'{VERSION}: verified {root}')
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    patch(args.root, args.check)
