# SPDX-License-Identifier: Apache-2.0
"""Clear adjacent native MoE barriers with one fill, preserving other scratch.

Independent implementation of the barrier-zero opportunity documented in
knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4 at e9ec61d. Its adapter code is
not imported. Layout assumptions are from b12x d44247b _allocate_arena_tensor
and _TPCoreWorkspacePlan: barrier_count immediately precedes barrier_epoch,
and every scratch allocation starts on at least a 16-byte boundary.
"""
import os

import torch


def clear_barriers(count, epoch):
    compatible = (
        os.environ.get('DS41_MOE_COALESCE_BARRIERS', '0') == '1'
        and count.dtype == torch.int32 and epoch.dtype == torch.int32
        and count.device == epoch.device
        and count.is_contiguous() and epoch.is_contiguous()
        and count.numel() > 0 and epoch.numel() > 0
        and count.untyped_storage().data_ptr() == epoch.untyped_storage().data_ptr()
    )
    if compatible:
        end = count.storage_offset() + count.numel()
        # Alignment is four int32 elements. No other arena allocation can fit
        # inside the alignment padding between these consecutive allocations.
        expected_epoch = ((end + 3) // 4) * 4
        if epoch.storage_offset() == expected_epoch:
            length = expected_epoch + epoch.numel() - count.storage_offset()
            count.as_strided((length,), (1,), count.storage_offset()).zero_()
            return
    count.zero_()
    epoch.zero_()
