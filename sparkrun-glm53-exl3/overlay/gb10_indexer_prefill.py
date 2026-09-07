"""Compact metadata for the single-request, non-compressed B12X paged indexer.

CPU sequence lengths are upper bounds for page-table sizing only. Causal lengths
always come from live device metadata, including async speculative correction.
"""
from dataclasses import dataclass

import torch
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerPrefillChunkMetadata,
    build_prefill_chunk_metadata,
)


def local_count(length, rank, world, interleave):
    base = length // (world * interleave) * interleave
    return base + min(max(length - base * world - rank * interleave, 0), interleave)


@triton.jit
def _causal_metadata(qsl, seq, starts, ends, global_cu, local_cu,
                     source_table, page_table, table_width,
                     req, query_offset, rows, rank: tl.constexpr,
                     world: tl.constexpr, interleave: tl.constexpr,
                     BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    query_len = tl.load(qsl + req + 1) - tl.load(qsl + req)
    total = tl.load(seq + req)
    total_base = total // (world * interleave) * interleave
    total_local = total_base + tl.minimum(tl.maximum(total - total_base * world - rank * interleave, 0), interleave)
    context = total - query_len + query_offset + offsets + 1
    base = context // (world * interleave) * interleave
    local = base + tl.minimum(tl.maximum(context - base * world - rank * interleave, 0), interleave)
    tl.store(starts + offsets, 0, offsets < rows)
    tl.store(ends + offsets, local, offsets < rows)
    # CPU lengths may overestimate live device lengths after speculation.
    # Never expose an unallocated trailing page to B12X's K gather.
    page_ids = tl.load(source_table + offsets,
        mask=(offsets < table_width) & (offsets < (total_local + 63) // 64), other=-1)
    tl.store(page_table + offsets, page_ids, offsets < table_width)
    if tl.program_id(0) == 0:
        tl.store(global_cu, 0)
        tl.store(global_cu + 1, total)
        tl.store(local_cu, 0)
        tl.store(local_cu + 1, total_local)


@dataclass
class PagedPrefillChunk(DeepseekV32IndexerPrefillChunkMetadata):
    b12x_seq_lens: torch.Tensor | None = None


def build_paged_chunk(start_idx, end_idx, query_start_loc, query_start_loc_cpu,
                      uncompressed_seq_lens, compressed_seq_lens,
                      compressed_seq_lens_cpu, block_table, compress_ratio,
                      query_slice=None, skip_kv_gather=False, dcp_rank=0,
                      dcp_world_size=1, cp_kv_cache_interleave_size=1):
    if end_idx != start_idx + 1 or compress_ratio != 1:
        return build_prefill_chunk_metadata(
            start_idx, end_idx, query_start_loc, query_start_loc_cpu,
            uncompressed_seq_lens, compressed_seq_lens, compressed_seq_lens_cpu,
            block_table, compress_ratio, query_slice, skip_kv_gather, dcp_rank,
            dcp_world_size, cp_kv_cache_interleave_size)
    total = int(compressed_seq_lens_cpu[start_idx].item())
    if total == 0:
        return None
    token_base = int(query_start_loc_cpu[start_idx].item())
    query_len = int(query_start_loc_cpu[end_idx].item()) - token_base
    begin, end = (0, query_len) if query_slice is None else (query_slice.start, query_slice.stop)
    if not 0 <= begin < end <= query_len:
        raise ValueError('invalid B12X prefill query slice')
    if dcp_world_size < 1 or not 0 <= dcp_rank < dcp_world_size or cp_kv_cache_interleave_size < 1:
        raise ValueError('invalid B12X prefill DCP geometry')
    rows = end - begin
    device = block_table.device
    starts = torch.empty(rows, dtype=torch.int32, device=device)
    ends = torch.empty_like(starts)
    global_cu = torch.empty(2, dtype=torch.int32, device=device)
    local_cu = torch.empty_like(global_cu)
    local_upper = local_count(total, dcp_rank, dcp_world_size, cp_kv_cache_interleave_size)
    width = min(block_table.shape[1], max(1, (local_upper + 63) // 64))
    pages = torch.empty((1, width), dtype=torch.int32, device=device)
    _causal_metadata[(triton.cdiv(max(rows, width), 256),)](
        query_start_loc, uncompressed_seq_lens, starts, ends, global_cu, local_cu,
        block_table[start_idx], pages, width,
        start_idx, begin, rows, dcp_rank, dcp_world_size, cp_kv_cache_interleave_size, 256)
    return PagedPrefillChunk(
        block_table=pages,
        cu_seqlen_ks=starts, cu_seqlen_ke=ends, cu_seq_lens=global_cu,
        # Paged B12X reads the original page table; it never gathers through
        # the generic indexer's O(context) token-to-request map.
        token_to_seq=torch.empty(0, dtype=torch.int32, device=device),
        total_seq_lens=total, token_start=token_base + begin,
        token_end=token_base + end, num_reqs=1,
        skip_kv_gather=skip_kv_gather or begin > 0,
        local_cu_seq_lens=local_cu,
        local_total_seq_lens=local_upper,
        max_local_total_seq_lens=local_count(total, 0, dcp_world_size, cp_kv_cache_interleave_size),
        b12x_seq_lens=ends,
    )
