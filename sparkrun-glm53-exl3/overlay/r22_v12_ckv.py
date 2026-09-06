# Inserted into the pinned B12X MLA backend by patch_r22_v12.py.
# torch, triton and tl are imported by that backend.

@triton.jit
def _v12_ckv_metadata_kernel(
    req_ptr, ids_ptr, starts_ptr, lens_ptr, seq_ptr, qsl_ptr,
    out_ptr, count_ptr, causal_ptr,
    ids_s0, ids_s1, starts_s0, starts_s1, lens_s0, lens_s1,
    out_s0, out_s1, padded_tokens,
    WORLD: tl.constexpr, INTERLEAVE: tl.constexpr,
    WIDTH: tl.constexpr, BLOCK: tl.constexpr,
):
    # One CTA owns the entire row. No atomic compaction, initialization
    # launches, or temporary tensors for per-token causal lengths.
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    req = tl.load(req_ptr + row)
    end = tl.load(qsl_ptr + req + 1)
    causal = tl.load(seq_ptr + req) - end + row + 1
    tok = tl.load(ids_ptr + row * ids_s0 + cols * ids_s1,
                  mask=cols < WIDTH, other=-1)
    owner = (tok // INTERLEAVE) % WORLD
    local = (tok // (WORLD * INTERLEAVE)) * INTERLEAVE + tok % INTERLEAVE
    valid_tok = (cols < WIDTH) & (tok >= 0)
    start = tl.load(starts_ptr + owner * starts_s0 + req * starts_s1,
                    mask=valid_tok, other=0)
    length = tl.load(lens_ptr + owner * lens_s0 + req * lens_s1,
                     mask=valid_tok, other=0)
    valid = valid_tok & (local >= 0) & (local < length)
    offset = tl.cumsum(valid.to(tl.int32)) - 1
    count = tl.minimum(tl.sum(valid.to(tl.int32)), causal)
    # Tail and compacted output addresses are disjoint, including count=0.
    tl.store(out_ptr + row * out_s0 + cols * out_s1, -1,
             mask=(cols < WIDTH) & (cols >= count))
    tl.store(out_ptr + row * out_s0 + offset * out_s1,
             owner * padded_tokens + start + local,
             mask=valid & (offset < count))
    tl.store(count_ptr + row, count)
    tl.store(causal_ptr + row, causal)


def _v12_prepare_ckv_metadata(
    req_ids, token_indices, rank_starts, rank_lens, global_lens,
    query_start_loc, out, counts, causal, *, dcp_size,
    interleave, padded_tokens,
):
    if token_indices.shape != out.shape or token_indices.ndim != 2:
        raise ValueError("v12 CKV output must match the rank-2 top-k input")
    rows, width = token_indices.shape
    if not 0 < width <= 4096 or dcp_size <= 0 or interleave <= 0:
        raise ValueError("unsupported v12 CKV geometry")
    if rank_starts.shape != rank_lens.shape or rank_starts.shape[0] != dcp_size:
        raise ValueError("v12 CKV rank metadata mismatch")
    if any(t.dtype != torch.int32 for t in
           (req_ids, token_indices, rank_starts, rank_lens, global_lens,
            query_start_loc, out, counts, causal)):
        raise TypeError("v12 CKV metadata must be int32")
    if any(t.ndim != 1 or not t.is_contiguous() for t in
           (req_ids, global_lens, query_start_loc, counts, causal)):
        raise ValueError("v12 CKV vectors must be contiguous")
    if req_ids.numel() < rows or counts.numel() < rows or causal.numel() < rows:
        raise ValueError("v12 CKV vectors are too short")
    if global_lens.numel() != rank_starts.shape[1] or query_start_loc.numel() < global_lens.numel() + 1:
        raise ValueError("v12 CKV request geometry mismatch")
    if rows:
        _v12_ckv_metadata_kernel[(rows,)](
            req_ids, token_indices, rank_starts, rank_lens, global_lens,
            query_start_loc, out, counts, causal,
            *token_indices.stride(), *rank_starts.stride(), *rank_lens.stride(),
            *out.stride(), padded_tokens,
            WORLD=dcp_size, INTERLEAVE=interleave, WIDTH=width,
            BLOCK=triton.next_power_of_2(width), num_warps=4,
        )


def _v12_can_borrow_query(q, scratch, enabled, rows, heads, head_dim):
    # The caller consumes the output before another layer reuses its buffers.
    # The sparse-MLA kernel reads q; it never writes through this alias.
    if not (enabled and q.dtype == torch.bfloat16
            and tuple(q.shape) == (rows, heads, head_dim)
            and q.is_contiguous() and q.data_ptr() % 16 == 0):
        return False
    q_begin, scratch_begin = q.data_ptr(), scratch.data_ptr()
    q_end = q_begin + q.numel() * q.element_size()
    scratch_end = scratch_begin + scratch.numel() * scratch.element_size()
    return q_end <= scratch_begin or scratch_end <= q_begin
