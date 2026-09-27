# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from pinned tests/v1/attention/test_b12x_v41_ced_metadata.py.
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from vllm.models.deepseek_v4_1.ced import CEDState, gather_rows

def _state(lengths, seq_lens=None, full=None, prefix=None):
    lengths = np.asarray(lengths, dtype=np.int32)
    nr = len(lengths)
    starts = torch.tensor(np.r_[0, lengths.cumsum()], dtype=torch.int32, device='cuda')
    seq = torch.tensor(lengths if seq_lens is None else seq_lens, dtype=torch.int32, device='cuda')
    state = CEDState(4, 8192, (128, 256, 384, 512, 8192), 'cuda')
    state.stage(starts, seq, lengths, full or [False] * nr, prefix or [0] * nr)
    return (state, starts, seq)

@pytest.mark.parametrize("skip, graph", [(0, 0), (1, 0), (0, 1), (1, 1)])
def test_dspark_compact_context_preserves_query_anchor_and_rejections(monkeypatch, default_vllm_config, skip, graph):
    monkeypatch.setenv("DS41_SKIP_PREFILL_DRAFT", str(skip))
    monkeypatch.setenv("DS41_COMPACT_CONTEXT_GRAPH", str(graph))
    from vllm.v1.attention.backends.utils import PAD_SLOT_ID
    from vllm.v1.worker.gpu.input_batch import InputBuffers
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import prepare_dflash_inputs
    device = torch.device('cuda')
    state, starts, seq = _state([1, 4096], [11, 4096])
    target_positions = torch.cat((torch.tensor([10], device=device), torch.arange(4096, device=device)))
    batch = SimpleNamespace(num_reqs=2, num_scheduled_tokens=np.array([1, 4096]), positions=target_positions, query_start_loc=starts, idx_mapping=torch.arange(2, device=device))
    buffers = InputBuffers(4, 8192, device)
    slots = torch.empty(8192, dtype=torch.int64, device=device)
    context_positions = torch.empty_like(slots)
    context_slots = torch.empty_like(slots)
    sample_indices = torch.empty(8, dtype=torch.int64, device=device)
    sample_pos = torch.empty_like(sample_indices)
    sample_reqs = torch.empty(8, dtype=torch.int32, device=device)
    temperature = torch.ones(4, device=device)
    seeds = torch.zeros(4, dtype=torch.int64, device=device)
    table = torch.arange(1, 4 * 40 + 1, dtype=torch.int32, device=device).view(4, 40)
    table[1, 31] = 0
    prepare_dflash_inputs(buffers, slots, context_positions, context_slots, sample_indices, sample_pos, sample_reqs, temperature, seeds, batch, torch.tensor([1, 1], device=device), torch.tensor([0, 2], device=device), torch.tensor([77, 88, 0, 0], device=device), torch.zeros(4, device=device), temperature, seeds, table, 128, 0, 1, 1, 999, 2, 2, 4, 8192, 8192, True)
    indices = state.get_indices()
    packed_positions = gather_rows(context_positions, indices)
    packed_slots = gather_rows(context_slots, indices).masked_fill(indices < 0, -1)
    assert buffers.positions[:4].tolist() == [11, 12, 4094, 4095]
    assert buffers.input_ids[:4].tolist() == [77, 999, 88, 999]
    assert sample_pos[:4].tolist() == [12, 13, 4095, 4096]
    assert packed_positions[:3].tolist() == [10, 3968, 3969]
    assert packed_positions[127:129].tolist() == [0, 0]
    assert packed_slots[0].item() == 138
    assert torch.all(packed_slots[1:] == -1)
    pool = torch.full((1024,), -7.0, device=device)
    writable = packed_slots >= 0
    pool[packed_slots[writable]] = 5
    assert pool[0].item() == -7
    assert torch.count_nonzero(pool != -7).item() == 1
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.spec_decode import speculator as base_speculator
    from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dflash
    table[1, 31] = 72
    context_pool = torch.full((25600, 2), -7.0, device=device)
    aux = torch.arange(4097 * 2, dtype=torch.float32, device=device).view(4097, 2)
    combine_rows = []

    def combine(hidden):
        combine_rows.append(hidden.shape[0])
        return hidden + 3

    def project(hidden, positions, context_slot_mapping):
        valid = context_slot_mapping >= 0
        context_pool[context_slot_mapping[valid]] = hidden[valid] + positions[valid, None]
    proposer = dflash.DFlashSpeculator.__new__(dflash.DFlashSpeculator)
    proposer.model = SimpleNamespace(combine_hidden_states=combine, precompute_and_store_context_kv=project)
    proposer.model_state = SimpleNamespace(get_ced_indices=state.get_indices, prepare_draft_attn_metadata=lambda **_: None)
    proposer._speculator_name = "DSpark"
    proposer.vllm_config = default_vllm_config
    proposer.model.capture_context_preparation = lambda: None
    proposer.hidden_states = torch.zeros((8192, 2), device=device)
    proposer.context_positions = context_positions
    proposer._context_slot_mappings = context_slots[None]
    proposer._context_preparer = SimpleNamespace(can_run=lambda count: pytest.fail('Compact prefill entered decode context graph'))
    proposer._layer_group_idx = None
    proposer.draft_kv_cache_group_id = 0
    proposer.draft_kv_cache_group_ids = [0]
    proposer.block_tables = SimpleNamespace(cp_size=1, get_group_cp_parameters=lambda gid: (0, 1, 1), slot_mappings=slots[None], input_block_tables=[table], kernel_block_sizes=[128])
    proposer.input_buffers = buffers
    proposer.sample_indices = sample_indices
    proposer.sample_pos = sample_pos
    proposer.sample_idx_mapping = sample_reqs
    proposer.temperature = temperature
    proposer.seeds = seeds
    proposer.parallel_drafting_token_id = 999
    proposer.num_query_per_req = proposer.num_speculative_steps = 2
    proposer.max_num_reqs = 4
    proposer.max_num_tokens = proposer.max_model_len = 8192
    proposer.sample_from_anchor = True
    proposer.dp_size = 1
    proposer.dp_rank = 0
    proposer.pcp_manager = None
    proposer.query_cudagraph_manager = None
    proposer.draft_attn_layer_names = ['draft']
    proposer.draft_cp_size = 1
    proposer.kv_cache_config = None
    proposer._group_causal = False
    proposer.arange = torch.arange(5, dtype=torch.int32)
    proposer.arange_np = np.arange(5, dtype=np.int32)
    proposer.draft_is_prefilling = torch.zeros(4, dtype=torch.bool)
    proposer.idx_mapping = batch.idx_mapping
    proposer.attn_groups = []

    def check_draft_boundaries(**metadata):
        torch.testing.assert_close(metadata['query_start_loc_cpu'], metadata['query_start_loc_gpu'].cpu(), rtol=0, atol=0)
        assert metadata['max_query_len'] == 2
        return None
    monkeypatch.setattr(base_speculator, 'build_attn_metadata', check_draft_boundaries)
    proposer._prepare_eplb_forward = lambda count: None
    proposer._generate_draft = lambda *args, **kwargs: None
    proposer.draft_tokens = torch.zeros((4, 2), device=device, dtype=torch.int64)
    batch.has_prefill = True
    batch.is_prefilling_np = np.array([False, True])
    batch.num_computed_prefill_tokens_np = np.array([10, 0])
    batch.prefill_len_np = np.array([10, 4096])
    batch.num_draft_tokens = 0
    batch.has_structured_output_reqs = False
    batch.num_tokens = 4097
    batch.seq_lens_cpu_upper_bound = seq.cpu()
    monkeypatch.setattr(dflash, 'dispatch_cg_and_sync_dp', lambda *args, **kwargs: (SimpleNamespace(num_reqs=2, num_tokens=4, cg_mode=CUDAGraphMode.NONE), None))
    monkeypatch.setattr(dflash, 'build_slot_mappings_by_layer', lambda *args: None)
    arguments = dict(input_batch=batch, attn_metadata={}, slot_mappings={}, last_hidden_states=aux, aux_hidden_states=[aux], num_sampled=torch.tensor([1, 1], device=device), num_rejected=torch.tensor([0, 2], device=device), last_sampled=torch.tensor([77, 88, 0, 0], device=device), next_prefill_tokens=torch.zeros(4, device=device), temperature=temperature, seeds=seeds)
    query_capacity = proposer.max_num_reqs * proposer.num_query_per_req
    slots[query_capacity:].fill_(-3137)
    proposer.propose(**arguments)
    assert combine_rows == [129]
    expected_pool = torch.full_like(context_pool, -7)
    expected_pool[138] = aux[0] + 3 + 10
    expected_pool[72 * 128:72 * 128 + 126] = aux[3969:4095] + 3 + torch.arange(3968, 4094, device=device)[:, None]
    torch.testing.assert_close(context_pool, expected_pool)
    assert buffers.positions[:4].tolist() == [11, 12, 4094, 4095]
    assert torch.all(slots[4:query_capacity] == PAD_SLOT_ID)
    assert torch.all(slots[query_capacity:] == -3137)
    proposer.propose(**arguments, context_kv_is_restored=True)
    assert combine_rows == [129]
    torch.testing.assert_close(context_pool, expected_pool)

