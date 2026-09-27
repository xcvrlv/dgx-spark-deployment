# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from local-inference-lab/vllm 1794dcf tests/models/test_dspark_mla.py.
# Uses native b12x projections, rotary and cache writes; no full model required.
from types import SimpleNamespace
import pytest
import torch
import torch.nn as nn

@pytest.mark.parametrize("width", [128, 5120])
@pytest.mark.parametrize('layer_groups', [None, [0, 1, 0]])
@torch.inference_mode()
def test_v41_context_graph_replay_matches_checkpoint_projection(default_vllm_config, workspace_init, monkeypatch, layer_groups, width):
    """Real native projections, rotary and cache writes across shrinking batches."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip('native V4.1 context preparation requires SM12x')
    from contextlib import nullcontext
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.attention import compressed_sparse_mla
    from b12x.attention.compressed_sparse_mla import rotary
    from b12x.preparation import PreparationSession, PreparedCall
    from vllm.model_executor.warmup.b12x_prepare import _units_from_modules
    from vllm.models.deepseek_v4_1.attention import DeepseekV4Attention
    from vllm.models.deepseek_v4_1.b12x_layers import B12xFP8LinearMethod, B12xRMSNorm
    from vllm.models.deepseek_v4_1.nvidia.dspark import DSparkContextCudaGraphs, DSparkDeepseekV4Model, _ContextKVProjection
    from vllm.utils.b12x import B12xWorkload, b12x_unit_providers
    from vllm.v1.worker import workspace
    from vllm.v1.worker.gpu import cudagraph_utils
    monkeypatch.setattr(cudagraph_utils, 'get_pp_group', lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True))
    monkeypatch.setattr(cudagraph_utils, 'is_global_first_rank', lambda: False)
    monkeypatch.setattr(cudagraph_utils, 'graph_capture', lambda device: nullcontext())
    monkeypatch.setenv("DS41_COMPACT_CONTEXT_GRAPH", "1")
    config = default_vllm_config
    config.scheduler_config.max_num_batched_tokens = 128
    config.scheduler_config.max_num_seqs = 16
    config.compilation_config.max_cudagraph_capture_size = 96
    device = torch.device('cuda', torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(1831)

    class NativeLinear(nn.Module):

        def __init__(self, n, k):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(n, k, device=device, generator=generator).to(torch.float8_e4m3fn), requires_grad=False)
            self.weight_scale_inv = torch.randint(122, 128, (n // 32, k // 32), dtype=torch.uint8, device=device, generator=generator).view(torch.float8_e8m0fnu)
            self.method = B12xFP8LinearMethod(SimpleNamespace(weight_block_size=[32, 32]))
            self.method.process_weights_after_loading(self)

        def forward(self, x):
            return self.method.apply(self, x)
    with torch.device(device):
        model = DSparkDeepseekV4Model.__new__(DSparkDeepseekV4Model)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(hidden_size=width, dspark_target_layer_ids=(0, 1, 2))
        model.main_proj = NativeLinear(width, width * 3)
        model.main_norm = B12xRMSNorm(width)
        model.layers = nn.ModuleList()
        model._context_kv_projections = []
        page_bytes = compressed_sparse_mla.page_nbytes(32, cache_format='deepseek_v41', cache_kind='swa')
        phases = torch.arange(512, dtype=torch.float32)[:, None] * (torch.arange(32, dtype=torch.float32)[None, :] + 1) / 128
        for _ in range(3):
            attn = DeepseekV4Attention.__new__(DeepseekV4Attention)
            nn.Module.__init__(attn)
            attn._ready = True
            attn.q_lora_rank = 256
            attn.fused_wqa_wkv = NativeLinear(768, width)
            attn.kv_norm = B12xRMSNorm(512)
            attn.rotary_emb = SimpleNamespace(cos_sin_cache=torch.cat((phases.cos(), phases.sin()), dim=-1))
            attn.swa_cache_layer = SimpleNamespace(kv_cache=torch.full((8, page_bytes), 91, dtype=torch.uint8), block_size=32)
            attn._helper_plans = {'kv': rotary.plan(rotary.Query(max_rows=128, heads=1, dim=512, cos_sin_dtype='float32'), device=device), 'swa_cache_write': compressed_sparse_mla.plan_cache_writer(compressed_sparse_mla.CacheWriterQuery(max_rows=128, page_size=32, cache_kind='swa', slot_dtype='int64'), device=device)}
            layer = nn.Module()
            layer.attn = attn
            model.layers.append(layer)
            model._context_kv_projections.append(_ContextKVProjection(attn, 128))
        hidden = torch.zeros(128, width, dtype=torch.bfloat16)
        positions = torch.zeros(128, dtype=torch.int64)
        slots = torch.full((2, 128), -1, dtype=torch.int64)
        context = DSparkContextCudaGraphs(model, config, hidden, positions, slots, layer_groups, 96)
        assert context.manager.compilation_config.cudagraph_capture_sizes == [1, 2, 4, 8, 16, 32, 64, 96, 128]
        session = PreparationSession(device=device, autotune=False)
        workload = B12xWorkload(stage='weights', token_counts=tuple(context.manager.compilation_config.cudagraph_capture_sizes), fixed_token_counts=(), output_dtype=torch.bfloat16, max_tokens=128, max_seqs=16, max_model_len=512)
        units = [unit for provider in b12x_unit_providers() for unit in provider.get_b12x_preparation_units(provider, workload)]
        units.extend(_units_from_modules(model, workload))
        prep_requests = [request for unit in units for request in unit.requests]
        for index, layer in enumerate(model.layers):

            def prepare_rotary(state, table=layer.attn.rotary_emb.cos_sin_cache):
                source = torch.ones((1, 1, 512), dtype=torch.bfloat16, device=device)
                output = torch.empty_like(source)
                positions = torch.zeros(1, dtype=torch.int64, device=device)
                return PreparedCall(run=lambda: state.run(source, positions, table, out=output))

            def prepare_writer(state):
                source = torch.ones((1, 512), dtype=torch.bfloat16, device=device)
                cache = torch.empty((1, page_bytes), dtype=torch.uint8, device=device)
                slots = torch.zeros(1, dtype=torch.int64, device=device)
                return PreparedCall(run=lambda: state.run(source, cache, slots))
            for role, prepare in (('kv', prepare_rotary), ('swa_cache_write', prepare_writer)):
                prep_requests.append(layer.attn._helper_plans[role].request(name=f'context.{index}.{role}', prepare_call=prepare))
        result = session.prepare(prep_requests, autotune=False)
        assert result is not None
        x = torch.randn(128, width, dtype=torch.bfloat16, generator=generator)
        for layer, projection in zip(model.layers, model._context_kv_projections):
            full = layer.attn.fused_wqa_wkv(x)
            torch.testing.assert_close(projection(x), full[:, 256:], rtol=0, atol=0)
        for capacity in context.manager.compilation_config.cudagraph_capture_sizes:
            context._forward(capacity)
        workspace.lock_workspace()
        with kernel_resolution_guard('DSpark context capacities are prepared'):
            try:
                with session.capture():
                    context.capture()
                for rows in (128, 7, 96, 3, 128, 1):
                    aux = [torch.randn(rows, width, dtype=torch.bfloat16, generator=generator) for _ in range(3)]
                    positions.fill_(1000000)
                    positions[:rows].copy_(torch.arange(rows, dtype=torch.int64) + rows)
                    slots[0].copy_(torch.arange(128, dtype=torch.int64) + 32)
                    slots[1].copy_(torch.arange(128, dtype=torch.int64) + 64)
                    slots[:, rows - 1] = -1
                    if rows > 1:
                        slots[1, 0] = -1
                    main_x = model.combine_hidden_states(torch.cat(aux, dim=-1))
                    expected = []
                    for i, layer in enumerate(model.layers):
                        attn = layer.attn
                        attn.swa_cache_layer.kv_cache.fill_(91)
                        kv = attn.kv_norm(attn.fused_wqa_wkv(main_x)[:, 256:])
                        group = 0 if layer_groups is None else layer_groups[i]
                        attn.insert_context_kv(kv, positions[:rows], slots[group, :rows])
                        expected.append(attn.swa_cache_layer.kv_cache.clone())
                        attn.swa_cache_layer.kv_cache.fill_(91)
                    context.run(aux, rows)
                    torch.accelerator.synchronize()
                    torch.testing.assert_close(hidden[:rows], main_x, rtol=0, atol=0)
                    for layer, cache in zip(model.layers, expected):
                        torch.testing.assert_close(layer.attn.swa_cache_layer.kv_cache, cache, rtol=0, atol=0)
                    for source in aux:
                        source.fill_(float('nan'))
                    slots.fill_(0)
                    context.run(aux, rows, context_kv_is_restored=True)
                    for layer, cache in zip(model.layers, expected):
                        torch.testing.assert_close(layer.attn.swa_cache_layer.kv_cache, cache, rtol=0, atol=0)
                assert not context.can_run(129)
                with pytest.raises(ValueError):
                    context.run(aux, 129)
            finally:
                context.close()
                workspace.unlock_workspace()
        session.close()
