"""CPU-only routing for the pinned V4.1 DSpark prefill overlay.

No device reads, graph capture, allocation, or mutable request state here.
Both switches default off and are fixed by the launcher for every TP rank.
"""
import os

COMPACT_ROWS = 128


def supported_parallelism(config):
    parallel = config.parallel_config
    return all(getattr(parallel, name, None) == 1 for name in (
        'data_parallel_size', 'pipeline_parallel_size',
        'prefill_context_parallel_size', 'decode_context_parallel_size',
    ))


def native_dspark(speculator):
    return (speculator._speculator_name == 'DSpark'
            and callable(getattr(speculator.model, 'capture_context_preparation', None))
            and speculator.pcp_manager is None
            and supported_parallelism(speculator.vllm_config))


def compact_context_enabled(config):
    return (os.environ.get('DS41_COMPACT_CONTEXT_GRAPH') == '1'
            and supported_parallelism(config)
            and config.scheduler_config.max_num_batched_tokens >= COMPACT_ROWS)


def use_compact_context_graph(speculator, batch, context_rows):
    # Mixed batches and prompt-logprob paths with uncompressed rows stay eager.
    return (os.environ.get('DS41_COMPACT_CONTEXT_GRAPH') == '1'
            and native_dspark(speculator)
            and compact_context_enabled(speculator.vllm_config)
            and batch.num_reqs == 1
            and batch.has_prefill and bool(batch.is_prefilling_np[0])
            and batch.num_draft_tokens == 0
            and context_rows == COMPACT_ROWS
            and batch.num_tokens >= COMPACT_ROWS)


def skip_prefill_draft(speculator, batch, *, dummy_run, is_profile,
                       dp_sync, context_kv_is_restored):
    if (os.environ.get('DS41_SKIP_PREFILL_DRAFT') != '1'
            or dummy_run or is_profile or context_kv_is_restored
            or dp_sync is not None or not native_dspark(speculator)
            or batch.num_reqs <= 0 or not batch.has_prefill
            or batch.num_draft_tokens != 0
            or batch.has_structured_output_reqs):
        return False
    # Strictly before the prompt boundary: equality MUST generate the first
    # usable drafts. These are immutable CPU snapshots from before this step;
    # GPU request counters have already advanced by the time propose() runs.
    for i in range(batch.num_reqs):
        scheduled = int(batch.num_scheduled_tokens[i])
        computed = int(batch.num_computed_prefill_tokens_np[i])
        if (not bool(batch.is_prefilling_np[i]) or scheduled <= 0
                or computed + scheduled >= int(batch.prefill_len_np[i])):
            return False
    return True
