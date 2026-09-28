#!/usr/bin/env python3
"""Local-only configuration migration for the 2026-09-28 performance bundle."""
import argparse
import copy
import json
from pathlib import Path
import tempfile

import fleet

HERE = Path(__file__).resolve().parent
LATEST_VLLM = '502d6cb5acd2ba2a62ecf58497be558c9d86089f'
LATEST_B12X = 'd44247b6171f7c2f9787341ae884b537887d7df9'
IMAGE = 'spark-vllm-ds41:kk-502d6cb-b12x-d44247b-performance-v1'
CANDIDATE = {
    'engram_overlap': False,
    'dspark_markov_nvfp4': True,
    'dspark_draft_nvfp4_head': True,
    'draft_sample_method': 'probabilistic',
    'rejection_sample_method': 'block',
    'enable_adaptive_verification': True,
    'l2_prefetch': True,
    'engram_projection_tp': False,
    'decode_graph_policy': 'exact',
    'h2d_staging': True,
    'b12x_defer_gc': True,
    'moe_coalesce_barriers': True,
    'bounded_prefix_hashes': True,
    'dspark_skip_prefill_draft': True,
    'dspark_compact_context_graph': True,
}
CONTROL = {
    'engram_overlap': True,
    'dspark_markov_nvfp4': False,
    'dspark_draft_nvfp4_head': False,
    'draft_sample_method': 'greedy',
    'rejection_sample_method': 'standard',
    'enable_adaptive_verification': True,
    'l2_prefetch': True,
    'engram_projection_tp': False,
    'decode_graph_policy': 'upstream',
    'h2d_staging': False,
    'b12x_defer_gc': False,
    'moe_coalesce_barriers': False,
    'bounded_prefix_hashes': False,
    'dspark_skip_prefill_draft': False,
    'dspark_compact_context_graph': False,
}


def candidate(source, *, control=False, batch_tokens=None, autotune=None, overrides=None):
    c = copy.deepcopy(source)
    if c.get('display_kv', {}).get('enabled'):
        raise ValueError('Display-KV is incompatible with this Karmic bundle')
    c.update(image=IMAGE, upstream_branch='dev/karmic-kraken',
             vllm_commit=LATEST_VLLM, b12x_commit=LATEST_B12X,
             reduced_tuning=False, performance_bundle='ds41-performance-v1')
    c.pop('graph_request_buckets', None)
    c.update(CONTROL if control else CANDIDATE)
    c.setdefault('b12x_compile_workers', 4)
    c.setdefault('swa_block_size', 128)
    # Existing measured memory settings and resident-scale choices are preserved.
    # A larger batch is explicitly selected, never silently substituted.
    if batch_tokens is not None:
        c['max_num_batched_tokens'] = batch_tokens
    c['prefill_8192_graph'] = not control and c['max_num_batched_tokens'] == 8192
    if autotune is not None:
        c['b12x_autotune'] = autotune
        c['b12x_bounded_autotune'] = autotune
    elif c.get('b12x_autotune', False):
        c['b12x_bounded_autotune'] = True
    if c.get('b12x_bounded_autotune', False):
        c.update(b12x_compile_workers=1, b12x_preparation_trace=True,
                 b12x_hang_dump=True)
        c.setdefault('b12x_race_budget_mib', 1024)
        c.setdefault('b12x_memory_reserve_mib', 4096)
    if c['draft_tokens'] == 0:
        for key in ('dspark_markov_nvfp4', 'dspark_draft_nvfp4_head',
                    'dspark_skip_prefill_draft', 'dspark_compact_context_graph'):
            c[key] = False
    for key, value in (overrides or {}).items():
        if key not in set(CANDIDATE) | {'engram_resident_scales', 'shm_busy_loop_s',
                                      'prefill_8192_graph', 'adaptive_verification_cost_scale'}:
            raise ValueError(f'Unknown performance setting: {key}')
        c[key] = value
    return c


def write(source, output, **options):
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve() or output.exists():
        raise ValueError('Use a new output filename; existing recipes are preserved')
    c = candidate(fleet.load_config(source), **options)
    with tempfile.TemporaryDirectory() as directory:
        staged = Path(directory) / 'config.json'
        staged.write_text(json.dumps(c), encoding='utf-8')
        fleet.load_config(staged)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        json.dump(c, handle, indent=2)
        handle.write('\n')
    print(f'Wrote {output}; no fleet actions were performed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--control', action='store_true')
    parser.add_argument('--batch-tokens', type=int, choices=(4096, 8192))
    parser.add_argument('--autotune', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--set', action='append', default=[], metavar='KEY=JSON_VALUE')
    args = parser.parse_args()
    overrides = {}
    for assignment in args.set:
        key, value = assignment.split('=', 1)
        overrides[key] = json.loads(value)
    write(args.from_config, args.output, control=args.control,
          batch_tokens=args.batch_tokens, autotune=args.autotune, overrides=overrides)
