#!/usr/bin/env python3
"""Migrate an operator fleet config to the pinned Karmic Kraken recipe."""
import argparse
import json
from pathlib import Path

import fleet

HERE = Path(__file__).resolve().parent


def configure(source, output):
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError('Use a separate output filename to preserve the old recipe')
    old = json.loads(source.read_text())
    template = json.loads((HERE / 'cluster-karmic-c16.json').read_text())
    for key in ('image', 'upstream_branch', 'vllm_commit', 'b12x_commit',
                'max_num_seqs', 'max_model_len', 'max_num_batched_tokens',
                'gpu_memory_utilization', 'draft_tokens', 'startup_timeout',
                'swa_block_size', 'roce_optimizations', 'b12x_autotune',
                'b12x_compile_workers', 'reduced_tuning', 'fabric_check'):
        old[key] = template[key]
    # The new image carries only the RoCE transport patch. Drop controls for
    # the old graph/shape and display-KV experiments from the new profile.
    for key in ('display_kv', 'graph_request_buckets', 'torch_profile',
                'adaptive_speculative_tokens_window',
                'adaptive_speculative_tokens_initial'):
        old.pop(key, None)
    for node in old['nodes']:
        node['host'] = node['ip']
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(old, indent=2) + '\n')
    fleet.load_config(output)
    print(output)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--from-config', default=str(HERE / 'cluster-r38-c8.json'))
    p.add_argument('--output', default=str(HERE / '.build/cluster-karmic-c16.json'))
    a = p.parse_args()
    configure(a.from_config, a.output)
