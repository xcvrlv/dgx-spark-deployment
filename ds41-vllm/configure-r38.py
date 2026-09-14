#!/usr/bin/env python3
"""Create the current JJ c8 recipe while retaining operator paths and transport."""
import argparse
import json
from pathlib import Path
import fleet

HERE = Path(__file__).resolve().parent


def configure(source, output, draft_tokens=5, graph_coverage=True):
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError('Use a separate output filename to preserve the old recipe')
    c = json.loads(source.read_text())
    defaults = json.loads((HERE / 'cluster-r38-c8.json').read_text())
    for key in ('image', 'max_num_seqs', 'max_model_len', 'gpu_memory_utilization',
                'max_num_batched_tokens', 'swa_block_size', 'roce_optimizations'):
        c[key] = defaults[key]
    c['draft_tokens'] = draft_tokens
    c['graph_request_buckets'] = graph_coverage
    c.setdefault('omp_num_threads', 2)
    for key in ('adaptive_speculative_tokens_window', 'adaptive_speculative_tokens_initial'):
        c.pop(key, None)
    for node in c['nodes']:
        node['host'] = node['ip']
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(c, indent=2) + '\n')
    fleet.load_config(output)
    print(output)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--from-config', default=str(HERE / 'cluster-r38-c8.json'))
    p.add_argument('--output', default=str(HERE / '.build/cluster-r38-c8.json'))
    p.add_argument('--draft-tokens', type=int, choices=(0,1,3,5,7), default=5)
    p.add_argument('--no-graph-coverage', action='store_true')
    a = p.parse_args()
    configure(a.from_config, a.output, a.draft_tokens, not a.no_graph_coverage)
