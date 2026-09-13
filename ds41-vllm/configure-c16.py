#!/usr/bin/env python3
"""Create a c16 candidate from existing fleet settings, preserving paths and SSH."""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-config', default=str(HERE / 'cluster-c16.json'))
    parser.add_argument('--output', default=str(HERE / '.build/cluster-c16.json'))
    parser.add_argument('--control', action='store_true', help='disable all four RoCEnante port switches')
    parser.add_argument('--prefill8192', action='store_true', help='experimental larger prefill batches')
    args = parser.parse_args()
    config = json.loads(Path(args.from_config).read_text())
    defaults = json.loads((HERE / 'cluster-c16.json').read_text())
    for key in ('image', 'max_num_seqs', 'max_model_len', 'gpu_memory_utilization', 'roce_optimizations'):
        config[key] = defaults[key]
    config['max_num_batched_tokens'] = 8192 if args.prefill8192 else 4096
    if args.control:
        config['roce_optimizations'] = {key: False for key in defaults['roce_optimizations']}
    for node in config['nodes']:
        node['host'] = node['ip']
    path = Path(args.output)
    if path.resolve() == Path(args.from_config).resolve():
        raise SystemExit('Choose a separate output file to preserve the source profile.')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + '\n')
    print(path)
