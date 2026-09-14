#!/usr/bin/env python3
"""Create an isolated c16 graph-coverage profile from the actual operator config."""
import argparse
import json
from pathlib import Path
import fleet


def configure(source, output, control=False):
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError('Use a separate output file to preserve the comparison profile')
    c = fleet.load_config(source)
    if not c['draft_tokens']:
        raise ValueError('Graph coverage trial requires DSpark enabled (e.g. draft_tokens=5)')
    c['max_num_seqs'] = 16
    c['graph_request_buckets'] = not control
    if not c['image'].endswith('-graphs-v1'):
        c['image'] += '-graphs-v1'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(c, indent=2) + '\n')
    fleet.load_config(output)
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--control', action='store_true')
    args = parser.parse_args()
    configure(args.from_config, args.output, args.control)
