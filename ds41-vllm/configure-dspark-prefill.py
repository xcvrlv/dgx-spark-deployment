#!/usr/bin/env python3
"""Copy the current operator recipe and enable the optional DSpark prefill image."""
import argparse
import json
from pathlib import Path
import tempfile

import fleet


def configure(source, output, image, *, skip=True, graph=True):
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve() or output.exists():
        raise ValueError('Use a new output filename to preserve existing recipes')
    config = fleet.load_config(source)
    if image == config['image']:
        raise ValueError('Use a new child-image tag to preserve rollback')
    if fleet.source_pins(config) == fleet.LATEST_PINS:
        raise ValueError('Latest performance image includes DSpark prefill; use performance-settings.py')
    config.update(image=image, dspark_skip_prefill_draft=skip,
                  dspark_compact_context_graph=graph, graph_memory_debug=True)
    # Validate the completed candidate before creating the operator's file.
    data = json.dumps(config, indent=2) + '\n'
    with tempfile.TemporaryDirectory() as directory:
        candidate = Path(directory) / 'candidate.json'
        candidate.write_text(data, encoding='utf8')
        fleet.load_config(candidate)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf8') as handle:
        handle.write(data)
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-config', required=True)
    parser.add_argument('--output', default='fleet.dspark-prefill.json')
    parser.add_argument('--image', default='spark-vllm-ds41:kk-1794dcf-b12x-a7d7d29-dspark-prefill-v1')
    parser.add_argument('--skip-prefill-draft', choices=('on', 'off'), default='on')
    parser.add_argument('--compact-context-graph', choices=('on', 'off'), default='on')
    args = parser.parse_args()
    configure(args.from_config, args.output, args.image,
              skip=args.skip_prefill_draft == 'on', graph=args.compact_context_graph == 'on')
