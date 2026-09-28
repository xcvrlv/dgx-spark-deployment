#!/usr/bin/env python3
"""Build an opt-in 8192-token graph image from the qualified autotune image."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import fleet

HERE = Path(__file__).resolve().parent
IMAGE = 'spark-vllm-ds41:kk-1794dcf-b12x-a7d7d29-autotune-b8192-v2'


def candidate(source, image):
    c = fleet.load_config(source)
    if fleet.source_pins(c) == fleet.LATEST_PINS:
        raise ValueError('Latest performance image includes the 8192 graph; use performance-settings.py --batch-tokens 8192')
    if c['image'] == image:
        raise ValueError('Use a new child image tag')
    if c['max_num_batched_tokens'] != 4096 or not c['b12x_bounded_autotune']:
        raise ValueError('The source must be the qualified 4096-token bounded recipe')
    c.update(image=image, max_num_batched_tokens=8192, prefill_8192_graph=True)
    return c


def build(source, output, image=IMAGE, *, share=False):
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve() or output.exists():
        raise ValueError('Use a new output recipe; preserve the source')
    c = candidate(source, image)
    with tempfile.TemporaryDirectory() as directory:
        staged = Path(directory) / 'candidate.json'
        staged.write_text(json.dumps(c))
        fleet.load_config(staged)
    base = fleet.load_config(source)['image']
    labels = json.loads(subprocess.check_output(['docker', 'image', 'inspect', base], text=True))[0]['Config']['Labels']
    for name, value in (
        ('org.opencontainers.image.revision', c['vllm_commit']),
        ('local-inference.b12x.commit', c['b12x_commit']),
        ('local-inference.bounded-autotune', 'ds41-bounded-autotune-v1'),
        ('local-inference.preparation-memory', 'image-built-v1'),
        ('local-inference.dspark-prefill-overlay', 'ds41-dspark-prefill-v1'),
    ):
        if labels.get(name) != value:
            raise ValueError(f'Base image missing qualified label: {name}')
    subprocess.run(['docker', 'build', '-f', str(HERE / 'Dockerfile.prefill-8192'),
                    '--build-arg', f'BASE_IMAGE={base}', '-t', image, str(HERE)], check=True)
    with output.open('x') as handle:
        json.dump(c, handle, indent=2)
        handle.write('\n')
    print(f'Created {output} from {source}; 8192-token graph opt-in enabled', flush=True)
    if share:
        subprocess.run([sys.executable, str(HERE / 'fleet.py'), '--config', str(output), 'share'], check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('fleet.autotune-b8192.json'))
    parser.add_argument('--image', default=IMAGE)
    parser.add_argument('--share', action='store_true')
    args = parser.parse_args()
    build(args.from_config, args.output, args.image, share=args.share)
