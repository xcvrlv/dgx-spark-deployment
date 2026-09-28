#!/usr/bin/env python3
"""Build a bounded-autotune child image from the actual operator recipe."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import fleet

HERE = Path(__file__).resolve().parent
IMAGE = 'spark-vllm-ds41:kk-1794dcf-b12x-a7d7d29-autotune-v2'


def candidate(source, image, batch_tokens=None):
    config = fleet.load_config(source)
    if fleet.source_pins(config) == fleet.LATEST_PINS:
        raise ValueError('Latest performance image includes bounded tuning; use performance-settings.py --autotune')
    if config['image'] == image:
        raise ValueError('Use a new image tag to preserve the source image')
    config.update(image=image, b12x_autotune=True, b12x_bounded_autotune=True,
                  b12x_compile_workers=1, b12x_preparation_trace=True,
                  b12x_hang_dump=True, b12x_race_budget_mib=1024,
                  b12x_memory_reserve_mib=4096, graph_memory_debug=True)
    if batch_tokens is not None:
        config['max_num_batched_tokens'] = batch_tokens
    return config


def build(source, output, image=IMAGE, *, batch_tokens=None, share=False):
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve() or output.exists():
        raise ValueError('Use a new output recipe; existing recipes are preserved')
    base = fleet.load_config(source)
    config = candidate(source, image, batch_tokens)
    with tempfile.TemporaryDirectory() as directory:
        staged = Path(directory) / 'candidate.json'
        staged.write_text(json.dumps(config))
        fleet.load_config(staged)
    print(f"Source: {source}; base image: {base['image']}; child: {image}", flush=True)
    print(f"Batch limit: {config['max_num_batched_tokens']}; model limit: {config['max_model_len']}", flush=True)
    inspect = json.loads(subprocess.check_output(['docker', 'image', 'inspect', base['image']], text=True))[0]
    labels = inspect['Config'].get('Labels') or {}
    for label, value in (('org.opencontainers.image.revision', config['vllm_commit']),
                         ('local-inference.b12x.commit', config['b12x_commit'])):
        if labels.get(label) != value:
            raise ValueError(f'Base image pin mismatch: {label}')
    if any(config.get(key, False) for key in fleet.DSPARK_PREFILL_OPTIONS):
        if labels.get('local-inference.dspark-prefill-overlay') != 'ds41-dspark-prefill-v1':
            raise ValueError('The source image lacks the requested DSpark overlay')
    subprocess.run(['docker', 'build', '-f', str(HERE / 'Dockerfile.autotune'),
                    '--build-arg', f"BASE_IMAGE={base['image']}", '-t', image, str(HERE)], check=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        json.dump(config, handle, indent=2)
        handle.write('\n')
    print(f'Wrote {output}', flush=True)
    if share:
        subprocess.run([sys.executable, str(HERE / 'fleet.py'), '--config', str(output), 'share'], check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('fleet.autotune.json'))
    parser.add_argument('--image', default=IMAGE)
    parser.add_argument('--batch-tokens', type=int, choices=(4096, 8192))
    parser.add_argument('--share', action='store_true')
    args = parser.parse_args()
    build(args.from_config, args.output, args.image, batch_tokens=args.batch_tokens, share=args.share)
