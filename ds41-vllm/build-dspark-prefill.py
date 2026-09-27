#!/usr/bin/env python3
"""Build the optional DSpark prefill image from an existing operator recipe."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile

import fleet

HERE = Path(__file__).resolve().parent
DEFAULT_IMAGE = 'spark-vllm-ds41:kk-1794dcf-b12x-a7d7d29-dspark-prefill-v1'
configure = runpy.run_path(str(HERE / 'configure-dspark-prefill.py'))['configure']


def discover(directory=HERE):
    """List recipes, without guessing which one launched the current service."""
    directory = Path(directory)
    paths = sorted({*directory.glob('*.json'), *(directory / '.build').glob('*.json')})
    candidates = []
    for path in paths:
        try:
            config = fleet.load_config(path)
            if (config.get('upstream_branch') == 'dev/karmic-kraken'
                    and config['draft_tokens'] > 0
                    and config['image'] != DEFAULT_IMAGE):
                candidates.append((path, config))
        except (OSError, ValueError, KeyError, AssertionError, TypeError):
            continue
    return candidates


def print_candidates(candidates):
    for number, (path, config) in enumerate(candidates, 1):
        print(f'{number}. {path}')
        print(f'   image={config["image"]}; K={config["draft_tokens"]}; '
              f'resident_scales={config.get("engram_resident_scales", False)}')
        print(f'   model={config["model_path"]}')


def select_source():
    candidates = discover()
    print_candidates(candidates)
    if not candidates:
        raise ValueError('No Karmic recipes found here or in .build. '
                         'Pass the actual recipe path with --from-config.')
    if not sys.stdin.isatty():
        raise ValueError('Pass --from-config with the recipe used for your current launch.')
    value = input('Select the recipe used for your current launch (number; Enter cancels): ').strip()
    if not value:
        raise ValueError('Cancelled; no image or recipe changed.')
    try:
        number = int(value)
    except ValueError:
        raise ValueError('Enter a recipe number, or pass its path with --from-config.') from None
    if not 1 <= number <= len(candidates):
        raise ValueError('Recipe number is outside the displayed list.')
    return candidates[number - 1][0]


def build(source, output, image=DEFAULT_IMAGE, *, share=False):
    source, output = Path(source), Path(output)
    if not source.is_file():
        raise ValueError(f'Source recipe does not exist: {source}. '
                         'Use --list-configs to find the actual filename.')
    if source.resolve() == output.resolve():
        raise ValueError('Source and output must differ; keep the working recipe for rollback.')
    config = fleet.load_config(source)
    base_image = config.get('image')
    if not isinstance(base_image, str) or not base_image.strip():
        raise ValueError(f'Source recipe has no usable image: {source}')
    if not isinstance(image, str) or not image.strip():
        raise ValueError('The child image tag must not be empty.')
    # Stage and validate the complete recipe before invoking Docker. Publish it
    # only after a successful build, so a failed build leaves no phantom recipe.
    with tempfile.TemporaryDirectory() as directory:
        staged = Path(directory) / 'candidate.json'
        with contextlib.redirect_stdout(io.StringIO()):
            configure(source, staged, image)
        candidate = json.loads(staged.read_text(encoding='utf8'))
        if output.exists() and json.loads(output.read_text(encoding='utf8')) != candidate:
            raise ValueError(f'Output already contains a different recipe: {output}. '
                             'Use a new --output filename.')
        print(f'Source recipe: {source}\nBase image: {base_image}\nChild image: {image}', flush=True)
        subprocess.run(['docker', 'image', 'inspect', '--format', '{{.Id}}', base_image], check=True)
        subprocess.run(['docker', 'build', '-f', str(HERE / 'Dockerfile.dspark-prefill'),
                        '--build-arg', f'BASE_IMAGE={base_image}', '-t', image, str(HERE)], check=True)
        if not output.exists():
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open('x', encoding='utf8') as handle:
                handle.write(staged.read_text(encoding='utf8'))
    print(f'Candidate recipe: {output}', flush=True)
    if share:
        subprocess.run([sys.executable, str(HERE / 'fleet.py'), '--config', str(output), 'share'], check=True)
    print('Build complete. Follow DSPARK-PREFILL.md for the GPU checks and memory-qualified launch.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-config', type=Path)
    parser.add_argument('--output', type=Path, default=Path('fleet.dspark-prefill.json'))
    parser.add_argument('--image', default=DEFAULT_IMAGE)
    parser.add_argument('--list-configs', action='store_true')
    parser.add_argument('--share', action='store_true', help='Distribute the image after a successful build')
    args = parser.parse_args()
    if args.list_configs:
        print_candidates(discover())
        return 0
    try:
        source = args.from_config if args.from_config is not None else select_source()
        build(source, args.output, args.image, share=args.share)
    except (OSError, ValueError, KeyError, AssertionError, TypeError, EOFError) as error:
        print(f'Build stopped: {error}', file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as error:
        print(f'Build stopped: {error.cmd[0]} exited {error.returncode}; later steps were not run.', file=sys.stderr)
        return error.returncode if error.returncode > 0 else 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
