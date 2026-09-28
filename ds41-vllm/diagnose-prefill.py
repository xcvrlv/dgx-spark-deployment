#!/usr/bin/env python3
"""Inspect the running prefill deployment without restarting it or using CUDA.

Checks container identity, effective settings and hash-guarded source patches.
Activation logs prove execution, not a speedup. Missing markers are inconclusive.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import sys

import fleet

ROOT = Path(__file__).resolve().parent
PATCH_FILES = ('prefill_hashes.py', 'dspark_prefill.py', 'dspark_prefill_runtime.py')
MARKERS = {
    'compact_context_graph': 'DS41 compact-prefill context graph replay active',
    'skip_prefill_draft': 'DS41 skipping unused intermediate-prefill drafts',
}
ENV_KEYS = (
    'DS41_SKIP_PREFILL_DRAFT', 'DS41_COMPACT_CONTEXT_GRAPH',
    'VLLM_USE_V2_MODEL_RUNNER', 'VLLM_DEBUG_GRAPH_MEMORY_ACCOUNTING',
    'VLLM_USE_BREAKABLE_CUDAGRAPH', 'OMP_NUM_THREADS',
    'DS41_H2D_STAGING', 'DS41_DEFER_AUTOTUNE_GC', 'DS41_BOUNDED_PREFIX_HASHES',
    'DS41_SHM_BUSY_LOOP_S', 'DS41_PREFILL_8192_GRAPH',
    'VLLM_DS41_ENGRAM_OVERLAP', 'VLLM_DS41_MARKOV_NVFP4',
    'VLLM_DS41_DRAFT_NVFP4_HEAD',
    'VLLM_DS41_L2_PREFETCH',
    'DS41_MOE_COALESCE_BARRIERS',
)
ARG_KEYS = (
    '--engram-config', '--speculative-config', '--max-num-batched-tokens',
    '--max-model-len', '--max-num-seqs', '--gpu-memory-utilization',
    '--compilation-config', '--kernel-config', '--swa-block-size',
)

# Metadata and source reads only: do not import vllm, torch or b12x. Match the
# installed validation scripts to this checkout before running their --check API.
CONTAINER_PROBE = r'''
import hashlib, json, runpy
from importlib.metadata import distribution
from pathlib import Path
expected = EXPECTED_HASHES
root = Path('/opt/ds41/patches')
result = {'patch_files': {}, 'patch_checks': {}}
for name, digest in expected.items():
    path = root / name
    actual = hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest() if path.is_file() else None
    result['patch_files'][name] = {'sha256': actual, 'matches_checkout': actual == digest}
if all(item['matches_checkout'] for item in result['patch_files'].values()):
    package = distribution('vllm').locate_file('vllm')
    result['package_path'] = str(package)
    for name in ('prefill_hashes.py', 'dspark_prefill.py'):
        try:
            runpy.run_path(str(root / name))['patch'](package, check=True)
            result['patch_checks'][name] = 'passed'
        except Exception as exc:
            result['patch_checks'][name] = str(exc)
print(json.dumps(result))
'''

# The 8192 graph overlay changes the already-patched model runner. Verify the
# complete composition rather than asking an earlier layer to match its output.
BUNDLE_PROBE = r'''
import hashlib, json, runpy
from importlib.metadata import distribution
from pathlib import Path
expected = EXPECTED_HASHES
result = {'patch_files': {}, 'patch_checks': {}}
for relative, digest in expected.items():
    path = Path('/opt/ds41') / relative
    actual = hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest() if path.is_file() else None
    result['patch_files'][relative] = {'sha256': actual, 'matches_checkout': actual == digest}
if all(item['matches_checkout'] for item in result['patch_files'].values()):
    try:
        roots = {name: distribution(name).locate_file(name) for name in ('vllm', 'b12x')}
        runpy.run_path('/opt/ds41/performance-check.py')['check'](
            roots, '/opt/ds41/performance-manifest.json')
        result['patch_checks']['performance_bundle'] = 'passed'
    except Exception as exc:
        result['patch_checks']['performance_bundle'] = str(exc)
print(json.dumps(result))
'''

HOST_PROBE = r'''
import json, re, subprocess, sys
from pathlib import Path
name, image = sys.argv[1:3]
def docker(*args, source=None):
    p = subprocess.run(['docker', *args], input=source, text=True,
                       capture_output=True, timeout=90)
    if p.returncode:
        raise RuntimeError(p.stderr.strip() or p.stdout.strip())
    return p.stdout
def attempt(fn):
    try:
        return fn()
    except Exception as exc:
        return {'error': str(exc)}
def container():
    c = json.loads(docker('container', 'inspect', name))[0]
    config = c['Config']
    env = dict(s.split('=', 1) for s in config.get('Env', []) if '=' in s)
    cmd = config.get('Cmd') or []
    # Only whitelist diagnostic fields; never dump the complete environment.
    args = {}
    for key in ARG_KEYS:
        for i, value in enumerate(cmd):
            if value == key and i + 1 < len(cmd):
                args[key] = cmd[i + 1]
            elif value.startswith(key + '='):
                args[key] = value[len(key) + 1:]
    return {'image_id': c['Image'], 'image_reference': config.get('Image'),
            'state': {k: c['State'].get(k) for k in ('Running', 'OOMKilled', 'StartedAt')},
            'labels': {k: v for k, v in (config.get('Labels') or {}).items()
                       if k.startswith(('local-inference.', 'org.opencontainers.image.'))},
            'environment': {k: env.get(k) for k in ENV_KEYS}, 'arguments': args}
def logs():
    p = subprocess.run(['docker', 'logs', '--tail', '20000', name], text=True,
                       capture_output=True, timeout=90)
    if p.returncode:
        raise RuntimeError(p.stderr.strip())
    lines = (p.stdout + '\n' + p.stderr).splitlines()
    pattern = re.compile(r'DS41 |Graph capturing finished|GPU KV cache size|\[CG MEM\]|preempt|out of memory|OutOfMemory|engram|resident.scale', re.I)
    return {'tail_limit': 20000,
            'activation': {k: any(v in line for line in lines) for k, v in MARKERS.items()},
            'relevant_lines': [s for s in lines if pattern.search(s)][-100:]}
def memory():
    values = {k: int(v.split()[0]) for k, v in
              (s.split(':', 1) for s in Path('/proc/meminfo').read_text().splitlines())}
    return {'available_gib': values['MemAvailable'] / 1024**2,
            'swap_used_gib': (values['SwapTotal'] - values['SwapFree']) / 1024**2,
            'note': 'One snapshot cannot establish peak headroom or new swap activity.'}
print(json.dumps({
    'container': attempt(container),
    'tag_image_id': attempt(lambda: json.loads(docker('image', 'inspect', image))[0]['Id']),
    'source': attempt(lambda: json.loads(docker('exec', '-i', name, 'python3', '-', source=CONTAINER_PROBE))),
    'logs': attempt(logs), 'memory': attempt(memory),
}))
'''


def probe_script(config, rank):
    digests = {name: hashlib.sha256((ROOT / 'patches' / name).read_bytes()
                                  .replace(b'\r\n', b'\n')).hexdigest()
               for name in PATCH_FILES}
    container_probe = CONTAINER_PROBE.replace('EXPECTED_HASHES', repr(digests))
    if fleet.source_pins(config) == fleet.LATEST_PINS:
        digests = {relative: hashlib.sha256((ROOT / source).read_bytes()
                                          .replace(b'\r\n', b'\n')).hexdigest()
                   for relative, source in (
                       ('performance-check.py', 'performance-check.py'),
                       ('performance-manifest.json', 'patches/performance-manifest.json'))}
        container_probe = BUNDLE_PROBE.replace('EXPECTED_HASHES', repr(digests))
    header = (f'ENV_KEYS = {ENV_KEYS!r}\nARG_KEYS = {ARG_KEYS!r}\n'
              f'MARKERS = {MARKERS!r}\nCONTAINER_PROBE = {container_probe!r}\n')
    command = shlex.join(['python3', '-', f'{fleet.NAME}-{rank}', config['image']])
    return command + " <<'DS41_DIAGNOSTIC_PY'\n" + header + HOST_PROBE + '\nDS41_DIAGNOSTIC_PY\n'


def assess(config, rank, data):
    issues = []
    for key in ('container', 'tag_image_id', 'source', 'logs', 'memory'):
        item = data.get(key)
        if item is None or isinstance(item, dict) and 'error' in item:
            issues.append(f'{key}: {item}')
    container = data.get('container', {})
    if 'error' not in container and container:
        if not container.get('state', {}).get('Running'):
            issues.append('Serving container is not running')
        if container.get('state', {}).get('OOMKilled'):
            issues.append('Docker reports OOMKilled')
        if container.get('image_id') != data.get('tag_image_id'):
            issues.append('Running image ID differs from the configured tag (or tag is unavailable)')
        labels = container.get('labels', {})
        for label, expected in (
            ('org.opencontainers.image.revision', config.get('vllm_commit')),
            ('local-inference.b12x.commit', config.get('b12x_commit')),
            (('local-inference.performance-bundle', 'ds41-performance-v1')
             if fleet.source_pins(config) == fleet.LATEST_PINS else
             ('local-inference.prefill-hash-overlay', 'ds41-bounded-hashes-v1')),
            ('local-inference.dspark-prefill-overlay', 'ds41-dspark-prefill-v1'),
        ):
            if expected is not None and labels.get(label) != expected:
                issues.append(f'Image label mismatch: {label}')
        expected_env = fleet.environment(config, rank)
        actual_env = container.get('environment', {})
        for key in ENV_KEYS:
            if key in expected_env and actual_env.get(key) != expected_env[key]:
                issues.append(f'Runtime environment mismatch: {key}')
        if (config.get('upstream_branch') == 'dev/karmic-kraken'
                and actual_env.get('VLLM_USE_BREAKABLE_CUDAGRAPH') == '0'):
            issues.append('Breakable CUDA graphs are explicitly disabled in the running container')
        expected_args = fleet.serve_args(config, rank)
        for key in ARG_KEYS:
            actual = container.get('arguments', {}).get(key)
            expected = expected_args[expected_args.index(key) + 1] if key in expected_args else None
            # Docker preserves the launcher argv; compare JSON structurally too.
            if key.endswith('-config'):
                try:
                    actual = json.loads(actual) if actual is not None else None
                    expected = json.loads(expected) if expected is not None else None
                except (ValueError, TypeError):
                    pass
            if actual != expected:
                issues.append(f'Runtime argument mismatch: {key}')
    source = data.get('source', {})
    latest = fleet.source_pins(config) == fleet.LATEST_PINS
    files = ('performance-check.py', 'performance-manifest.json') if latest else PATCH_FILES
    checks = ('performance_bundle',) if latest else ('prefill_hashes.py', 'dspark_prefill.py')
    for name in files:
        if not source.get('patch_files', {}).get(name, {}).get('matches_checkout'):
            issues.append(f'Installed patch file missing or differs from checkout: {name}')
    for name in checks:
        if source.get('patch_checks', {}).get(name) != 'passed':
            issues.append(f'Source validation did not pass: {name}')
    if data.get('memory', {}).get('available_gib', 2) < 2:
        issues.append('Host MemAvailable is below 2 GiB at the time of this snapshot')
    return issues


def collect(config, rank):
    try:
        data = json.loads(fleet.remote(config, rank, probe_script(config, rank), timeout=420))
    except Exception as exc:
        return {'rank': rank, 'issues': [str(exc)]}
    return {'rank': rank, **data, 'issues': assess(config, rank, data)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = fleet.load_config(args.config)
    if args.output.exists():
        parser.error('Use a new output filename; existing reports are preserved')
    print('Reading running containers, source checks and recent logs on all four nodes...', flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        ranks = list(pool.map(lambda rank: collect(config, rank), range(4)))
    ids = {r.get('container', {}).get('image_id') for r in ranks}
    same_image = None not in ids and len(ids) == 1
    report = {'checked_at': datetime.now(timezone.utc).isoformat(),
              'config': str(Path(args.config).resolve()), 'expected_image': config['image'],
              'same_running_image_on_all_ranks': same_image, 'ranks': ranks}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf8') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    for rank in ranks:
        print(f"rank {rank['rank']}: " + ('checks passed' if not rank['issues'] else '; '.join(rank['issues'])))
        for key in MARKERS:
            observed = rank.get('logs', {}).get('activation', {}).get(key, False)
            print(f"  {key}: " + ('execution observed in logs' if observed else 'not observed in inspected logs'))
    print(f'Identical running image on all four ranks: {same_image}')
    print(f'Report: {args.output}')
    print('These checks do not measure throughput or qualify peak memory headroom.')
    return int(not same_image or any(r['issues'] for r in ranks))


if __name__ == '__main__':
    sys.exit(main())
