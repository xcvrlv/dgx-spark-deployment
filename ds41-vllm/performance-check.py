#!/usr/bin/env python3
"""Verify every composed source overlay without loading CUDA or model modules."""
import argparse
import hashlib
from importlib.metadata import distribution
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def check(roots, manifest, *, expected_manifest_sha256=None):
    raw = Path(manifest).read_bytes().replace(b'\r\n', b'\n')
    if expected_manifest_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_manifest_sha256:
        raise ValueError('Performance source manifest differs from the operator checkout')
    report = json.loads(raw)
    if report['schema'] != 'ds41-performance-source/v1':
        raise ValueError('Unexpected performance source manifest')
    for item in report['files']:
        path = Path(roots[item['tree']]) / item['path']
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != item['sha256']:
            raise ValueError(f'Performance bundle source mismatch: {path}')
    return len(report['files'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=HERE / 'performance-manifest.json')
    parser.add_argument('--expected-manifest-sha256')
    args = parser.parse_args()
    roots = {name: distribution(name).locate_file(name) for name in ('vllm', 'b12x')}
    count = check(roots, args.manifest, expected_manifest_sha256=args.expected_manifest_sha256)
    print(f'DS41 performance bundle: {count} composed source files verified')
