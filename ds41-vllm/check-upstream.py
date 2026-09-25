#!/usr/bin/env python3
"""Check upstream before a build; report newer heads without changing source pins."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.request


def check(versions):
    pins = {}
    for line in Path(versions).read_text().splitlines():
        if line.startswith(('VLLM_COMMIT=', 'B12X_COMMIT=')):
            key, value = line.split('=', 1)
            pins[key] = value
    report = {'checked_at': datetime.now(timezone.utc).isoformat(), 'repositories': {}}
    refs = [('vllm', 'vllm', 'dev/karmic-kraken', 'VLLM_COMMIT'),
            ('vllm_jovian', 'vllm', 'dev/jovian-judgement', None),
            ('b12x', 'b12x', 'HEAD', 'B12X_COMMIT')]
    for name, repo, ref, key in refs:
        url = f'https://api.github.com/repos/local-inference-lab/{repo}/commits/{ref}'
        req = urllib.request.Request(url, headers={'User-Agent': 'spark-deployment-upstream-audit'})
        with urllib.request.urlopen(req, timeout=30) as response:
            commit = json.load(response)
        report['repositories'][name] = {'pin': pins[key] if key else None,
                                        'head': commit['sha'],
                                        'different': pins[key] != commit['sha'] if key else None,
                                        'url': commit['html_url']}
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    parser.add_argument('--versions', default=str(Path(__file__).with_name('versions.env')))
    args = parser.parse_args()
    report = check(args.versions)
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text + '\n')
