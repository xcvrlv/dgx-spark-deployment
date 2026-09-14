#!/usr/bin/env python3
"""Filter non-version wheel artifact tags in our disposable build clone only."""
import argparse
import json
from pathlib import Path
import re
import subprocess

PATTERN = 'vllm-jovian-cu134-*'


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def prepare(repo, expected_commit, restore=False):
    repo = Path(repo).resolve()
    workspace = Path(__file__).resolve().parent / '.build'
    if not repo.is_relative_to(workspace.resolve()):
        raise ValueError('Only disposable clones under ds41-vllm/.build may be changed')
    if git(repo, 'rev-parse', 'HEAD') != expected_commit:
        raise ValueError('Build commit differs from expected pin')
    record = workspace / f'artifact-tags-{expected_commit}.json'
    saved = json.loads(record.read_text()) if record.exists() else {}
    if restore:
        for ref, oid in saved.items():
            if not ref.startswith('refs/tags/vllm-jovian-cu134-') or not re.fullmatch('[0-9a-f]{40,64}', oid):
                raise ValueError('Invalid saved artifact tag')
            current = subprocess.run(['git','-C',str(repo),'show-ref','--verify','--hash',ref], capture_output=True, text=True)
            if current.returncode == 0:
                if current.stdout.strip() != oid:
                    raise ValueError(f'Ref changed since backup: {ref}')
                continue
            git(repo, 'update-ref', ref, oid, '0'*len(oid))
        return
    refs = git(repo, 'for-each-ref', '--format=%(refname) %(objectname)', f'refs/tags/{PATTERN}')
    entries = dict(line.split(' ', 1) for line in refs.splitlines())
    for ref, oid in entries.items():
        if ref in saved and saved[ref] != oid:
            raise ValueError(f'Ref changed since backup: {ref}')
        saved[ref] = oid
    # Save rollback data before removing refs; update-ref checks the previous OID.
    record.write_text(json.dumps(saved, indent=2)+'\n')
    for ref, oid in entries.items():
        git(repo, 'update-ref', '-d', ref, oid)
    print(f'Filtered {len(entries)} wheel artifact tags from build clone; backup: {record}')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('repo')
    p.add_argument('--commit', required=True)
    p.add_argument('--restore', action='store_true')
    a=p.parse_args()
    prepare(a.repo,a.commit,a.restore)
