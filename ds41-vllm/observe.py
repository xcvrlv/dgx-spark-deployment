#!/usr/bin/env python3
"""Collect fleet metrics/counters around a foreground benchmark; optional CPU/CUDA trace."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import subprocess
import time
import urllib.request

import fleet

HOST_PROBE = r'''
import json, time
from pathlib import Path
out = {'time': time.time(), 'counters': {}, 'errors': {}}
for group in ('counters', 'hw_counters'):
    for p in Path('/sys/class/infiniband').glob('*/ports/*/' + group + '/*'):
        try:
            out['counters'][str(p)] = int(p.read_text().strip())
        except (OSError, ValueError) as e:
            out['errors'][str(p)] = str(e)
for name in ('meminfo', 'pressure/cpu', 'pressure/memory'):
    try:
        out[name] = Path('/proc', name).read_text()
    except OSError as e:
        out[name] = str(e)
print(json.dumps(out))
'''


def delta(before, after):
    result = {}
    for rank, end in after.items():
        start = before.get(rank, {})
        if 'counters' not in start or 'counters' not in end:
            continue
        elapsed = end['time'] - start['time']
        rows = {}
        for key, value in end['counters'].items():
            if key not in start['counters']:
                continue
            change = value - start['counters'][key]
            row = {'delta': change, 'reset_or_wrap': change < 0}
            # Standard IB port data counters count octets divided by four.
            if key.endswith(('/counters/port_xmit_data', '/counters/port_rcv_data')) and change >= 0 and elapsed > 0:
                row['bytes_per_second'] = change * 4 / elapsed
            rows[key] = row
        result[rank] = {'elapsed_seconds': elapsed, 'counters': rows}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('provide a benchmark command after --')
    c = fleet.load_config(args.config)
    if args.profile and not c.get('torch_profile'):
        parser.error('enable torch_profile in the launch config and restart first')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'config.json').write_text(json.dumps(c, indent=2))
    base = f"http://{c['nodes'][0]['ip']}:{c['port']}"
    def http(path, post=False):
        req = urllib.request.Request(base + path, method='POST' if post else 'GET')
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.read().decode()
    def save(name, fn):
        try:
            value = fn()
        except Exception as e:
            value = f'COLLECTION ERROR: {e}'
        (out / name).write_text(value)
    def snapshot(label):
        def probe(rank):
            try:
                return str(rank), json.loads(fleet.remote(c, rank, "python3 - <<'PY'\n" + HOST_PROBE + '\nPY\n'))
            except Exception as e:
                return str(rank), {'error': str(e)}
        with ThreadPoolExecutor(max_workers=4) as pool:
            data = dict(pool.map(probe, range(4)))
        (out / (label + '-hosts.json')).write_text(json.dumps(data, indent=2))
        save(label + '-metrics.txt', lambda: http('/metrics'))
        return data
    before = snapshot('before')
    started = False
    began = time.time()
    code = None
    try:
        if args.profile:
            http('/start_profile', True)
            started = True
        with (out / 'benchmark.log').open('w') as log:
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
    finally:
        ended = time.time()
        if started:
            save('stop-profile.txt', lambda: http('/stop_profile', True))
        after = snapshot('after')
        (out / 'rdma-deltas.json').write_text(json.dumps(delta(before, after), indent=2))
        (out / 'run.json').write_text(json.dumps({'command': command, 'start': began, 'end': ended, 'returncode': code, 'profiled': args.profile}, indent=2))
        for rank in range(4):
            save(f'rank-{rank}.log', lambda rank=rank: fleet.remote(c, rank, f'docker logs --tail 3000 {fleet.NAME}-{rank} 2>&1'))
            save(f'rank-{rank}-inspect.json', lambda rank=rank: fleet.remote(c, rank, f'docker inspect {fleet.NAME}-{rank}'))
            if started:
                dest = out / f'rank-{rank}-profiles'
                dest.mkdir()
                source = f"{c['ssh_user']}@{c['nodes'][rank]['host']}:{c['cache_path']}/profiles/."
                # Reuse the launcher's SSH transport options for scp.
                result = subprocess.run(['scp', '-r', *fleet.ssh(c, rank)[1:-1], source, str(dest)], capture_output=True, text=True)
                (out / f'rank-{rank}-trace-copy.txt').write_text(f'exit={result.returncode}\n{result.stdout}\n{result.stderr}')
    print(out)
    raise SystemExit(code if code is not None else 1)


if __name__ == '__main__':
    main()
