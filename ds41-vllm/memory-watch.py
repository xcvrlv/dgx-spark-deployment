#!/usr/bin/env python3
"""Sample all four hosts' shared RAM around a foreground startup or benchmark.

Read-only qualification, not an allocator or an OOM prevention mechanism.
Fails qualification for missing samples, <2 GiB available, or swap activity.
Sub-sample memory peaks can be missed. No CUDA contexts are created.
"""
import argparse
import json
import math
from pathlib import Path
import shlex
import subprocess
import threading
import time

import fleet

GIB = 1 << 30

# stdin remains open until the foreground command ends. EOF shuts the remote
# sampler down, including when this process exits unexpectedly.
PROBE = r'''
import json, select, sys, time
from pathlib import Path
interval = float(sys.argv[1])
while True:
    mem = {k: int(v.split()[0]) * 1024 for k, v in
           (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())}
    vm = dict(line.split() for line in Path('/proc/vmstat').read_text().splitlines())
    print(json.dumps({'time': time.time(), 'total_bytes': mem['MemTotal'],
        'available_bytes': mem['MemAvailable'],
        'swap_used_bytes': mem['SwapTotal'] - mem['SwapFree'],
        'swap_in_pages': int(vm['pswpin']), 'swap_out_pages': int(vm['pswpout'])}), flush=True)
    if select.select([sys.stdin], [], [], interval)[0]:
        break
'''


def summarize(samples, minimum_bytes, error=None):
    if not samples:
        return {'passed': False, 'samples': 0, 'error': error or 'No host samples'}
    first, last = samples[0], samples[-1]
    available = min(s['available_bytes'] for s in samples)
    swap_in = last['swap_in_pages'] - first['swap_in_pages']
    swap_out = last['swap_out_pages'] - first['swap_out_pages']
    return {
        'passed': error is None and available >= minimum_bytes and swap_in == swap_out == 0,
        'samples': len(samples), 'error': error,
        'elapsed_seconds': last['time'] - first['time'],
        'minimum_available_gib': available / GIB,
        'maximum_swap_used_gib': max(s['swap_used_bytes'] for s in samples) / GIB,
        'swap_in_pages': swap_in, 'swap_out_pages': swap_out,
    }


class Sampler:
    def __init__(self, config, rank, output, interval):
        self.samples = []
        self.error = None
        self.ready = threading.Event()
        self.stopping = False
        self.last_received = None
        self.maximum_gap = max(5.0, 3 * interval)
        self.stderr = (output / f'rank-{rank}-ssh.txt').open('w', encoding='utf8')
        self.raw = output / f'rank-{rank}.jsonl'
        command = shlex.join(['python3', '-u', '-c', PROBE, str(interval)])
        try:
            self.process = subprocess.Popen(
                fleet.ssh(config, rank) + [command], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=self.stderr, text=True,
            )
        except BaseException:
            self.stderr.close()
            raise
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        try:
            with self.raw.open('w', encoding='utf8') as out:
                for line in self.process.stdout:
                    sample = json.loads(line)
                    # Validate required fields before allowing the command to run.
                    for key in ('time', 'total_bytes', 'available_bytes',
                                'swap_used_bytes', 'swap_in_pages', 'swap_out_pages'):
                        if not isinstance(sample[key], (int, float)) or not math.isfinite(sample[key]) or sample[key] < 0:
                            raise ValueError(f'Invalid host field: {key}')
                    received = time.monotonic()
                    if self.last_received is not None and received - self.last_received > self.maximum_gap:
                        self.error = 'Host sampling gap exceeded the measurement tolerance'
                    self.last_received = received
                    self.samples.append(sample)
                    out.write(json.dumps(sample) + '\n')
                    out.flush()
                    self.ready.set()
            if not self.stopping:
                self.error = 'Sampler ended before foreground command completed'
        except Exception as exc:
            self.error = str(exc)
        finally:
            self.ready.set()

    def stop(self):
        if self.last_received is not None and time.monotonic() - self.last_received > self.maximum_gap:
            self.error = self.error or 'No recent host samples at command completion'
        self.stopping = True
        self.process.stdin.close()
        try:
            code = self.process.wait(timeout=10)
            if code:
                self.error = self.error or f'SSH sampler exited {code}; see SSH log'
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
            self.error = self.error or 'SSH sampler did not stop'
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            self.error = self.error or 'Sampler reader did not finish'
        self.process.stdout.close()
        self.stderr.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=0.5)
    parser.add_argument('--minimum-available-gib', type=float, default=2.0)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('provide a startup or benchmark command after --')
    if not math.isfinite(args.interval) or not 0.1 <= args.interval <= 5:
        parser.error('--interval must be between 0.1 and 5 seconds')
    if not math.isfinite(args.minimum_available_gib) or args.minimum_available_gib < 2:
        parser.error('--minimum-available-gib must be at least 2 GiB')
    config = fleet.load_config(args.config)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf8')
    samplers = []
    code = None
    failure = None
    minimum = math.ceil(args.minimum_available_gib * GIB)
    try:
        for rank in range(4):
            samplers.append(Sampler(config, rank, args.output, args.interval))
        deadline = time.monotonic() + 20
        for rank, sampler in enumerate(samplers):
            sampler.ready.wait(max(0, deadline - time.monotonic()))
            if not summarize(sampler.samples, minimum, sampler.error)['passed']:
                raise RuntimeError(f'rank {rank}: initial RAM qualification failed; command not started')
        code = subprocess.run(command).returncode
    except (Exception, KeyboardInterrupt) as exc:
        failure = str(exc) or type(exc).__name__
    finally:
        for sampler in samplers:
            sampler.stop()
        ranks = {str(rank): summarize(s.samples, minimum, s.error)
                 for rank, s in enumerate(samplers)}
        report = {'command': command, 'returncode': code, 'error': failure,
                  'minimum_required_gib': args.minimum_available_gib,
                  'interval_seconds': args.interval, 'ranks': ranks,
                  'passed': code == 0 and failure is None and len(ranks) == 4
                            and all(r['passed'] for r in ranks.values()),
                  'scope': 'Sampled host MemAvailable and swap; not a hard memory reservation. '
                           'A breach fails qualification but does not stop serving containers.'}
        (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
        print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
