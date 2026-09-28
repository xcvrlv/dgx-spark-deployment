#!/usr/bin/env python3
"""Run bounded fleet tuning with memory, progress and elapsed-time guards.

Requires the serving containers to be stopped first. On failure, stop only
containers using this candidate image and retain them and their logs. A sampled
guard is not a hard physical-memory reservation and can miss very short peaks.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
import urllib.request

import fleet

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('memory_watch', HERE / 'memory-watch.py')
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)

STATUS_PROBE = r'''
import datetime, json, subprocess, sys
from pathlib import Path
name, cache, expected = sys.argv[1:]
p = subprocess.run(['docker', 'container', 'inspect', name], capture_output=True, text=True, timeout=10)
if p.returncode:
    print(json.dumps({'pending': True})); sys.exit(0)
c = json.loads(p.stdout)[0]
if c['Image'] != expected:
    raise RuntimeError('Serving container image differs from the candidate')
started = datetime.datetime.fromisoformat(c['State']['StartedAt'].replace('Z', '+00:00')).timestamp()
files = sorted((Path(cache) / 'b12x-preparation-trace').glob('ds41-progress-*.json'), key=lambda f: f.stat().st_mtime, reverse=True)
progress = None
for file in files:
    if file.stat().st_mtime < started:
        continue
    value = json.loads(file.read_text())
    if value['time'] >= started:
        progress = value; break
print(json.dumps({'id': c['Id'], 'running': c['State']['Running'],
                  'oom': c['State']['OOMKilled'], 'progress': progress}))
'''


def status(config, rank, image_id):
    command = shlex.join(['python3', '-c', STATUS_PROBE, f'{fleet.NAME}-{rank}',
                          config['cache_path'], image_id])
    return json.loads(fleet.remote(config, rank, command, timeout=25))


def fingerprint(states):
    keys = ('job_started', 'phase', 'request', 'completed', 'measured', 'cached',
            'compiled', 'prepared', 'batch', 'round', 'compile_plans', 'done', 'failed')
    return tuple(tuple(state.get('progress', {}).get(k) for k in keys)
                 if state.get('progress') else (state.get('pending'), state.get('running'))
                 for state in states)


def memory_failure(samplers, now, stop_bytes=3 * watch.GIB, swap_baselines=None):
    for rank, sampler in enumerate(samplers):
        if sampler.error or not sampler.samples:
            return f'rank {rank}: memory sampler failed: {sampler.error}'
        if sampler.last_received is None or now - sampler.last_received > sampler.maximum_gap:
            return f'rank {rank}: memory samples are stale'
        # Check every received sample, not merely the last one.
        if any(s['available_bytes'] < stop_bytes for s in sampler.samples):
            return f'rank {rank}: MemAvailable fell below the 3 GiB stop threshold'
        # Host-wide paging can occur with ample physical headroom. It cannot
        # identify the serving process or prove exhaustion; report it separately.
    return None


def memory_summary(sampler, swap_baseline=None):
    report = watch.summarize(sampler.samples, 2 * watch.GIB, sampler.error)
    if sampler.samples:
        baseline = sampler.samples[0] if swap_baseline is None else swap_baseline
        report['swap_monitor_started_at'] = baseline['time']
        report['monitored_swap_out_pages'] = sampler.samples[-1]['swap_out_pages'] - baseline['swap_out_pages']
        report['swap_policy'] = 'Recorded separately; physical headroom determines memory qualification.'
        report['passed'] = (sampler.error is None
                            and report['minimum_available_gib'] >= 2)
    return report


def stop_candidate(config, rank, image_id, worker_pid=None):
    # Inspect before stopping, so an unrelated image is never interrupted.
    program = r'''
import json, subprocess, sys
name, expected, worker = sys.argv[1:]
p = subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=10)
if p.returncode == 0:
    c = json.loads(p.stdout)[0]
    if c['Image'] == expected and c['State']['Running']:
        if worker.isdigit() and int(worker) > 1:
            subprocess.run(['docker','exec',c['Id'],'kill','-USR1',worker],capture_output=True,timeout=5)
        subprocess.run(['docker','stop','--time','10',c['Id']],check=True,timeout=25)
'''
    return fleet.remote(config, rank, shlex.join(['python3', '-c', program,
                        f'{fleet.NAME}-{rank}', image_id, str(worker_pid or '')]), timeout=40)


def collect_logs(config, output):
    for rank in range(4):
        try:
            log = fleet.remote(config, rank, shlex.join(
                ['docker', 'logs', '--tail', '30000', f'{fleet.NAME}-{rank}']) + ' 2>&1', timeout=40)
            (output / f'rank-{rank}-container.log').write_text(log, encoding='utf8')
        except Exception as exc:
            (output / f'rank-{rank}-log-error.txt').write_text(str(exc))


def parse_iteration_histogram(content):
    buckets = {}
    for line in content.splitlines():
        if not line.startswith('vllm:iteration_tokens_total_bucket{'):
            continue
        match = re.search(r'\ble="([^"]+)"', line)
        if match:
            ceiling = float(match.group(1))
            buckets[ceiling] = buckets.get(ceiling, 0) + float(line.rsplit(' ', 1)[1])
    if not all(ceiling in buckets for ceiling in (4096.0, 8192.0, float('inf'))):
        raise RuntimeError('Missing serving iteration-token histogram buckets')
    return {'observations_above_4096_up_to_8192': buckets[8192.0] - buckets[4096.0],
            'observations_above_8192': buckets[float('inf')] - buckets[8192.0]}


def serving_batch_histogram(config):
    url = f"http://{config['nodes'][0]['ip']}:{config['port']}/metrics"
    with urllib.request.urlopen(url, timeout=10) as response:
        return parse_iteration_histogram(response.read().decode())


def batch_route_evidence(config):
    if not config.get('prefill_8192_graph'):
        return None
    probe = r'''
import json, subprocess, sys
name = sys.argv[1]
result = subprocess.run(['docker', 'inspect', name], check=True,
                        capture_output=True, text=True, timeout=10)
container = json.loads(result.stdout)[0]
environment = dict(item.split('=', 1) for item in container['Config']['Env'] if '=' in item)
command = container['Config']['Cmd']
logs = subprocess.run(['docker', 'logs', name], check=True,
                      capture_output=True, text=True, timeout=30)
content = logs.stdout + logs.stderr
print(json.dumps({
    'environment': environment.get('DS41_PREFILL_8192_GRAPH'),
    'command_budget': command[command.index('--max-num-batched-tokens') + 1],
    'captured_8192': 'Capturing CUDA graphs PIECEWISE tokens=8192 reqs=1' in content,
    'runtime_eligible_8192': 'DS41 runtime 8192-token prefill graph eligible' in content,
    'serving_replay_8192': 'DS41 serving 8192-token PIECEWISE graph replay active' in content,
}))
'''
    def one(rank):
        command = shlex.join(['python3', '-c', probe, f'{fleet.NAME}-{rank}'])
        data = json.loads(fleet.remote(config, rank, command, timeout=45))
        # vLLM emits per-graph debug accounting on rank 0 only. Every rank
        # still has to receive the limit and replay the graph outside dummy runs.
        if (data['environment'] != '1' or data['command_budget'] != '8192'
                or not data['runtime_eligible_8192']
                or not data['serving_replay_8192']
                or (rank == 0 and not data['captured_8192'])):
            raise RuntimeError(f'rank {rank}: 8192 batch/graph path not confirmed: {data}')
        return data
    with ThreadPoolExecutor(max_workers=4) as pool:
        ranks = {str(rank): value for rank, value in enumerate(pool.map(one, range(4)))}
    # Prompt statistics can accumulate across prefill chunks before an output
    # is reported. Keep these observations as diagnostics, never route proof.
    try:
        histogram = serving_batch_histogram(config)
    except Exception as exc:
        histogram = {'unavailable': str(exc)}
    return {'ranks': ranks, 'serving_batch_histogram': histogram}


def run(config_path, output, *, stall_seconds=900, total_seconds=7200):
    config = fleet.load_config(config_path)
    if not config.get('b12x_bounded_autotune'):
        raise ValueError('Use a bounded-autotune recipe built with build-autotune.py')
    if config['max_num_batched_tokens'] != 4096 and not (
            config['max_num_batched_tokens'] == 8192
            and config.get('prefill_8192_graph') is True):
        raise ValueError('Qualify only the audited 4096 or opt-in 8192-token prefill graph')
    if stall_seconds < 60 or total_seconds < stall_seconds:
        raise ValueError('Invalid watchdog time limits')
    output = Path(output)
    if output.exists():
        raise ValueError('Use a new results directory')
    image_id = subprocess.check_output(['docker', 'image', 'inspect', '--format', '{{.Id}}', config['image']], text=True).strip()
    for rank in range(4):
        fleet.remote(config, rank, f'! docker container inspect {fleet.NAME}-{rank} >/dev/null 2>&1')
    output.mkdir(parents=True)
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    samplers, states = [], []
    swap_baselines = [None] * 4
    failure = None
    batch_route = None
    process = None
    phases = []
    began = last_progress = time.monotonic()
    last_fingerprint = None
    next_status = 0.0
    try:
        for rank in range(4):
            sampler = watch.Sampler(config, rank, output, .5)
            samplers.append(sampler)
            if not sampler.ready.wait(20) or sampler.error:
                raise RuntimeError(f'rank {rank}: memory sampler unavailable')
        commands = [
            ('startup', [sys.executable, str(HERE / 'fleet.py'), '--config', str(config_path), 'start']),
            ('prefill-128k', [sys.executable, str(HERE / 'benchmark.py'), '--config', str(config_path),
                             '--input-tokens', '128519', '--max-tokens', '64', '--concurrency', '1',
                             '--requests', '3', '--output', str(output / 'benchmark-128k.json')]),
        ]
        for phase, command in commands:
            print(f'{phase}: logging to {output / (phase + ".log")}', flush=True)
            with (output / (phase + '.log')).open('w') as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                while True:
                    now = time.monotonic()
                    reason = memory_failure(samplers, now, swap_baselines=swap_baselines)
                    if reason:
                        raise RuntimeError(reason)
                    if now - began > total_seconds:
                        raise TimeoutError('Autotune qualification exceeded its total time limit')
                    if now >= next_status or process.poll() is not None:
                        with ThreadPoolExecutor(max_workers=4) as pool:
                            states = list(pool.map(lambda rank: status(config, rank, image_id), range(4)))
                        (output / 'latest-progress.json').write_text(json.dumps(states, indent=2))
                        current = fingerprint(states)
                        if current != last_fingerprint:
                            last_progress, last_fingerprint = time.monotonic(), current
                            for rank, state in enumerate(states):
                                p = state.get('progress') or {}
                                if p:
                                    print(f"rank {rank}: {p['phase']} {p['completed']}/{p['total']}; "
                                          f"measured={p['measured']} compiled={p['compiled']} {p['request']}", flush=True)
                        for rank, state in enumerate(states):
                            p = state.get('progress') or {}
                            if p and swap_baselines[rank] is None:
                                # Checkpoint loading can page out host memory with
                                # tens of GiB still available. Report swap writes
                                # from the first observed preparation record onward;
                                # keep full-startup counters and headroom too.
                                swap_baselines[rank] = samplers[rank].samples[-1].copy()
                                print(f'rank {rank}: preparation swap monitor armed', flush=True)
                            if state.get('oom') or state.get('running') is False or p.get('failed') or p.get('tuning_stopped'):
                                raise RuntimeError(f'rank {rank}: failed/stopped preparation or worker exit: {state}')
                        next_status = time.monotonic() + 5
                    if phase == 'startup' and time.monotonic() - last_progress > stall_seconds:
                        raise TimeoutError('No meaningful fleet preparation progress within the stall limit')
                    if process.poll() is not None:
                        if process.returncode:
                            raise RuntimeError(f'{phase} exited {process.returncode}; see its log')
                        phases.append(phase)
                        process = None
                        break
                    time.sleep(1)
        if not all(state.get('progress') for state in states):
            raise RuntimeError('Missing bounded-autotune progress evidence on one or more ranks')
        totals = [state['progress']['totals'] for state in states]
        if sum(item['measured'] + item['cache_hits'] for item in totals) == 0:
            raise RuntimeError('No measured candidates or cached selections confirmed')
        batch_route = batch_route_evidence(config)
        if batch_route is not None:
            phases.append('8192-route')
    except (Exception, KeyboardInterrupt) as exc:
        failure = str(exc) or type(exc).__name__
        print(f'Qualification stopped: {failure}', flush=True)
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(stop_candidate, config, rank, image_id,
                                   (states[rank].get('progress') or {}).get('pid') if rank < len(states) else None)
                       for rank in range(4)]
            for future in futures:
                try:
                    future.result()
                except Exception as stop_error:
                    failure += f'; container stop failed: {stop_error}'
    finally:
        for sampler in samplers:
            try:
                sampler.stop()
            except Exception as exc:
                sampler.error = sampler.error or f'Memory monitor cleanup failed: {exc}'
        memory = {str(rank): memory_summary(s, swap_baselines[rank])
                  for rank, s in enumerate(samplers)}
        report = {'time': datetime.now(timezone.utc).isoformat(), 'image_id': image_id,
                  'error': failure, 'completed_phases': phases, 'memory': memory, 'states': states,
                  'batch_route': batch_route,
                  'passed': failure is None and len(memory) == 4 and all(x['passed'] for x in memory.values()),
                  'scope': 'Sampled startup, c16 smoke and three 128k requests; not a hard RAM guarantee or a throughput comparison.'}
        if failure is None and not report['passed']:
            report['error'] = 'Final memory-monitor validation failed'
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(stop_candidate, config, rank, image_id) for rank in range(4)]
                for future in futures:
                    try:
                        future.result()
                    except Exception as stop_error:
                        report['error'] += f'; container stop failed: {stop_error}'
        (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
        collect_logs(config, output)
        print(json.dumps(report, indent=2), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stall-seconds', type=int, default=900)
    parser.add_argument('--total-seconds', type=int, default=7200)
    args = parser.parse_args()
    sys.exit(run(args.config, args.output, stall_seconds=args.stall_seconds, total_seconds=args.total_seconds))
