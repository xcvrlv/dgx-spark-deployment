"""Opt-in preparation limits and progress records; never runs in serving kernels."""
import json
import importlib.util
import os
from pathlib import Path
import time

MIB = 1 << 20
PROFILE = 'ds41-bounded-autotune-v1'
_last_write = 0.0
_totals = {'measured': 0, 'cache_hits': 0, 'completed_jobs': 0}
_finished = set()


def memory_counter():
    # This image-built extension shares the image's exact Torch/CUDA ABI.
    # Import it directly: torch.load's persistent JIT lock can survive a killed
    # build and wait forever even when its .so already exists.
    path = Path('/opt/ds41/preparation-memory/b12x_preparation_memory.so')
    if not path.is_file():
        raise RuntimeError('Missing image-built b12x memory counter; rebuild Dockerfile.autotune')
    spec = importlib.util.spec_from_file_location('b12x_preparation_memory', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def enabled():
    return os.environ.get('DS41_B12X_BOUNDED_AUTOTUNE') == '1'


def positive_env(name, default, minimum, maximum):
    value = int(os.environ.get(name, default))
    if not minimum <= value <= maximum:
        raise ValueError(f'{name} must be in [{minimum}, {maximum}]')
    return value


def session_options():
    if not enabled():
        return {}
    return {'race_batch': 1, 'race_budget': positive_env(
        'DS41_B12X_RACE_BUDGET_MIB', 1024, 64, 4096) * MIB}


def available_bytes():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('Cannot read physical MemAvailable')


def reserve_bytes():
    return positive_env('DS41_B12X_RESERVE_MIB', 4096, 4096, 16384) * MIB


def require_headroom():
    if enabled() and available_bytes() < reserve_bytes():
        raise MemoryError('DS41 autotune physical memory reserve breached; stop and inspect the run')


def bounded_budget(requested, cuda_free, available, reserve):
    # This budgets candidate residency, not total process memory. Upstream
    # allocates a candidate before measuring its size; the external guard is
    # still needed, including during compilation and fixed-plan preparation.
    if available <= reserve or cuda_free <= 0:
        raise MemoryError('DS41 autotune has no memory above its physical reserve')
    return max(1, min(requested, cuda_free // 2, (available - reserve) // 2))


def write_progress(job, *, force=False):
    global _last_write
    if not enabled():
        return
    now = time.time()
    done = job._result is not None
    identity = (id(job), job._started)
    if done and identity not in _finished:
        _finished.add(identity)
        _totals['measured'] += job._benchmarked
        _totals['cache_hits'] += job._cache_hits
        _totals['completed_jobs'] += 1
    if not force and not done and job._error is None and now - _last_write < 1:
        return
    root = Path(os.environ['B12X_PREPARATION_TRACE_DIR'])
    root.mkdir(parents=True, exist_ok=True)
    compilations = job._compilations
    if job.session._pool is not None:
        summary = job.session._pool.summary()
        compilations = summary.cute_compilations + summary.triton_compilations
    payload = {
        'pid': os.getpid(), 'time': now, 'job_started': job._started,
        'phase': job._phase, 'request': getattr(job._active_request, 'name', None),
        'completed': job._completed_requests, 'total': job._total_requests,
        'measured': job._benchmarked, 'cached': job._cache_hits,
        'compiled': compilations, 'prepared': job._candidates_prepared,
        'batch': job._batch_index, 'round': job._completed_rounds,
        'compile_plans': job._timing.counts.get('compile_plan', 0),
        'done': done, 'failed': job._error is not None,
        'tuning_stopped': job.session._stop.is_set(), 'autotune': job.autotune,
        'totals': dict(_totals),
    }
    target = root / f'ds41-progress-{os.getpid()}.json'
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload) + '\n')
    temporary.replace(target)
    _last_write = now
