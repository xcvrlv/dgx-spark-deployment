"""Startup-only reclamation of unused allocator pages on unified-memory GB10."""
import ctypes
import gc
import json
import os
from pathlib import Path


def proc_kib(path, keys):
    try:
        lines = Path(path).read_text(encoding='utf-8').splitlines()
    except OSError:
        return {}
    result = {}
    for line in lines:
        key, _, value = line.partition(':')
        if key in keys:
            try:
                result[key] = int(value.split()[0])
            except (ValueError, IndexError):
                pass
    return result


def trim_heap():
    # No allocator replacement, process killing, or global drop_caches.
    # An absent glibc interface is a supported no-op.
    try:
        trim = ctypes.CDLL(None).malloc_trim
    except (AttributeError, OSError):
        return None
    trim.argtypes = [ctypes.c_size_t]
    trim.restype = ctypes.c_int
    return int(trim(0))


def snapshot(torch, device):
    return dict(
        proc_kib('/proc/meminfo', ('MemAvailable','MemFree','Cached','SwapFree')),
        **proc_kib('/proc/self/status', ('VmRSS','VmHWM')),
        cuda_free_bytes=int(torch.cuda.mem_get_info(device)[0]),
        torch_allocated_bytes=int(torch.cuda.memory_allocated(device)),
        torch_reserved_bytes=int(torch.cuda.memory_reserved(device)),
    )


def reclaim_startup_memory(stage, device):
    if os.getenv('VLLM_GB10_STARTUP_RECLAIM', '0') != '1':
        return None
    import torch
    from vllm.logger import init_logger
    if torch.cuda.get_device_capability(device) != (12, 1):
        return None
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('startup reclamation must not run inside CUDA capture')
    torch.cuda.synchronize(device)
    before = snapshot(torch, device)
    gc.collect()
    torch.cuda.empty_cache()
    trimmed = trim_heap()
    after = snapshot(torch, device)
    result = dict(stage=stage, pid=os.getpid(), heap_trim_result=trimmed,
                  before=before, after=after)
    init_logger(__name__).info('GB10 startup memory %s', json.dumps(result,sort_keys=True))
    return result
