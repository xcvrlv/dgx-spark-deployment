#!/usr/bin/env python3
"""Check the display-reserve KV span inside the container; fails closed.

Run by fleet.py preflight with the DRM card exposed, in the idiom of the
existing roce-check.py. It refuses rather than falling back to ordinary RAM,
because at this utilization the credited block count no longer fits there.

Exit 0 only if the span exists, CUDA can write and read it, and ordinary RAM
did not absorb it. Prints one JSON receipt either way.
"""
import importlib.util
import json
import sys
from pathlib import Path

HELPER = 'v1/worker/ds41_display_kv.py'


def helper_path():
    """Locate the installed helper without importing all of vllm."""
    from importlib.metadata import distribution
    return Path(distribution('vllm').locate_file('vllm'))/HELPER


def mem_available_bytes():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('No MemAvailable in /proc/meminfo')


def main():
    spec = importlib.util.spec_from_file_location('ds41_display_kv', helper_path())
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    display = helper.configured_bytes()
    if not display:
        print(json.dumps({'display_kv': 'disabled'}))
        return 0
    import torch

    torch.cuda.init()
    before = mem_available_bytes()
    owner = helper.Owner(ordinary_bytes=0)
    tensor = owner.tensor()
    tensor[:display].fill_(0x5A)
    torch.cuda.synchronize()
    if not bool((tensor[:display] == 0x5A).all()):
        raise RuntimeError('CUDA could not write and read the display span')
    absorbed = before - mem_available_bytes()
    if absorbed > display // 2:
        raise RuntimeError(
            f'The display span absorbed {absorbed} bytes of ordinary RAM; '
            'the credit is not backed by the display reserve'
        )
    print(json.dumps({'display_kv': 'ok', 'display_bytes': display,
                      'span_bytes': owner.size, 'ordinary_ram_absorbed': absorbed}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
