# SPDX-License-Identifier: Apache-2.0
# Adapted from FujitsuPolycom/sparkring integrations/b12x/patches/stream_gate_gc.py
# at 8b152d65c701f557f62ae6a9a1c3db90771c1df3. Copyright 2026 SparkRing contributors.
"""Keep automatic cyclic finalizers outside host-controlled CUDA waits."""
from contextlib import contextmanager
import gc
import os
import threading

_lock = threading.RLock()
_holders = 0
_restore = False


@contextmanager
def defer_automatic_gc():
    global _holders, _restore
    if os.environ.get('DS41_DEFER_AUTOTUNE_GC', '0') != '1':
        yield
        return
    with _lock:
        if _holders == 0:
            _restore = gc.isenabled()
            gc.disable()
        _holders += 1
    try:
        yield
    finally:
        with _lock:
            _holders -= 1
            if _holders == 0 and _restore:
                gc.enable()
