#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# H2D staging and GC handling adapted from FujitsuPolycom/sparkring
# 8b152d65c701f557f62ae6a9a1c3db90771c1df3. Copyright 2026 SparkRing contributors.
"""Source-guarded, independently switched SparkRing ports for Karmic 502d6cb."""
import argparse
import hashlib
from pathlib import Path

HELPER = 'preparation/ds41_gc.py'
COPY = b'        return gpu.copy_(cpu.pin_memory() if PIN_MEMORY else cpu, non_blocking=True)\n'
STAGED_COPY = b'''        if os.environ.get("DS41_H2D_STAGING", "0") != "1" or not PIN_MEMORY:
            return gpu.copy_(cpu.pin_memory() if PIN_MEMORY else cpu, non_blocking=True)
        # Snapshot mutable host rows before an async scheduler can rewrite them.
        # Torch's caching host allocator retains staging until the copy completes.
        staging = torch.empty_like(cpu, pin_memory=True)
        staging.copy_(cpu)
        return gpu.copy_(staging, non_blocking=True)
'''
HOLD = b'''    @contextmanager
    def hold(self, stream):
        self.streams[stream.cuda_stream] = stream
        self.sequence = (self.sequence + 1) & 0xFFFFFFFF
        target = self.sequence
        self._check(self.driver.cuStreamWaitValue32(
            stream.cuda_stream, self.device_pointer, target,
            int(self.driver.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_GEQ),
        ))
        try:
            yield
        finally:
            # A later release must also satisfy an earlier, still queued wait.
            self.flag.value = target
'''
GUARDED_HOLD = b'''    @contextmanager
    def hold(self, stream):
        with ds41_gc.defer_automatic_gc():
            self.streams[stream.cuda_stream] = stream
            self.sequence = (self.sequence + 1) & 0xFFFFFFFF
            target = self.sequence
            try:
                self._check(self.driver.cuStreamWaitValue32(
                    stream.cuda_stream, self.device_pointer, target,
                    int(self.driver.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_GEQ),
                ))
                yield
            finally:
                # Release queued work before automatic finalizers may resume.
                self.flag.value = target
'''
IPC_ASSIGNMENT = b'            self.busy_loop_s = busy_loop_s\n'
IPC_REPLACEMENT = b'''            if busy_loop_s is None:
                value = os.environ.get("DS41_SHM_BUSY_LOOP_S", "1")
                try:
                    busy_loop_s = float(value)
                except ValueError as error:
                    raise ValueError("DS41_SHM_BUSY_LOOP_S must be between 0 and 1") from error
                if not 0 <= busy_loop_s <= 1:
                    raise ValueError("DS41_SHM_BUSY_LOOP_S must be between 0 and 1")
            self.busy_loop_s = busy_loop_s
'''
PATCHES = {
    ('vllm', 'v1/utils.py'): ('5b6015d60ea96ab24b0f2e0b7701abaa90d6107bca01b51ac3d4bc361fbdfcff', [
        (b'import multiprocessing\n', b'import multiprocessing\nimport os\n'),
        (COPY, STAGED_COPY),
    ]),
    ('vllm', 'v1/core/block_pool.py'): ('720215b0508dbab462063b2a21d9f7ae1fdbf0bc3041f66ec9e19c6e2dd018d3', [
        (b'import weakref\n', b'import weakref\nimport os\n'),
        (b'        new_block_hashes = block_hashes[num_cached_blocks:]\n',
         b'        new_block_hashes = (\n'
         b'            block_hashes[num_cached_blocks:num_full_blocks]\n'
         b'            if os.environ.get("DS41_BOUNDED_PREFIX_HASHES", "0") == "1"\n'
         b'            else block_hashes[num_cached_blocks:]\n'
         b'        )\n'),
    ]),
    ('vllm', 'distributed/device_communicators/shm_broadcast.py'): ('240fd6a148aa6729e110380ca1188aaab6bc722e7c7ad5110c8aa301388bd967', [
        (b'        busy_loop_s: float = 1,\n', b'        busy_loop_s: float | None = None,\n'),
        (IPC_ASSIGNMENT, IPC_REPLACEMENT),
    ]),
    ('b12x', 'preparation/_measurement.py'): ('6eafe4fa0f6983de932ed5ae7e778d9907a30f37232465fa37f35ec2e5066850', [
        (b'from .types import _prime, _close_all\n',
         b'from .types import _prime, _close_all\nfrom . import ds41_gc\n'),
        (HOLD, GUARDED_HOLD),
    ]),
}


def transform(data, replacements, reverse=False):
    for old, new in reversed(replacements) if reverse else replacements:
        before, after = (new, old) if reverse else (old, new)
        if data.count(before) != 1:
            raise RuntimeError('SparkRing port anchor absent or ambiguous; re-audit upstream')
        data = data.replace(before, after, 1)
    return data


def patch(vllm, b12x, *, check=False, revert=False):
    roots = {'vllm': Path(vllm), 'b12x': Path(b12x)}
    helper = Path(__file__).with_name('sparkring_gc_runtime.py').read_bytes().replace(b'\r\n', b'\n')
    destination = roots['b12x'] / HELPER
    if destination.exists() and destination.read_bytes() != helper:
        raise RuntimeError('Unexpected installed GC helper')
    pending = []
    for (tree, relative), (digest, replacements) in PATCHES.items():
        path = roots[tree] / relative
        data = path.read_bytes()
        original = data if hashlib.sha256(data).hexdigest() == digest else transform(data, replacements, True)
        if hashlib.sha256(original).hexdigest() != digest:
            raise RuntimeError(f'Unexpected source: {path}; re-audit upstream')
        expected = original if revert else transform(original, replacements)
        compile(expected, str(path), 'exec')
        if check and data != expected:
            raise RuntimeError(f'Unexpected patch state: {path}')
        pending.append((path, expected))
    if check:
        if destination.exists() == revert:
            raise RuntimeError('Unexpected GC helper state')
        return
    for path, data in pending:
        path.write_bytes(data)
    if revert:
        destination.unlink(missing_ok=True)
    else:
        destination.write_bytes(helper)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('vllm', type=Path)
    parser.add_argument('b12x', type=Path)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--revert', action='store_true')
    args = parser.parse_args()
    patch(args.vllm, args.b12x, check=args.check, revert=args.revert)
