#!/usr/bin/env python3
"""World-coordinate the RoCE collective priming; checked against vllm 5bca5a5 (target unchanged since c9dc4e5) on 2026-09-15.

The RoCE prepare call primes a real four-rank all-reduce and all-gather
(``_preparation.prepared_call`` -> ``state.all_reduce``). The session only
yields the world-coordination requirement for requests that declare a
collective (``PreparationJob._run``), so without a declared
CollectiveRequirement every rank primes uncoordinated whenever it reaches the
request. Rank skew near the end of the weights batch then leaves the first
launch waiting for a slower peer until the kernel spin limit times out
(sequence 1) and poisons the runtime: "RoCE collective on rank 1 timed out
waiting for rank 0". Declaring the collective requirement makes the
coordinator authorize it only when every participant rank reported ready, so
all ranks prime in the same advance round. Per-rank tuning is disabled
because a real collective cannot be raced per rank and ``PreparationJob._run``
raises for collective declarations in tuned batches.
"""
import argparse
import hashlib
from pathlib import Path

RELATIVE = 'distributed/device_communicators/b12x_roce_all_reduce.py'
SOURCE_SHA = 'b8987d09ee05eb821f1d396b83a1e7e9b72ffc629e3e63d925bf581e0c214d97'
IMPORTS_OLD = (
    b'        from b12x.comm import roce\n'
    b'        from b12x.comm.roce import _preparation\n'
)
IMPORTS_NEW = (
    b'        from b12x.comm import roce\n'
    b'        from b12x.comm.roce import _preparation\n'
    b'        from b12x.preparation import CollectiveRequirement\n'
)
REQUEST_OLD = (
    b'        request = self._plan.request(\n'
    b'            name=self._request_name(),\n'
    b'            prepare_call=prepare,\n'
    b'        )\n'
)
REQUEST_NEW = (
    b'        request = self._plan.request(\n'
    b'            name=self._request_name(),\n'
    b'            prepare_call=prepare,\n'
    b'            collective=CollectiveRequirement(\n'
    b'                key=self._request_name(),\n'
    b'                ranks=tuple(sorted(self.global_ranks)),\n'
    b'            ),\n'
    b'        )\n'
)
TUNE_OLD = (
    b'                stage="weights",\n'
    b'                autotune=not workload.eager_only,\n'
)
TUNE_NEW = (
    b'                stage="weights",\n'
    b'                autotune=False,\n'
)
REPLACEMENTS = (
    (IMPORTS_NEW, IMPORTS_OLD),
    (REQUEST_NEW, REQUEST_OLD),
    (TUNE_NEW, TUNE_OLD),
)


def patch(root, *, check=False, revert=False):
    path = Path(root) / RELATIVE
    data = path.read_bytes()
    original = data
    for new, old in REPLACEMENTS:
        if original.count(new) == 1:
            original = original.replace(new, old, 1)
    if hashlib.sha256(original).hexdigest() != SOURCE_SHA:
        raise RuntimeError(f'Unexpected upstream source: {path}; re-audit before patching')
    for new, old in REPLACEMENTS:
        if original.count(old) != 1:
            raise RuntimeError(f'Unexpected upstream source: {path}; re-audit before patching')
    expected = original
    if not revert:
        for new, old in REPLACEMENTS:
            expected = expected.replace(old, new, 1)
    if check and data != expected:
        raise RuntimeError(f'Unexpected patch state: {path}')
    if not check and data != expected:
        compile(expected, str(path), 'exec')
        path.write_bytes(expected)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('package', type=Path)
    p.add_argument('--check', action='store_true')
    p.add_argument('--revert', action='store_true')
    a = p.parse_args()
    patch(a.package, check=a.check, revert=a.revert)
