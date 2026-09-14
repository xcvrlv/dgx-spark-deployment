#!/usr/bin/env python3
"""Hash-guarded DSpark request-bucket graph coverage for pinned JJ.

Re-audited 2026-09-14 13:39 UTC on JJ c9dc4e5; only the profiling
preparation callback changed since the original audited graph planner. Runtime rollback: DS41_GRAPH_REQUEST_BUCKETS=0.
"""
import argparse
import hashlib
from pathlib import Path

RELATIVE = 'v1/worker/gpu/cudagraph_utils.py'
SOURCE_SHAS = {'b76823c1d74eca1423d83252412c9e9a50173352dfc31dbb66afbeca09a25b64',
               'bef8f12751d8115eb2e52c9e1387276a82804f5c5931e0c1547d41321e086534'}
ANCHOR = '''        for num_tokens, num_active_loras in product(
            capture_sizes, self.lora_capture_cases
        ):
'''
INSERT = '''        # DS41 request buckets: cover mixed verification lengths at each
        # request capacity, without duplicating every request count up to c16.
        # The ordinary fallback and its dispatch ordering remain unchanged.
        if (
            capture_varlen_decode
            and speculative_config is not None
            and speculative_config.use_dspark()
            and os.getenv("DS41_GRAPH_REQUEST_BUCKETS", "0") == "1"
        ):
            request_capacities = set(range(1, min(8, self.max_num_reqs) + 1))
            request_capacities.add(self.max_num_reqs)
            for num_active_loras, request_capacity in product(
                self.lora_capture_cases, sorted(request_capacities)
            ):
                # Every request has at least one query; all other legal totals
                # include heterogeneous per-request speculative widths.
                for num_tokens in range(
                    request_capacity,
                    min(request_capacity * self.decode_query_len,
                        max_cg_capture_size) + 1,
                ):
                    desc = BatchExecutionDescriptor(
                        cg_mode=decode_mode,
                        num_tokens=num_tokens,
                        num_reqs=request_capacity,
                        max_query_len=self.decode_query_len,
                        num_active_loras=num_active_loras,
                    )
                    if desc not in descs_by_mode[decode_mode]:
                        descs_by_mode[decode_mode].append(desc)
            logger.info(
                "DS41 DSpark graph request buckets=%s, max query width=%d, token cap=%d",
                sorted(request_capacities), self.decode_query_len, max_cg_capture_size,
            )
'''


def transform(data):
    if data.count(ANCHOR.encode()) != 1:
        raise RuntimeError('Graph source anchor mismatch')
    return data.replace(ANCHOR.encode(), (INSERT + ANCHOR).encode(), 1)


def patch(root, *, check=False, revert=False):
    path = Path(root) / RELATIVE
    data = path.read_bytes()
    original = data.replace(INSERT.encode(), b'', 1) if data.count(INSERT.encode()) == 1 else data
    if hashlib.sha256(original).hexdigest() not in SOURCE_SHAS:
        raise RuntimeError(f'Unexpected upstream source: {path}; audit newer upstream first')
    expected = original if revert else transform(original)
    if check and data != expected:
        raise RuntimeError(f'Unexpected graph patch state: {path}')
    if not check and data != expected:
        path.write_bytes(expected)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package', type=Path)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--revert', action='store_true')
    args = parser.parse_args()
    patch(args.package, check=args.check, revert=args.revert)
