#!/usr/bin/env python3
"""Per-node usable KV capacity for the ds41 serving target, with the display credit.

Answers "what is the actual usable vLLM memory per node, and what does the display
reserve add?" using the terms vLLM itself uses at this pin, so the table explains
the reported capacity instead of restating it:

    requested    = ceil(total_memory * gpu_memory_utilization)      utils.py:505
    ordinary_kv  = requested - non_kv_cache - late_persistent
                             - cudagraph_estimate               gpu_worker.py:657
    num_blocks   = (ordinary_kv + credit) // bytes_per_block       kv_cache_utils.py:1568
    concurrency  = num_blocks / blocks_per_request                 kv_cache_utils.py:1001
    capacity     = int(concurrency) * max_model_len                  kv_cache_utils.py:2240

Why the display credit is safe to add and this budget is not:

- The credit is *additive* and lives outside `requested`, so it does not consume
  the ordinary margin `gpu_memory_utilization` leaves. That is the whole point.
- `--kv-cache-memory-bytes` is the opposite: it "ignores gpu_memory_utilization"
  (entrypoints/llm.py:121). Setting it to the reported maximum would claim the
  whole budget with no utilization guard at all, which is the 100% case to avoid.

Run with a real `rank-N.log` from observe.py, or with explicit numbers.
"""
import argparse
import json
import math
import re
from pathlib import Path

GIB = 2**30
MIB = 2**20
DISPLAY_MAX_MIB = 1792
FLEET_NODES = 4

INITIAL = re.compile(r'Initial free memory: ([\d.]+) GiB; Requested memory: ([\d.]+) \(util\), ([\d.]+) GiB')
AFTER = re.compile(r'Free memory after profiling: ([\d.]+) GiB \(total\)')
AVAILABLE = re.compile(r'Available KV cache memory: ([\d.]+) GiB')
CAPACITY = re.compile(r'GPU KV cache size: ([\d,]+) tokens, Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x')


def parse(text):
    """Read the accounting anchors vLLM logs at this pin."""
    initial, after, available = INITIAL.search(text), AFTER.search(text), AVAILABLE.search(text)
    capacity = CAPACITY.search(text)
    if not (initial and after and available and capacity):
        raise ValueError('Missing KV accounting anchors; re-audit before trusting this')
    return {
        'total_gib': float(initial.group(1)),
        'utilization': float(initial.group(2)),
        'requested_gib': float(initial.group(3)),
        'free_after_gib': float(after.group(1)),
        'ordinary_kv_gib': float(available.group(1)),
        'tokens': int(capacity.group(1).replace(',', '')),
        'max_model_len': int(capacity.group(2).replace(',', '')),
        'concurrency': float(capacity.group(3)),
    }


def budget(measured, credit_mib, safety_mib=0):
    """Ordinary and credited KV budget, and the residual ordinary margin."""
    requested = math.ceil(measured['total_gib'] * measured['utilization'] * GIB)
    ordinary = math.ceil(measured['ordinary_kv_gib'] * GIB)
    # The non-KV terms are what vLLM subtracted; report them rather than assume.
    non_kv = requested - ordinary
    credit = max(credit_mib - safety_mib, 0) * MIB
    # `non_kv + ordinary == requested` by construction, so the ordinary claim is
    # exactly the utilization factor. The credit is not drawn from total_memory,
    # which is why it cannot push the ordinary claim past that factor.
    ordinary_claim = non_kv + ordinary
    return {
        'requested_gib': requested / GIB,
        'non_kv_cache_gib': non_kv / GIB,
        'ordinary_kv_gib': ordinary / GIB,
        'credit_gib': credit / GIB,
        'credited_kv_gib': (ordinary + credit) / GIB,
        'ordinary_claim_fraction_of_total': ordinary_claim / (measured['total_gib'] * GIB),
        'credited_claim_fraction_of_total': (ordinary_claim + credit) / (measured['total_gib'] * GIB),
        'residual_ordinary_gib': measured['total_gib'] * GIB - ordinary_claim,
    }


def predict(measured, extra_bytes):
    """Predicted capacity once `extra_bytes` more backing is credited.

    A request at max_model_len consumes a fixed pool share, so the extra bytes buy
    whole extra full-length requests. Whether that gain is realised depends on where
    the floors fall: 1.75 GiB is not itself a request's worth of KV, so near a
    boundary it can buy one and away from one it can buy none. Floor effects make
    this a prediction, not an identity, which is why the headroom test allows a
    tolerance.
    """
    ordinary = measured['ordinary_kv_gib'] * GIB
    per_request = ordinary * measured['max_model_len'] / measured['tokens']
    if per_request <= 0:
        raise ValueError('Reported capacity is zero; nothing to scale from')
    requests = int((ordinary + extra_bytes) / per_request)
    ordinary_requests = int(ordinary / per_request)
    return {
        'bytes_per_max_len_request': per_request,
        'requests': requests,
        'extra_requests': requests - ordinary_requests,
        'predicted_tokens': requests * measured['max_model_len'],
    }


def report(measured, credit_mib, safety_mib):
    b = budget(measured, credit_mib, safety_mib)
    credited_capacity = predict(measured, b['credit_gib'] * GIB)
    ordinary_capacity = predict(measured, 0)
    observed_per_request = predict(measured, 0)['bytes_per_max_len_request']
    return {
        'measured': measured,
        'budget': b,
        'ordinary_tokens': ordinary_capacity['predicted_tokens'],
        'credited_tokens': credited_capacity['predicted_tokens'],
        'token_gain': credited_capacity['predicted_tokens'] - ordinary_capacity['predicted_tokens'],
        'extra_full_requests': credited_capacity['extra_requests'],
        'per_request_gib': observed_per_request / GIB,
        'fleet': {
            'nodes': FLEET_NODES,
            'credit_gib': b['credit_gib'] * FLEET_NODES,
            'token_gain': (credited_capacity['predicted_tokens'] - ordinary_capacity['predicted_tokens']) * FLEET_NODES,
            'extra_full_requests': credited_capacity['extra_requests'] * FLEET_NODES,
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('log', nargs='?', type=Path, help='a rank-N.log from observe.py')
    p.add_argument('--total-gib', type=float, help='device total memory at startup')
    p.add_argument('--utilization', type=float)
    p.add_argument('--requested-gib', type=float)
    p.add_argument('--ordinary-kv-gib', type=float, help='the logged available KV budget')
    p.add_argument('--tokens', type=int)
    p.add_argument('--max-model-len', type=int)
    p.add_argument('--credit-mib', type=int, default=DISPLAY_MAX_MIB,
                    help=f'display bytes to credit; the span is fixed at {DISPLAY_MAX_MIB} MiB')
    p.add_argument('--safety-mib', type=int, default=256,
                    help='display bytes to hold back rather than credit')
    p.add_argument('--example', action='store_true',
                    help='run with clearly-labelled illustrative numbers, not measurements')
    a = p.parse_args()

    if a.log:
        measured = parse(a.log.read_text(errors='replace'))
        source = str(a.log)
    elif a.example:
        measured = {'total_gib': 121.66, 'utilization': 0.85, 'requested_gib': 103.41,
                    'free_after_gib': 102.0, 'ordinary_kv_gib': 100.0, 'tokens': 2744640,
                    'max_model_len': 393216, 'concurrency': 6.98}
        source = 'EXAMPLE NUMBERS, NOT MEASURED — pass a rank-N.log for real ones'
    else:
        measured = {
            'total_gib': a.total_gib, 'utilization': a.utilization,
            # requested is derivable, so derive it rather than demanding it.
            'requested_gib': a.requested_gib if a.requested_gib is not None
                             else a.total_gib * a.utilization,
            'ordinary_kv_gib': a.ordinary_kv_gib,
            'tokens': a.tokens, 'max_model_len': a.max_model_len,
            'concurrency': (a.tokens / a.max_model_len) if a.tokens else 0.0,
        }
        missing = [k for k, v in measured.items() if v is None or v == 0]
        if missing:
            p.error('provide a log or all of: ' + ', '.join(missing))
        source = 'explicit numbers'

    if not 0 <= a.credit_mib <= DISPLAY_MAX_MIB:
        p.error(f'--credit-mib must be 0..{DISPLAY_MAX_MIB}')
    if a.safety_mib < 0 or a.safety_mib >= a.credit_mib + 1:
        p.error('--safety-mib must be smaller than the credit')

    out = report(measured, a.credit_mib, a.safety_mib)
    out['source'] = source
    b, m = out['budget'], out['measured']
    print(json.dumps(out, indent=2))
    print()
    print(f"per-node ordinary KV budget : {b['ordinary_kv_gib']:8.3f} GiB"
          f"  ({m['concurrency']:.2f} x {m['max_model_len']:,} tokens)")
    print(f"per-node display credit     : {b['credit_gib']:8.3f} GiB"
          f"  ({a.credit_mib} MiB less {a.safety_mib} MiB held back)")
    print(f"per-node credited KV budget : {b['credited_kv_gib']:8.3f} GiB")
    print(f"ordinary claim of total     : {b['ordinary_claim_fraction_of_total']*100:8.2f}%"
          f"  (utilization {m['utilization']}, residual {b['residual_ordinary_gib']/GIB:.3f} GiB)")
    print(f"credited claim of total     : {b['credited_claim_fraction_of_total']*100:8.2f}%"
          f"  (the credit is outside total_memory)")
    print(f"one max-length request     : {out['per_request_gib']:8.3f} GiB of KV")
    print(f"capacity {out['ordinary_tokens']:,} -> {out['credited_tokens']:,} tokens"
          f"  (+{out['token_gain']:,}, +{out['extra_full_requests']} full requests)")
    print(f"fleet of {out['fleet']['nodes']}              : +{out['fleet']['credit_gib']:.3f} GiB,"
          f" +{out['fleet']['token_gain']:,} tokens,"
          f" +{out['fleet']['extra_full_requests']} full requests")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
