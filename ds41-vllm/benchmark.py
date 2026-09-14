#!/usr/bin/env python3
"""Uncached streaming measurements for same-image communication/prefill A/B runs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import time
import urllib.request
import uuid

import fleet


def measure(config, tokens, max_tokens):
    url = f"http://{config['nodes'][0]['ip']}:{config['port']}/v1/completions"
    payload = {'model': fleet.MODEL, 'prompt': tokens, 'max_tokens': max_tokens,
               'temperature': 0, 'stream': True, 'stream_options': {'include_usage': True}}
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    start = time.monotonic()
    first = None
    usage = None
    with urllib.request.urlopen(req, timeout=1800) as response:
        for line in response:
            if not line.startswith(b'data: '):
                continue
            data = line[6:].strip()
            if data == b'[DONE]':
                break
            event = json.loads(data)
            if event.get('error'):
                raise RuntimeError(event['error'])
            if event.get('usage'):
                usage = event['usage']
            if first is None and any(c.get('text') for c in event.get('choices', [])):
                first = time.monotonic()
    end = time.monotonic()
    assert first is not None and usage and usage['completion_tokens'] > 0, 'Incomplete generation'
    return {'ttft_seconds': first - start, 'latency_seconds': end - start,
            'prompt_tokens': usage['prompt_tokens'], 'completion_tokens': usage['completion_tokens'],
            'decode_tokens_per_second': (usage['completion_tokens'] - 1) / max(end - first, 1e-9)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(fleet.HERE / 'cluster-r38-c8.json'))
    parser.add_argument('--input-tokens', type=int, default=8192)
    parser.add_argument('--max-tokens', type=int, default=128)
    parser.add_argument('--concurrency', type=int, default=1)
    parser.add_argument('--requests', type=int, default=3)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    config = fleet.load_config(args.config)
    assert 0 < args.concurrency <= config['max_num_seqs']
    assert args.requests > 0 and args.max_tokens > 0 and args.input_tokens > 0
    assert args.input_tokens + args.max_tokens <= config['max_model_len']
    prompts = []
    # Unique leading nonce prevents prefix-cache reuse within/across A/B runs.
    for i in range(args.requests):
        text = f'Document {uuid.uuid4().hex}:\n' + ('The observatory records the weather and reviews its measurements each morning.\n' * args.input_tokens)
        tokens = fleet.request(config, '/tokenize', {'model': fleet.MODEL, 'prompt': text})['tokens']
        assert len(tokens) >= args.input_tokens
        prompts.append(tokens[:args.input_tokens])
    image_ids = [fleet.remote(config, rank, 'docker image inspect --format ' +
                  fleet.shlex.quote('{{.Id}}') + ' ' + fleet.shlex.quote(config['image'])) for rank in range(4)]
    assert len(set(image_ids)) == 1
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(lambda tokens: measure(config, tokens, args.max_tokens), prompts))
    wall = time.monotonic() - start
    report = {'config': config, 'image_id': image_ids[0], 'concurrency': args.concurrency,
              'requests': results, 'wall_seconds': wall,
              'median_ttft_seconds': statistics.median(r['ttft_seconds'] for r in results),
              'aggregate_output_tokens_per_second': sum(r['completion_tokens'] for r in results) / wall}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('config', 'requests')}, indent=2))
