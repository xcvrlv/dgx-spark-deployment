"""Validate original config, local shard completeness and native FP8 Engram headers."""
import hashlib
import json
import struct
import sys
from pathlib import Path

CONFIG_SHA256 = '8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879'


def validate(root):
    root = Path(root).resolve(strict=True)
    raw = (root / 'config.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == CONFIG_SHA256, 'Not the pinned original config.json'
    config = json.loads(raw)
    assert config['quantization_config']['expert_dtype'] == 'fp4'
    index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
    headers = {}
    fp8_tables = 0
    for filename in sorted(set(index.values())):
        shard = (root / filename).resolve(strict=True)
        # HF snapshots may legitimately resolve to the mounted cache root's blobs.
        with shard.open('rb') as stream:
            length = struct.unpack('<Q', stream.read(8))[0]
            assert 0 < length < 128 * 1024 * 1024, filename
            header = json.loads(stream.read(length))
        headers[filename] = header
        end = max(v['data_offsets'][1] for k, v in header.items() if k != '__metadata__')
        assert shard.stat().st_size == 8 + length + end, f'Truncated/invalid shard: {shard}'
        for name, tensor in header.items():
            if '.engram.' in name and name.endswith(('.embed.weight', '.embed_tokens.weight')):
                assert tensor['dtype'] == 'F8_E4M3', (name, tensor)
                assert tensor['shape'][1] == 256, (name, tensor)
                fp8_tables += 1
    for name, filename in index.items():
        assert name in headers[filename], (name, filename)
    assert fp8_tables == len(config['text_config']['engram_layer_ids']), fp8_tables
    print(f'Original MXFP4/FP8 checkpoint: {len(headers)} complete shards, {fp8_tables} FP8 Engram tables')


if __name__ == '__main__':
    validate(sys.argv[1])
