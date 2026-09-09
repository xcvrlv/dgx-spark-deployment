#!/usr/bin/env python3
"""Exact loader contract plus a small streaming/ownership/fallback GPU gate."""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import tempfile

VERSION = 'glm53-r22-v20-instanttensor-r1'
SOURCE_HASHES = {
    'instanttensor/_impl.py': 'c0ec4260383ddcd9124cf185cfff842eca16f1fce7608670ad948c12484d4d54',
    'vllm/model_executor/model_loader/weight_utils.py': '10bb69d9a1d9a1c419dd2aa4472e1a9626e55b27601d6f8da24438538d61a021',
    'vllm/model_executor/model_loader/default_loader.py': 'cd5fe07f7ec1a5872db320b956f88fc3356857f4dce5c560e747cd32528f3423',
}


def source_contract():
    for name, expected in SOURCE_HASHES.items():
        package, relative = name.split('/', 1)
        root = Path(importlib.util.find_spec(package).origin).parent
        observed = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        if observed != expected:
            raise RuntimeError(f'Unexpected loader source: {name}: {observed}')
    versions = {name: importlib.metadata.version(name)
                for name in ('torch', 'instanttensor')}
    if versions['instanttensor'] != '0.1.9':
        raise RuntimeError(f'Unexpected InstantTensor: {versions}')
    return dict(revision=VERSION, source_contract='passed', versions=versions)


@contextmanager
def smoke_environment():
    # Deliberately wrap the GPU ring several times and force a CPU fallback.
    settings = dict(INSTANTTENSOR_BACKEND='BUFFERED',
                    INSTANTTENSOR_BUFFER_SIZE='4194304',
                    INSTANTTENSOR_MAX_FREE_MEM_USAGE='0.05',
                    INSTANTTENSOR_CHUNK_SIZE='1048576',
                    INSTANTTENSOR_CONCURRENCY='1', INSTANTTENSOR_IO_DEPTH='2')
    previous = {key: os.environ.get(key) for key in settings}
    os.environ.update(settings)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def gpu_check():
    import torch
    from safetensors.torch import save_file
    from vllm.model_executor.model_loader.weight_utils import instanttensor_weights_iterator

    if torch.cuda.mem_get_info()[0] < 1024**3:
        raise RuntimeError('Loader smoke needs an idle GPU with at least 1 GiB free')
    with smoke_environment(), tempfile.TemporaryDirectory() as directory:
        expected = {f'w{i:02}': torch.full((512, 1024), i, dtype=torch.bfloat16)
                    for i in range(12)}
        expected['oversize'] = torch.ones((4 * 1024 * 1024,), dtype=torch.int16)
        filename = str(Path(directory, 'stream.safetensors').resolve())
        save_file(expected, filename)
        index = {name: filename for name in expected}
        # Retain every tensor beyond iterator/context close: copy=True must
        # survive ring reuse, not just compare correctly at the yield point.
        received = list(instanttensor_weights_iterator(
            [filename], False, copy=True, distributed=False,
            indexed_tensor_files=index))
        torch.cuda.synchronize()
        if len(received) != len(expected) or {name for name, _ in received} != set(expected):
            raise RuntimeError('Missing or duplicate streamed tensors')
        for name, tensor in received:
            if tensor.dtype != expected[name].dtype or not torch.equal(tensor.cpu(), expected[name]):
                raise RuntimeError(f'Streamed tensor mismatch: {name}')
            if getattr(tensor, '_vllm_instanttensor_borrowed', False):
                raise RuntimeError(f'Unexpected borrowed storage: {name}')
            expected_device = 'cpu' if name == 'oversize' else 'cuda'
            if tensor.device.type != expected_device:
                raise RuntimeError(f'Unexpected staging path: {name}: {tensor.device}')
        # Exercise index restriction before physical I/O as the real loader does.
        selected = list(instanttensor_weights_iterator(
            [filename], False, copy=True, distributed=False,
            indexed_tensor_files={'w03': filename, 'w09': filename}))
        if {name for name, _ in selected} != {'w03', 'w09'} or len(selected) != 2:
            raise RuntimeError('Index-aware loading selected the wrong tensors')
        for name, tensor in selected:
            if not torch.equal(tensor.cpu(), expected[name]):
                raise RuntimeError(f'Index-aware tensor mismatch: {name}')
    return dict(gpu_loader='passed', retained_tensors=13,
                cpu_fallback='passed', indexed_selection='passed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    result = source_contract()
    if args.gpu:
        result.update(gpu_check())
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
