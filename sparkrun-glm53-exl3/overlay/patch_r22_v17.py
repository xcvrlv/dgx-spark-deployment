"""Pinned startup-memory and prefix metadata patch over v16."""
import hashlib
from pathlib import Path

VERSION = 'glm53-r22-v17-1'
WORKER = 'v1/worker/gpu_worker.py'
POOL = 'v1/core/block_pool.py'
HELPER = 'v1/worker/gb10_startup_memory.py'
INPUTS = {'v1/worker/gpu_worker.py': '2daaaf430c450ab0839516110f89a81e9952ec6286f25856e518c4b7d5c8e8e1', 'v1/core/block_pool.py': 'daf37f8c3e85d6debac02df2ac2356f5500555b613a369e8efdc66a0db94286b'}
OUTPUTS = {'v1/worker/gpu_worker.py': '3ebe27d0fe540d19769dd334f8a94b1fb4af5e4ac2c887300527ea933350301f', 'v1/core/block_pool.py': 'b5800600205b822bdc166e407956db2601169edc1fe5bc7b53735de945076f3f', 'v1/worker/gb10_startup_memory.py': '332b199ca952859020971155eea44bacb84c6b4ca432cd7579354e5296538767'}


def replace(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f'v17 anchor mismatch: {old[:90]!r}')
    return source.replace(old,new,1)


def worker(source):
    source = replace(source, 'import gc\n',
                     'import gc\nfrom vllm.v1.worker.gb10_startup_memory import reclaim_startup_memory\n')
    source = replace(source,
        '            self.model_runner.load_model(load_dummy_weights=load_dummy_weights)\n',
        '            self.model_runner.load_model(load_dummy_weights=load_dummy_weights)\n'
        '\n        reclaim_startup_memory("after-model-load", self.device)\n')
    source = replace(source,
        '        with self._maybe_get_memory_pool_context(tag="kv_cache"):\n',
        '        reclaim_startup_memory("before-kv-allocation", self.device)\n'
        '        with self._maybe_get_memory_pool_context(tag="kv_cache"):\n')
    source = replace(source, '        kernel_warmup(self)\n',
                     '        kernel_warmup(self)\n'
                     '        reclaim_startup_memory("after-kernel-warmup", self.device)\n')
    source = replace(source, '            cuda_graph_memory_bytes = self.model_runner.capture_model()\n',
                     '            cuda_graph_memory_bytes = self.model_runner.capture_model()\n'
                     '        reclaim_startup_memory("after-graph-capture", self.device)\n')
    return source


def pool(source):
    return replace(source,
        '        new_block_hashes = block_hashes[num_cached_blocks:]\n',
        '        # Future prompt hashes are not registered by this update.\n'
        '        new_block_hashes = block_hashes[num_cached_blocks:num_full_blocks]\n')


def patch(root, check=False):
    root = Path(root)
    pending = []
    for name, transform in ((WORKER,worker),(POOL,pool)):
        path = root/name
        source = path.read_text(encoding='utf-8')
        digest = hashlib.sha256(source.encode()).hexdigest()
        if digest == OUTPUTS[name]:
            continue
        if check or digest != INPUTS[name]:
            raise RuntimeError(f'unexpected v17 source: {path} ({digest})')
        result = transform(source)
        if hashlib.sha256(result.encode()).hexdigest() != OUTPUTS[name]:
            raise RuntimeError(f'v17 transformed hash mismatch: {name}')
        compile(result,str(path),'exec')
        pending.append((path,result))
    helper = Path(__file__).with_name('gb10_startup_memory.py').read_text(encoding='utf-8')
    if hashlib.sha256(helper.encode()).hexdigest() != OUTPUTS[HELPER]:
        raise RuntimeError('v17 helper hash mismatch')
    path = root/HELPER
    if path.exists():
        if path.read_text(encoding='utf-8') != helper:
            raise RuntimeError(f'v17 helper mismatch: {path}')
    elif check:
        raise RuntimeError(f'v17 helper missing: {path}')
    else:
        compile(helper,str(path),'exec')
        pending.append((path,helper))
    for path,source in pending:
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(source,encoding='utf-8',newline='\n')
