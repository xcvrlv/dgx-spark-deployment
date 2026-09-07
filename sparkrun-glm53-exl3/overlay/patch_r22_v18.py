"""Pinned B12X continuation-prefill overlay over v17."""
import hashlib
from pathlib import Path

VERSION = 'glm53-r22-v18-1'
COMMON = 'v1/attention/backends/mla/indexer.py'
INDEXER = 'v1/attention/backends/mla/b12x_indexer.py'
HELPER = 'v1/attention/backends/mla/gb10_indexer_prefill.py'
INPUTS = {COMMON: 'c77812d434e24ff9d9ed8495081b65d8770e633c1c6765a7688960f64a41f4c8',
          INDEXER: '6920327ca8ac67bd746ed96bdb71671ab84fd3991d9d4f274a76a7c8944c88c2'}
OUTPUTS = {'v1/attention/backends/mla/indexer.py': 'a0c0d701dfc85c51deb009f0e614c55cbd5f7de1aa25efceb1e669e10b129102', 'v1/attention/backends/mla/b12x_indexer.py': '7d1270dd8e28f077635702719a4aadcad5551b517a4d9a8613ee53ae0c450d6e', 'v1/attention/backends/mla/gb10_indexer_prefill.py': '35ed77f83836e42bfe5174bd4387b78b393457a544da7647ef02c69cc2a916b0'}


def replace(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f'v18 anchor mismatch: {old[:90]!r}')
    return source.replace(old, new, 1)


def common(source):
    return replace(source,
        '                metadata = build_prefill_chunk_metadata(\n',
        '                metadata = getattr(\n'
        '                    self, "_build_prefill_chunk_metadata", build_prefill_chunk_metadata\n'
        '                )(\n')


def indexer(source):
    source = replace(source, 'from vllm.v1.kv_cache_interface import KVCacheSpec\n',
        'from vllm.v1.kv_cache_interface import KVCacheSpec\n'
        'from vllm.v1.attention.backends.mla.gb10_indexer_prefill import build_paged_chunk\n')
    source = replace(source, '        self.use_flattening = False\n',
        '        sm121 = torch.cuda.get_device_capability(self.device) == (12, 1)\n'
        '        self._v18_compact = sm121 and os.getenv("VLLM_GB10_INDEXER_METADATA", "0") == "1"\n'
        '        self._v18_coalesce = (sm121\n'
        '            and os.getenv("VLLM_GB10_INDEXER_COALESCE", "0") == "1"\n'
        '            and int(os.getenv("B12X_PAGED_INDEX_SUPERTILE_K", "32768")) == 32768)\n'
        '        self.use_flattening = False\n')
    source = replace(source, '    def _supports_native_decode(self, next_n: int) -> bool:\n',
        '    def _build_prefill_chunk_metadata(self, *args, **kwargs):\n'
        '        if self._v18_compact:\n'
        '            return build_paged_chunk(*args, **kwargs)\n'
        '        from vllm.v1.attention.backends.mla.indexer import build_prefill_chunk_metadata\n'
        '        return build_prefill_chunk_metadata(*args, **kwargs)\n\n'
        '    def _supports_native_decode(self, next_n: int) -> bool:\n')
    source = replace(source, '    ) -> list[tuple[slice, slice]]:\n        return [\n',
        '    ) -> list[tuple[slice, slice]]:\n'
        '        if self._v18_coalesce:\n'
        '            # B12X streams K in 32K supertiles. Full-context M*N\n'
        '            # limits unnecessarily split the same query rows again.\n'
        '            # Clamp only the sizing input, never device causal lengths.\n'
        '            compressed_seq_lens_cpu = compressed_seq_lens_cpu.clamp(max=32768)\n'
        '        return [\n')
    source = replace(source,
        '                seq_lens = (chunk.cu_seqlen_ke - chunk.cu_seqlen_ks).contiguous()\n',
        '                seq_lens = getattr(chunk, "b12x_seq_lens", None)\n'
        '                if seq_lens is None:\n'
        '                    seq_lens = (chunk.cu_seqlen_ke - chunk.cu_seqlen_ks).contiguous()\n')
    return source


def patch(root, check=False):
    root = Path(root)
    pending = []
    for name, transform in ((COMMON, common), (INDEXER, indexer)):
        path = root/name
        source = path.read_text(encoding='utf-8')
        digest = hashlib.sha256(source.encode()).hexdigest()
        if digest == OUTPUTS[name]:
            continue
        if check or digest != INPUTS[name]:
            raise RuntimeError(f'unexpected v18 source: {path} ({digest})')
        result = transform(source)
        if hashlib.sha256(result.encode()).hexdigest() != OUTPUTS[name]:
            raise RuntimeError(f'v18 transformed hash mismatch: {name}')
        compile(result, str(path), 'exec')
        pending.append((path, result))
    helper = Path(__file__).with_name('gb10_indexer_prefill.py').read_text(encoding='utf-8')
    if hashlib.sha256(helper.encode()).hexdigest() != OUTPUTS[HELPER]:
        raise RuntimeError('v18 helper hash mismatch')
    path = root/HELPER
    if path.exists():
        if path.read_text(encoding='utf-8') != helper:
            raise RuntimeError(f'v18 helper mismatch: {path}')
    elif check:
        raise RuntimeError(f'v18 helper missing: {path}')
    else:
        compile(helper, str(path), 'exec')
        pending.append((path, helper))
    for path, source in pending:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding='utf-8', newline='\n')
