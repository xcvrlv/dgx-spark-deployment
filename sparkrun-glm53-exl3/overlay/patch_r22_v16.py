#!/usr/bin/env python3
"""Strict v16 DCP and memory overlay over the v15 composition."""
import hashlib
from pathlib import Path

VERSION = "glm53-r22-v16-1"
ATTENTION = "v1/attention/backends/mla/b12x_mla_sparse.py"
INDEXER = "v1/attention/backends/mla/b12x_indexer.py"
HELPER = "distributed/device_communicators/gb10_dcp.py"
RUNTIME = "comm/roce/roce_oneshot.py"
PROXY = "comm/roce/_roce_proxy.c"
INPUTS = {'v1/attention/backends/mla/b12x_mla_sparse.py': '05d68843ce90f972a26075a221f869fcd1084374cb14a5655b06a856c3df5f15', 'v1/attention/backends/mla/b12x_indexer.py': '269506b11518f91600bf58cede43fd61a53725ab3d06aa7d3f884791230ee079', 'comm/roce/roce_oneshot.py': '92a32ed7ad570f54228998da10dc8dd463a042458c2bbd6beec521d1668344d4', 'comm/roce/_roce_proxy.c': '10a74fe714069def0f2356addd7a2c3178d7e82e3a678954de7448a5a249c6e9'}
OUTPUTS = {'v1/attention/backends/mla/b12x_mla_sparse.py': '90099538aac0055c187504f2f7df9eff65facdc44e35178424562e518bf3d266', 'v1/attention/backends/mla/b12x_indexer.py': '6920327ca8ac67bd746ed96bdb71671ab84fd3991d9d4f274a76a7c8944c88c2', 'comm/roce/roce_oneshot.py': '1eb29a73a3fc0d65b5598cc0520691f9e126c706183f88bc525aa48e2cd74b98', 'comm/roce/_roce_proxy.c': 'b41e32ee56fb980bf8a5c75e855219665452a029b817643e4ace3bdf1717a6c1', 'distributed/device_communicators/gb10_dcp.py': '15e443a0119e234b3fa1bbc8a2ea8cef6af0212c4f4701293f489323fafad5a7'}


def replace(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"v16 source anchor mismatch: {old[:100]!r}")
    return text.replace(old, new, 1)


def attention(text):
    text = replace(text, "import numpy as np\n", "import numpy as np\nfrom vllm.distributed.device_communicators.gb10_dcp import try_roce_gather_into\n")
    text = replace(text, '    communicator = getattr(group, "device_communicator", None)\n',
                   '    if try_roce_gather_into(group, input_tensor, output_tensor):\n        return\n\n'
                   '    communicator = getattr(group, "device_communicator", None)\n')
    text = replace(text, "        self._ckv_gather_enabled = (\n",
                   '        self._v16_ckv_inplace = os.getenv("VLLM_GB10_CKV_INPLACE", "0") == "1"\n'
                   "        self._ckv_gather_enabled = (\n")
    text = replace(text, "                (self._ckv_local_capacity, self._cache_record_bytes),\n",
                   "                (0 if self._v16_ckv_inplace else self._ckv_local_capacity,\n"
                   "                 self._cache_record_bytes),\n")
    text = replace(text, "        expected_local_shape = (\n            self._ckv_local_capacity,\n",
                   "        expected_local_shape = (\n            0 if self._v16_ckv_inplace else self._ckv_local_capacity,\n")
    text = replace(text, "        padded_tokens = attn_metadata.dcp_padded_total_tokens\n",
                   "        padded_tokens = attn_metadata.dcp_padded_total_tokens\n"
                   "        if self._v16_ckv_inplace:\n"
                   "            # NCCL's in-place all-gather offset is rank * active shard\n"
                   "            # size, not rank * workspace capacity. In RoCE dim=0,\n"
                   "            # self copies map to the same input pack; peer output\n"
                   "            # ranges are disjoint from the input shard.\n"
                   "            rank = get_dcp_group().rank_in_group\n"
                   "            local_buffer = gathered_buffer[rank * padded_tokens:(rank + 1) * padded_tokens]\n")
    return text


def indexer(text):
    text = replace(text, "import bisect\n", "import bisect\nfrom vllm.distributed.device_communicators.gb10_dcp import try_roce_gather_into\n")
    text = replace(text, "_INDEX_HEAD_DIM = 128\n",
                   '_V16_MERGE_ROWS = int(os.getenv("VLLM_GB10_DCP_MERGE_ROWS", "0"))\n'
                   'if not 0 <= _V16_MERGE_ROWS <= 1024:\n'
                   '    raise ValueError("VLLM_GB10_DCP_MERGE_ROWS must be 0..1024 (0 disables)")\n\n'
                   'def _v16_merge_specs(rows, topk, world):\n'
                   '    return (((rows, topk, 2), torch.float32),\n'
                   '            ((rows, world * topk, 2), torch.float32))\n\n\n'
                   "_INDEX_HEAD_DIM = 128\n")
    begin = text.index("    packed = torch.empty(\n", text.index("def _merge_dcp_topk("))
    end = text.index("\n\n\nclass B12xSparseIndexer", begin)
    legacy = text[begin:end]
    # Row independence preserves the original stable top-k and exact scores.
    body = legacy[legacy.index("    _pack_dcp_candidates_kernel["):]
    body = body.replace("indices.shape[0]", "chunk_indices.shape[0]")
    body = body.replace("        indices,\n        scores,", "        chunk_indices,\n        chunk_scores,")
    body = body.replace("indices.stride(0)", "chunk_indices.stride(0)").replace("scores.stride(0)", "chunk_scores.stride(0)")
    body = body.replace("    gathered = get_dcp_group().all_gather(packed, dim=1)\n",
                        "    group = get_dcp_group()\n"
                        "    if not try_roce_gather_into(group, packed.flatten(1), gathered.flatten(1), 1):\n"
                        "        gathered.copy_(group.all_gather(packed, dim=1))\n")
    body = body.replace("out=indices)", "out=chunk_indices)")
    optimized = (
        "    if _V16_MERGE_ROWS:\n"
        "        rows = min(int(indices.shape[0]), _V16_MERGE_ROWS)\n"
        "        # The paged-indexer scratch is dead here. Scores and output\n"
        "        # indices are independent tensors; the reducer borrows no arena.\n"
        "        packed_buffer, gathered_buffer = current_workspace_manager().get_simultaneous(\n"
        "            *_v16_merge_specs(rows, topk, dcp_world_size))\n"
        "        for start in range(0, int(indices.shape[0]), rows):\n"
        "            end = min(start + rows, int(indices.shape[0]))\n"
        "            chunk_indices, chunk_scores = indices[start:end], scores[start:end]\n"
        "            packed, gathered = packed_buffer[:end-start], gathered_buffer[:end-start]\n"
        + "".join("        " + line if line.strip() else line for line in body.splitlines(keepends=True))
        + "\n        return\n"
    )
    text = text[:begin] + optimized + legacy + text[end:]
    anchor = "        self.dcp_rank = get_dcp_group().rank_in_group if self.dcp_world_size > 1 else 0\n"
    text = replace(text, anchor, anchor +
                   "        if self.dcp_world_size > 1 and _V16_MERGE_ROWS:\n"
                   "            # Reserve in every target/draft and ubatch lane before\n"
                   "            # profiling; graph replay never grows this workspace.\n"
                   "            current_workspace_manager().reserve_all(*_v16_merge_specs(\n"
                   "                min(max_q_rows, _V16_MERGE_ROWS), self.topk_tokens, self.dcp_world_size))\n")
    return text


def runtime(text):
    old = "            self._region = torch.zeros(\n                self._layout.total_bytes, dtype=torch.uint8, pin_memory=True\n            )\n"
    text = replace(text, old,
                   '            if os.getenv("B12X_ROCE_LAZY_PAYLOAD_INIT", "0") == "1":\n'
                   '                self._region = torch.empty(\n'
                   '                    self._layout.total_bytes, dtype=torch.uint8, pin_memory=True)\n'
                   '                # Every active payload byte is staged/written before\n'
                   '                # consumption. Only protocol state needs initial zeroes.\n'
                   '                self._region[self._layout.flag_off:self._layout.send_off].zero_()\n'
                   '                self._region[self._layout.ctrl_off:].zero_()\n'
                   '            else:\n' + ''.join('    '+line for line in old.splitlines(keepends=True)))
    return text


def proxy(text):
    text = replace(text, "    uint32_t outstanding[ROCE_MAX_PEERS];\n",
                   "    uint32_t outstanding[ROCE_MAX_PEERS];\n    uint32_t pending_completions;\n")
    text = replace(text, "    int balanced_fanout;\n",
                   "    int balanced_fanout;\n    int skip_empty_cq;\n")
    anchor = '    c->balanced_fanout = fanout_env && strcmp(fanout_env, "1") == 0;\n'
    text = replace(text, anchor, anchor +
                   '    c->skip_empty_cq = getenv("B12X_ROCE_SKIP_EMPTY_CQ") && strcmp(getenv("B12X_ROCE_SKIP_EMPTY_CQ"), "1") == 0;\n')
    text = replace(text, "static int drain_cq(roce_ctx_t *c, int h) {\n",
                   "static int drain_cq(roce_ctx_t *c, int h) {\n"
                   "    // Only this proxy thread posts and drains the send CQ. One\n"
                   "    // signaled completion follows each payload/flag chain.\n"
                   "    if (c->skip_empty_cq && c->hca[h].pending_completions == 0) return 0;\n")
    text = replace(text, "        c->hca[h].outstanding[wc[i].wr_id] -= 1;\n",
                   "        c->hca[h].outstanding[wc[i].wr_id] -= 1;\n        c->hca[h].pending_completions -= 1;\n")
    text = replace(text, "            hca->outstanding[p] += 1;\n",
                   "            hca->outstanding[p] += 1;\n            hca->pending_completions += 1;\n")
    return text


def patch(b12x_root, vllm_root, check=False):
    jobs = [(Path(vllm_root), ATTENTION, attention), (Path(vllm_root), INDEXER, indexer),
            (Path(b12x_root), RUNTIME, runtime), (Path(b12x_root), PROXY, proxy)]
    pending = []
    for root, name, transform in jobs:
        path = root / name
        source = path.read_text(encoding="utf-8")
        digest = hashlib.sha256(source.encode()).hexdigest()
        if digest == OUTPUTS[name]:
            continue
        if check or digest != INPUTS[name]:
            raise RuntimeError(f"unexpected v16 source: {path} ({digest})")
        result = transform(source)
        if hashlib.sha256(result.encode()).hexdigest() != OUTPUTS[name]:
            raise RuntimeError(f"v16 transformed hash mismatch: {name}")
        if name.endswith('.py'):
            compile(result, str(path), 'exec')
        pending.append((path, result))
    helper = Path(__file__).with_name('gb10_dcp.py').read_text(encoding='utf-8')
    if hashlib.sha256(helper.encode()).hexdigest() != OUTPUTS[HELPER]:
        raise RuntimeError('v16 helper hash mismatch')
    path = Path(vllm_root) / HELPER
    if path.exists():
        if path.read_text(encoding='utf-8') != helper:
            raise RuntimeError(f'v16 helper mismatch: {path}')
    elif check:
        raise RuntimeError(f'v16 helper missing: {path}')
    else:
        compile(helper, str(path), 'exec')
        pending.append((path, helper))
    for path, source in pending:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding='utf-8', newline='\n')
