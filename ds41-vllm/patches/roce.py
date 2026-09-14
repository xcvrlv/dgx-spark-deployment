#!/usr/bin/env python3
"""Audited RoCEnante-only port of GLM v15/v16; compute overlays excluded.
Rebased on b12x 3a8b879, checked 2026-09-14. Prepared-plan execution
changes are retained; the local port only changes transport initialization/proxy. All four switches default off
in code and are selected by the c16 communication recipe.
"""
import argparse
import hashlib
from pathlib import Path

VERSION = 'ds41-roce-v1'
INPUTS = {'comm/roce/_roce_proxy.c': 'a35f54cf75d6abf2427c5d144300647685816060348883d2cbf51d153efc5c70', 'comm/roce/roce_oneshot.py': 'c8a71b9a06eb4833d5eb64b1e11ad322699034c0b9b2fe45579c24d76f1b3ee3'}
OUTPUTS = {'comm/roce/_roce_proxy.c': 'b41e32ee56fb980bf8a5c75e855219665452a029b817643e4ace3bdf1717a6c1', 'comm/roce/roce_oneshot.py': '65cc77211f3ad3f82a65dc29184fd27745ac3f35c9b717ab76240d7e4399083c'}

def replace(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"v15 source anchor mismatch: {old[:100]!r}")
    return text.replace(old, new, 1)


def proxy_v15(text):
    text = replace(text, "    uint32_t outstanding[ROCE_MAX_PEERS];\n",
                   "    uint32_t outstanding[ROCE_MAX_PEERS];\n    uint32_t inline_bytes[ROCE_MAX_PEERS];\n")
    text = replace(text, "    int gid_index;\n", "    int gid_index;\n    int inline_payload;\n    int balanced_fanout;\n")
    text = replace(text, "    c->gid_index = gid_index;\n",
                   "    c->gid_index = gid_index;\n"
                   "    const char *inline_env = getenv(\"B12X_ROCE_INLINE_PAYLOAD\");\n"
                   "    const char *fanout_env = getenv(\"B12X_ROCE_BALANCED_FANOUT\");\n"
                   "    c->inline_payload = inline_env && strcmp(inline_env, \"1\") == 0;\n"
                   "    c->balanced_fanout = fanout_env && strcmp(fanout_env, \"1\") == 0;\n")
    text = replace(text, "        attr.cap.max_inline_data = 16;\n        hca->qp[p] = ibv_create_qp(hca->pd, &attr);\n",
                   "        attr.cap.max_inline_data = c->inline_payload ? 64 : 16;\n"
                   "        hca->qp[p] = ibv_create_qp(hca->pd, &attr);\n"
                   "        if (hca->qp[p] == NULL && c->inline_payload) {\n"
                   "            attr.cap.max_inline_data = 16;\n"
                   "            hca->qp[p] = ibv_create_qp(hca->pd, &attr);\n"
                   "        }\n"
                   "        hca->inline_bytes[p] = attr.cap.max_inline_data;\n")
    start = text.index("static int post_op(")
    end = text.index("static void *proxy_main", start)
    part = text[start:end]
    part = replace(part, "    for (int p = 0; p < c->world; p++) {\n",
                   "    // All posts are asynchronous. Rotate first destination by rank\n"
                   "    // so a switched full mesh does not first converge on rank zero.\n"
                   "    for (int turn = 0; turn < c->world; turn++) {\n"
                   "        int p = c->balanced_fanout ? (c->rank + turn + 1) % c->world : turn;\n")
    part = replace(part, "                data_wr.opcode = IBV_WR_RDMA_WRITE;\n",
                   "                data_wr.opcode = IBV_WR_RDMA_WRITE;\n"
                   "                // Inline copies the committed GPU-staged bytes into the\n"
                   "                // WQE, avoiding a NIC DMA fetch for small MTP packets.\n"
                   "                if (c->inline_payload && stripe_bytes <= hca->inline_bytes[p]) {\n"
                   "                    data_wr.send_flags = IBV_SEND_INLINE;\n"
                   "                }\n")
    return text[:start] + part + text[end:]


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


def proxy_v16(text):
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


def patch(root, check=False):
    jobs = [('comm/roce/_roce_proxy.c', lambda s: proxy_v16(proxy_v15(s))),
            ('comm/roce/roce_oneshot.py', runtime)]
    pending = []
    for name, transform in jobs:
        path = Path(root) / name
        source = path.read_text(encoding='utf-8')
        digest = hashlib.sha256(source.encode()).hexdigest()
        if digest == OUTPUTS[name]:
            continue
        if check or digest != INPUTS[name]:
            raise RuntimeError(f'Unexpected upstream source: {name}: {digest}; re-audit before patching')
        output = transform(source)
        assert hashlib.sha256(output.encode()).hexdigest() == OUTPUTS[name]
        if name.endswith('.py'):
            compile(output, name, 'exec')
        pending.append((path, output))
    for path, output in pending:
        path.write_text(output, encoding='utf-8', newline='\n')
    print(VERSION, 'verified' if check else 'applied/verified')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    patch(args.root, args.check)
