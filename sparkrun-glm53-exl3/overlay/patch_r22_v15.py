#!/usr/bin/env python3
"""Pinned v15 SM121 compute and switched-fabric optimizations over v14."""
import hashlib
from pathlib import Path

VERSION = "glm53-r22-v15-1"
KERNEL = "moe/_shared/kernels/w4a16/kernel.py"
ACTIVATION = "moe/_shared/kernels/w4a16/gb10_activation.py"
PROXY = "comm/roce/_roce_proxy.c"
SKINNY = "models/deepseek_v32/nvidia/glm52_low_latency_gemm.py"
INPUTS = {'moe/_shared/kernels/w4a16/kernel.py': '0880884d719b1200a4cb20a985138625ee505ffab58422fe4c1735a94b9042a5', 'comm/roce/_roce_proxy.c': 'a35f54cf75d6abf2427c5d144300647685816060348883d2cbf51d153efc5c70', 'models/deepseek_v32/nvidia/glm52_low_latency_gemm.py': '912e0d144520205c2606e62c5e1db64091a4c75b273bac8dccc6b54eaa8a262a'}
OUTPUTS = {'moe/_shared/kernels/w4a16/kernel.py': '2e0e11bdb37e71d8fe5d926d46fd1aaa23a384060e10276b2f7efbef4a87f975', 'comm/roce/_roce_proxy.c': '10a74fe714069def0f2356addd7a2c3178d7e82e3a678954de7448a5a249c6e9', 'models/deepseek_v32/nvidia/glm52_low_latency_gemm.py': 'ec1ebb95f1acc4e6adbb2ef6da696d1b846d768553a48cedeacf218478f2feb6', 'moe/_shared/kernels/w4a16/gb10_activation.py': '52a5c8da91029734b762552bfe578df4547118b95322ece63067aa7f78e857c0'}


def replace(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"v15 source anchor mismatch: {old[:100]!r}")
    return text.replace(old, new, 1)


def kernel(text):
    text = replace(text, "import torch\n", "import torch\nfrom .gb10_activation import reciprocal as gb10_reciprocal\n")
    text = replace(text, "        route_major_a: bool = False,\n",
                   "        route_major_a: bool = False,\n        input_route_divisor: int = 1,\n")
    text = replace(text, "        self.route_major_a = bool(route_major_a)\n",
                   "        self.route_major_a = bool(route_major_a)\n"
                   "        self.input_route_divisor = int(input_route_divisor)\n")
    text = replace(text, "            self.route_major_a,\n",
                   "            self.route_major_a,\n            ('v15_input_divisor', self.input_route_divisor),\n")
    # Three FC1 route readers: ordinary packed, single-token, direct top-k.
    for indent, count in (("                ", 2), ("            ", 1)):
        old = indent + "if cutlass.const_expr(self.route_major_a):\n" + indent + "    rd_row = idx\n"
        if text.count(old) != count:
            raise RuntimeError("v15 FC1 route reader count mismatch")
        text = text.replace(old, old.replace("rd_row = idx", "rd_row = idx // Int32(self.input_route_divisor)"))
    start = text.index("class W4A16FusedMoeKernel:")
    end = text.index("class W4A16ActivationKernel:", start)
    part = text[start:end]
    part = replace(part, "        self.fast_math = bool(fast_math)\n",
                   "        self.fast_math = bool(fast_math)\n"
                   "        self.gb10_sigmoid = (self.fast_math\n"
                   "            and torch.cuda.get_device_capability() == (12, 1)\n"
                   "            and os.getenv('VLLM_GB10_SIGMOID', '0') == '1')\n")
    anchor = "            and os.getenv('VLLM_GB10_SHARED_INPUT_ROTATION', '0') == '1')\n"
    part = replace(part, anchor, anchor +
                   "        self.gb10_compact_input = (self.gb10_shared_input and not bool(coupled_hadamard)\n"
                   "            and os.getenv('VLLM_GB10_COMPACT_INPUT', '0') == '1')\n")
    part = replace(part, "            route_major_a=self.full_rotation,\n",
                   "            route_major_a=self.full_rotation,\n"
                   "            input_route_divisor=(self.top_k if self.gb10_compact_input else 1),\n")
    part = replace(part, "            self.fast_math,\n",
                   "            self.fast_math,\n            ('v15_sigmoid', self.gb10_sigmoid),\n"
                   "            ('v15_compact_input', self.gb10_compact_input),\n")
    begin = part.index("                for slot in cutlass.range_constexpr(self.top_k):\n")
    finish = part.index("            unit += gw_stride", begin)
    old_stores = part[begin:finish]
    # Keep the original route-sized allocation because it aliases FC2 output.
    # Only the first M rows carry FC1 input in the compact specialization.
    body = old_stores.splitlines(keepends=True)[2:]
    compact = "                if cutlass.const_expr(self.gb10_compact_input):\n"
    compact += "                    out_base = token * Int32(self.hidden_size) + col0\n"
    compact += "".join(body)
    compact += "                else:\n" + "".join("    " + line for line in old_stores.splitlines(keepends=True))
    part = part[:begin] + compact + part[finish:]
    part = replace(part, "        return cutlass.Float32(1.0) / (cutlass.Float32(1.0) + e)\n",
                   "        denominator = cutlass.Float32(1.0) + e\n"
                   "        if cutlass.const_expr(self.gb10_sigmoid):\n"
                   "            if denominator >= cutlass.Float32(1.0) and denominator < cutlass.Float32(2.0**126):\n"
                   "                return gb10_reciprocal(denominator)\n"
                   "        return cutlass.Float32(1.0) / denominator\n")
    return text[:start] + part + text[end:]


def proxy(text):
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


def skinny(text):
    text = replace(text, "import torch\n", "import os\nimport torch\n")
    anchor = "def _is_sm103() -> bool:\n    return current_platform.is_device_capability((10, 3))\n"
    addition = '''

def _gb10_enabled() -> bool:
    return (current_platform.is_device_capability((12, 1))
            and os.getenv("VLLM_GB10_SKINNY_GEMM", "0") == "1")


def _device_supported() -> bool:
    return _is_sm103() or _gb10_enabled()


def _projection_spec(shape):
    if not _gb10_enabled():
        return GLM52_PROJECTIONS.get(shape)
    # Separate SM121 SIMT profiles; never select the SM103 dsv3 fused-A op.
    if shape not in GLM52_PROJECTIONS:
        return None
    n, k = shape
    return GLM52ProjectionSpec(n=n, k=k, cute_configs=(
        (1, SkinnyGemmConfig(1, 128, 4, static_k=k)),
        (2, SkinnyGemmConfig(2, 128, 2, static_k=k)),
    ))
'''
    text = replace(text, anchor, anchor + addition)
    if text.count("not _is_sm103():") != 2 or text.count("GLM52_PROJECTIONS.get(tuple(weight.shape))") != 2:
        raise RuntimeError("v15 skinny plan anchors changed")
    text = text.replace("not _is_sm103():", "not _device_supported():")
    text = text.replace("GLM52_PROJECTIONS.get(tuple(weight.shape))", "_projection_spec(tuple(weight.shape))")
    return text


def patch(b12x_root, vllm_root, check=False):
    helper = Path(__file__).with_name("gb10_activation.py").read_text(encoding="utf-8")
    if hashlib.sha256(helper.encode()).hexdigest() != OUTPUTS[ACTIVATION]:
        raise RuntimeError("v15 activation helper hash mismatch")
    pending = []
    for root, name, transform in ((b12x_root, KERNEL, kernel), (b12x_root, PROXY, proxy),
                                  (vllm_root, SKINNY, skinny), (b12x_root, ACTIVATION, None)):
        path = Path(root) / name
        source = path.read_text(encoding="utf-8") if path.exists() else None
        digest = hashlib.sha256(source.encode()).hexdigest() if source is not None else None
        if digest == OUTPUTS[name]:
            continue
        if check or digest != INPUTS.get(name):
            raise RuntimeError(f"unexpected v15 source: {path} ({digest})")
        result = transform(source) if transform else helper
        if name.endswith('.py'):
            compile(result, str(path), 'exec')
        if hashlib.sha256(result.encode()).hexdigest() != OUTPUTS[name]:
            raise RuntimeError(f"v15 output mismatch: {path}")
        pending.append((path, result))
    for path, result in pending:
        path.write_text(result, encoding='utf-8', newline='\n')
    print(f"{VERSION}: verified {b12x_root} and {vllm_root}")
