#!/usr/bin/env python3
"""Pinned v14 overlay: shared-H prefill and SM121 draft argmax."""
import hashlib
from pathlib import Path

VERSION = "glm53-r22-v14-1"
KERNEL = "moe/_shared/kernels/w4a16/kernel.py"
LOGITS = "model_executor/layers/logits_processor.py"
HELPER = "model_executor/layers/gb10_argmax.py"
INPUTS = {'moe/_shared/kernels/w4a16/kernel.py': '4da593709d5d18b10e73773b5b2cd28bee33bbf33b92d0bb2fc90844d7344894', 'model_executor/layers/logits_processor.py': '36c5a32a55ab0fbb061d9632962ccfb33322c5a24f3fb2b237bb05740f0b3ad1'}
OUTPUTS = {'moe/_shared/kernels/w4a16/kernel.py': '0880884d719b1200a4cb20a985138625ee505ffab58422fe4c1735a94b9042a5', 'model_executor/layers/logits_processor.py': '1c17732210f761a9698791f01856ec0633100e59c09e83306d0d4c6ebede88dd', 'model_executor/layers/gb10_argmax.py': '2b8d3c27bb37af047a4ce3f7425312e23ce73dd4765ffd0c9bcea9fb878425bb'}


def replace(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"v14 source anchor mismatch: {old[:100]!r}")
    return text.replace(old, new, 1)


def kernel(text):
    text = replace(text, "        self.broadcast_suh = bool(broadcast_suh)\n",
                   "        self.broadcast_suh = bool(broadcast_suh)\n"
                   "        self.gb10_shared_input = (self.broadcast_suh and int(size_m) > 64\n"
                   "            and torch.cuda.get_device_capability() == (12, 1)\n"
                   "            and os.getenv('VLLM_GB10_SHARED_INPUT_ROTATION', '0') == '1')\n")
    text = replace(text, "            self.broadcast_suh,\n            self.rotation_input_dtype,\n",
                   "            self.broadcast_suh,\n            ('v14_shared_input', self.gb10_shared_input),\n"
                   "            self.rotation_input_dtype,\n")
    start = text.index("    @cute.jit\n    def _run_input_rotation(\n")
    end = text.index("    @cute.jit\n    def _run_input_rotation_coupled(\n", start)
    original = text[start:end]
    shared = original.replace("def _run_input_rotation(", "def _run_input_rotation_shared(", 1)
    begin = shared.index("        # One warp owns")
    finish = shared.index("        lane =", begin)
    shared = shared[:begin] + "        # Shared scales: compute each token/H128 once, then fan out to routes.\n" + shared[finish:]
    shared = replace(shared, "        total_units = route_count * nblk\n", "        total_units = active_m * nblk\n")
    begin = shared.index("            route_pos = unit // nblk\n")
    finish = shared.index("                col0 =", begin)
    shared = shared[:begin] + (
        "            token = unit // nblk\n"
        "            blk = unit - token * nblk\n"
        "            if token < active_m:\n") + shared[finish:]
    begin = shared.index("                if cutlass.const_expr(self.broadcast_suh):\n")
    finish = shared.index("\n                x0 =", begin)
    shared = shared[:begin] + "                s_base = col0\n" + shared[finish:]
    begin = shared.index("                a_gate_flat[out_base")
    finish = shared.index("            unit += gw_stride", begin)
    stores = shared[begin:finish]
    shared = shared[:begin] + (
        "                for slot in cutlass.range_constexpr(self.top_k):\n"
        "                    out_base = (token * Int32(self.top_k) + Int32(slot)) * Int32(self.hidden_size) + col0\n"
        + "".join("    " + line for line in stores.splitlines(keepends=True))) + shared[finish:]
    # Drop route metadata loads: this branch never needs expert IDs or counts.
    shared = replace(shared, "        live_routes = active_m * Int32(self.top_k)\n"
        "        route_count = packed_route_count[Int32(0)].to(Int32)\n"
        "        if cutlass.const_expr(self.direct_topk_routes):\n"
        "            route_count = live_routes\n", "")
    begin = original.index("        # One warp owns")
    body = original[begin:]
    call = ("        if cutlass.const_expr(self.gb10_shared_input):\n"
            "            self._run_input_rotation_shared(\n"
            "                x_input_flat, a_gate_flat, a_up_flat, suh_gate_flat, suh_up_flat,\n"
            "                packed_route_indices, block_expert_ids, packed_route_count, expert_map_flat,\n"
            "                weight_num_experts, route_num_experts, tid, cta, grid_x, active_m)\n"
            "        else:\n")
    wrapped = original[:begin] + call + "".join("    " + line if line.strip() else line for line in body.splitlines(keepends=True))
    return text[:start] + shared + wrapped + text[end:]


def logits(text):
    start = text.index("    def get_top_tokens(\n")
    end = text.index("    def get_top_k_tokens(\n", start)
    anchor = "        # Mask out padding entries beyond org_vocab_size on this shard.\n"
    part = replace(text[start:end], anchor,
        "        from . import gb10_argmax\n"
        "        shard = lm_head.shard_indices\n"
        "        valid = logits.shape[-1] - shard.num_org_vocab_padding\n"
        "        start = shard.org_vocab_start_index\n"
        "        # Only the original, contiguous vocabulary shard; added-vocab/LoRA stays stock.\n"
        "        if (gb10_argmax.enabled(logits)\n"
        "                and valid == shard.org_vocab_end_index - start\n"
        "                and start + logits.shape[-1] < 2**24):\n"
        "            pair = gb10_argmax.local_pair(logits, valid, start)\n"
        "            if tp_size > 1:\n"
        "                pair = tensor_model_parallel_all_gather(pair, dim=-1)\n"
        "            return gb10_argmax.global_tokens(pair, tp_size)\n\n" + anchor)
    part = replace(part,
        "            [local_max_vals.float(), global_indices.float()], dim=-1\n",
        "            [local_max_vals.float(), global_indices.float(),\n"
        "             torch.zeros_like(local_max_vals, dtype=torch.float32),\n"
        "             torch.zeros_like(local_max_vals, dtype=torch.float32)], dim=-1\n")
    part = replace(part, "gathered.view(hidden_states.shape[0], tp_size, 2)",
                   "gathered.view(hidden_states.shape[0], tp_size, 4)")
    part = part.replace("# [batch, 2] -> [batch, 2 * tp_size]",
                        "# v14: [batch, 4] packets keep RoCEnante rows 16-byte aligned.")
    part = part.replace("# [batch, tp_size, 2] where", "# [batch, tp_size, 4] where")
    return text[:start] + part + text[end:]


def patch(b12x_root, vllm_root, check=False):
    helper = Path(__file__).with_name("gb10_argmax.py").read_text(encoding="utf-8")
    if hashlib.sha256(helper.encode()).hexdigest() != OUTPUTS[HELPER]:
        raise RuntimeError("v14 helper hash mismatch")
    pending = []
    for root, name, transform in ((b12x_root, KERNEL, kernel), (vllm_root, LOGITS, logits), (vllm_root, HELPER, None)):
        path = Path(root) / name
        source = path.read_text(encoding="utf-8") if path.exists() else None
        digest = hashlib.sha256(source.encode()).hexdigest() if source is not None else None
        if digest == OUTPUTS[name]:
            continue
        if check or digest != INPUTS.get(name):
            raise RuntimeError(f"unexpected v14 source: {path} ({digest})")
        result = transform(source) if transform else helper
        compile(result, str(path), "exec")
        if hashlib.sha256(result.encode()).hexdigest() != OUTPUTS[name]:
            raise RuntimeError(f"v14 output mismatch: {path}")
        pending.append((path, result))
    for path, result in pending:
        path.write_text(result, encoding="utf-8", newline="\n")
    print(f"{VERSION}: verified {b12x_root} and {vllm_root}")
