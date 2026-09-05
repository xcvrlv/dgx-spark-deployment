#!/usr/bin/env python3
"""Late Python-only performance overlay for the exact R22/R7 v9 composition.

Apply to a vllm package directory (source or installed). Hashes cover both
states; all files are checked and compiled before any file is written.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


VERSION = "glm53-r22-performance-v1"
# Generated from the pinned R22 source after patch_r22_exl3.py, not master.
INPUT_HASHES = {
    "v1/attention/backends/mla/b12x_mla_sparse.py": "cc8ba39726524d5b1e580a62f9a66caf369be59b04d628bd30c4e1d2a0f73a25",
    "models/deepseek_v32/attention.py": "27072b9aec68eb42f4a3ad80255e8aeb68efb97c23928de5210cde785f31cd59",
    "distributed/device_communicators/cuda_communicator.py": "32c2b0edb40abb61b61ba43e1276bc84cd58064a595663ed202d8aaf4a3c798d",
    "envs.py": "d1e4b4f922da6cba03fb774771f2daaddb9042b816bdd22a6921aa8729bab89c",
}
OUTPUT_HASHES = {
    "v1/attention/backends/mla/b12x_mla_sparse.py": "f8aced19c4c7c1f8ea69dd597595d5961b2ee4b3dc9b642dcca8b4d8d2c14a1a",
    "models/deepseek_v32/attention.py": "d7404824ea5c3fae1af6849db019c71967647c378f76fa03bf7d425932c34c19",
    "distributed/device_communicators/cuda_communicator.py": "fe4216600c66b90d85c51d20eef061c23907d6cb99e310ae1c710a36fe34cc00",
    "envs.py": "1c10ec3181dd9b452660d8eb7a5dcbda8f9050434161c61dcae810d93f06cff8",
}


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one source anchor, found {count}: {old[:100]!r}")
    return text.replace(old, new, 1)


def patch_attention_backend(text: str) -> str:
    text = replace_once(text, "    is_glm_next: bool,\n", "    is_glm_next: bool,\n    is_glm_dsa: bool = False,\n")
    text = replace_once(text, "        and is_glm_next\n", "        and (is_glm_next or is_glm_dsa)\n")
    text = replace_once(
        text,
        "        self._ckv_gather_requested = (\n"
        "            self.requires_glm_next_selector_metadata\n"
        "            and self.dcp_world_size > 1\n",
        "        self._ckv_glm_dsa = _is_glm_dsa_config(\n"
        "            vllm_config.model_config.hf_text_config\n"
        "        ) and self.kv_cache_spec.cache_dtype_str == \"fp8_ds_mla\"\n"
        "        self._ckv_gather_requested = (\n"
        "            (self.requires_glm_next_selector_metadata or self._ckv_glm_dsa)\n"
        "            and not self.use_pcp\n"
        "            and self.dcp_world_size > 1\n",
    )
    text = replace_once(
        text,
        "            ckv_topk_tokens = int(hf_config.index_topk) + int(hf_config.index_kpool) - 1\n",
        "            # GLM-DSA has unpooled top-k indices; GLM5Next adds its tail.\n"
        "            ckv_topk_tokens = int(hf_config.index_topk)\n"
        "            if self.requires_glm_next_selector_metadata:\n"
        "                ckv_topk_tokens += int(hf_config.index_kpool) - 1\n",
    )
    text = replace_once(
        text,
        "            is_glm_next=self.requires_glm_next_selector_metadata,\n",
        "            is_glm_next=self.requires_glm_next_selector_metadata,\n"
        "            is_glm_dsa=self._ckv_glm_dsa,\n",
    )
    text = replace_once(
        text,
        "        self._ckv_gather_enabled = (\n"
        "            self._is_glm_next\n"
        "            and self.dcp_world_size > 1\n",
        "        self._ckv_gather_enabled = (\n"
        "            (self._is_glm_next or (self._is_glm_dsa and not self._uses_nvfp4_cache))\n"
        "            and self.pcp_world_size == 1\n"
        "            and self.dcp_world_size > 1\n",
    )
    text = replace_once(
        text,
        "        self._reserve_planned_workspaces()\n",
        "        self._reserve_planned_workspaces()\n"
        "        # DSA uses fixed 64-token pages: reserve CKV before KV profiling.\n"
        "        # GLM5Next reserves after its hybrid page geometry is finalized.\n"
        "        if self._is_glm_dsa and self._ckv_gather_enabled:\n"
        "            self._reserve_attention_workspaces()\n",
    )
    text = replace_once(
        text,
        '            logger.info_once("Using full-CKV gather for GLM5Next B12X DCP prefill")\n',
        '            logger.info_once("Using full-CKV gather for %s B12X DCP prefill",\n'
        '                             "GLM-DSA" if self._is_glm_dsa else "GLM5Next")\n',
    )
    # The existing gather/mapping kernels already support native byte records,
    # unequal rank lengths, prefix-cache pages and causal chunked prefill.
    text = text.replace("GLM5Next CKV gather requires native ", "B12X CKV gather requires native ")
    return text


def patch_nvidia_attention(text: str) -> str:
    text = replace_once(
        text,
        "        if self.use_pcp and self.impl.dcp_world_size > self.impl.pcp_world_size:\n",
        "        # A complete gathered cache requires only this rank's query heads.\n"
        "        full_ckv_dcp = (\n"
        "            not self.use_pcp\n"
        "            and self.impl.uses_full_ckv_dcp(attn_metadata, num_actual)\n"
        "        )\n"
        "        if self.use_pcp and self.impl.dcp_world_size > self.impl.pcp_world_size:\n",
    )
    text = replace_once(
        text,
        "        elif not self.use_pcp and self.impl.dcp_world_size > 1:\n",
        "        elif not self.use_pcp and self.impl.dcp_world_size > 1 and not full_ckv_dcp:\n",
    )
    text = replace_once(
        text,
        "        if self.impl.dcp_world_size > 1:\n",
        "        if self.impl.dcp_world_size > 1 and not full_ckv_dcp:\n",
    )
    return text


def patch_communicator(text: str) -> str:
    text = replace_once(
        text,
        "        self.use_custom_allreduce = use_custom_allreduce\n",
        "        # RoCEnante is valid across hosts, including attention's DCP group.\n"
        "        # Keep all single-host custom transports restricted to TP.\n"
        "        if unique_name.split(\":\", 1)[0] == \"dcp\":\n"
        "            from vllm.distributed.parallel_state import _ENABLE_CUSTOM_ALL_REDUCE\n"
        "\n"
        "            use_roce_allreduce = (\n"
        "                _ENABLE_CUSTOM_ALL_REDUCE\n"
        "                and envs.VLLM_ENABLE_ROCE_ALLREDUCE\n"
        "                and envs.VLLM_ROCE_DCP_ENABLE\n"
        "            )\n"
        "\n"
        "        self.use_custom_allreduce = use_custom_allreduce\n",
    )
    text = replace_once(
        text,
        "            return b12x_ar_comm.all_gather(input_, dim)\n",
        "            return b12x_ar_comm.all_gather(input_, dim)\n"
        "        if (\n"
        "            self.use_roce_allreduce\n"
        "            and b12x_ar_comm is not None\n"
        "            and not b12x_ar_comm.disabled\n"
        "            and 0 < dim < input_.dim() - 1\n"
        "            and input_.is_contiguous()\n"
        "        ):\n"
        "            # [tokens, heads, width] -> [tokens, heads*width] is a view.\n"
        "            # Concatenating the final dimension preserves head order and\n"
        "            # lets RoCEnante write the final query layout without a copy.\n"
        "            flat = input_.flatten(start_dim=dim)\n"
        "            if b12x_ar_comm.should_all_gather(flat, dim):\n"
        "                shape = list(input_.shape)\n"
        "                shape[dim] *= self.world_size\n"
        "                return b12x_ar_comm.all_gather(flat, dim).view(shape)\n",
    )
    text = replace_once(
        text,
        "    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1):\n"
        "        world_size = self.world_size\n",
        "    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1):\n"
        "        world_size = self.world_size\n"
        "        # Small DCP output: one-shot all-reduce plus a local head slice\n"
        "        # avoids NCCL's pack/reduce-scatter/unpack sequence. Larger\n"
        "        # prefill tensors retain the bandwidth-efficient NCCL path.\n"
        "        normalized_dim = dim % input_.dim()\n"
        "        roce = self.b12x_ar_comm\n"
        "        if (\n"
        "            self.unique_name.split(\":\", 1)[0] == \"dcp\"\n"
        "            and self.use_roce_allreduce\n"
        "            and not envs.VLLM_BATCH_INVARIANT\n"
        "            and roce is not None\n"
        "            and not roce.disabled\n"
        "            and input_.shape[normalized_dim] % world_size == 0\n"
        "            and 0 < input_.numel() * input_.element_size()\n"
        "            <= envs.VLLM_ROCE_DCP_RS_MAX_BYTES\n"
        "            and roce.should_custom_ar(input_)\n"
        "        ):\n"
        "            reduced = roce.custom_all_reduce(input_)\n"
        "            assert reduced is not None\n"
        "            chunk = input_.shape[normalized_dim] // world_size\n"
        "            return reduced.narrow(\n"
        "                normalized_dim, self.rank_in_group * chunk, chunk\n"
        "            ).contiguous()\n",
    )
    return text


def patch_envs(text: str) -> str:
    text = replace_once(
        text,
        "    VLLM_DCP_Q_REPLICATE: bool = False\n",
        "    VLLM_ROCE_DCP_ENABLE: bool = False\n"
        "    VLLM_ROCE_DCP_RS_MAX_BYTES: int = 0\n"
        "    VLLM_DCP_Q_REPLICATE: bool = False\n",
    )
    text = replace_once(
        text,
        '    "VLLM_DCP_Q_REPLICATE": lambda: bool(int(os.getenv("VLLM_DCP_Q_REPLICATE", "0"))),\n',
        '    "VLLM_ROCE_DCP_ENABLE": lambda: bool(int(os.getenv("VLLM_ROCE_DCP_ENABLE", "0"))),\n'
        '    "VLLM_ROCE_DCP_RS_MAX_BYTES": lambda: int(os.getenv("VLLM_ROCE_DCP_RS_MAX_BYTES", "0")),\n'
        '    "VLLM_DCP_Q_REPLICATE": lambda: bool(int(os.getenv("VLLM_DCP_Q_REPLICATE", "0"))),\n',
    )
    return text


TRANSFORMS = {
    "v1/attention/backends/mla/b12x_mla_sparse.py": patch_attention_backend,
    "models/deepseek_v32/attention.py": patch_nvidia_attention,
    "distributed/device_communicators/cuda_communicator.py": patch_communicator,
    "envs.py": patch_envs,
}


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def patch(root: Path, *, check: bool = False) -> None:
    pending = {}
    for relative, transform in TRANSFORMS.items():
        path = root / relative
        source = path.read_text(encoding="utf-8")
        observed = digest(source)
        if observed == OUTPUT_HASHES[relative]:
            continue
        if check or observed != INPUT_HASHES[relative]:
            raise RuntimeError(f"{relative}: unexpected source SHA256 {observed}")
        output = transform(source)
        if digest(output) != OUTPUT_HASHES[relative]:
            raise RuntimeError(f"{relative}: unexpected performance overlay output")
        compile(output, str(path), "exec")
        pending[path] = output
    for path, output in pending.items():
        path.write_text(output, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    patch(args.package, check=args.check)
    print(f"{VERSION}: {'verified' if args.check else 'applied'} {args.package}")


if __name__ == "__main__":
    main()
