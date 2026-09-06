#!/usr/bin/env python3
"""Late v12 attention overlay; strict UTF-8 hashes against the v11 image."""
import argparse
import hashlib
from pathlib import Path

VERSION = "glm53-r22-v12-1"
TARGET = "v1/attention/backends/mla/b12x_mla_sparse.py"
INPUT_HASH = "f8aced19c4c7c1f8ea69dd597595d5961b2ee4b3dc9b642dcca8b4d8d2c14a1a"
OUTPUT_HASH = "05d68843ce90f972a26075a221f869fcd1084374cb14a5655b06a856c3df5f15"
HELPER_HASH = "55424584c76a209fd54479179c28cfd46de280844575fb919634b192c28a6f26"


def replace(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f"v12 source anchor mismatch: {old[:100]!r}")
    return source.replace(old, new, 1)


def transform(source, helper):
    source = replace(source, "import numpy as np\n", "import os\nimport numpy as np\n")
    source = replace(source, "class B12xMLASparseBackend(AttentionBackend):",
                     helper + "\n\nclass B12xMLASparseBackend(AttentionBackend):")
    source = replace(source, "    ckv_active_counts: torch.Tensor | None = None\n",
                     "    ckv_active_counts: torch.Tensor | None = None\n"
                     "    ckv_causal_lens: torch.Tensor | None = None\n")
    source = replace(source, "            self.ckv_active_counts_buffer = torch.empty(\n",
                     "            self.ckv_causal_lens_buffer = torch.empty(\n"
                     "                (max_tokens,), dtype=torch.int32, device=device\n"
                     "            )\n            self.ckv_active_counts_buffer = torch.empty(\n")
    source = replace(source, "            self.ckv_active_counts_buffer = None\n",
                     "            self.ckv_active_counts_buffer = None\n"
                     "            self.ckv_causal_lens_buffer = None\n")
    source = replace(source, "                metadata.ckv_active_counts = self.ckv_active_counts_buffer[:num_tokens]\n",
                     "                metadata.ckv_active_counts = self.ckv_active_counts_buffer[:num_tokens]\n"
                     "                metadata.ckv_causal_lens = self.ckv_causal_lens_buffer[:num_tokens]\n")
    source = replace(source, "                metadata.dcp_ckv_gather_eligible = True\n",
                     "                metadata.dcp_ckv_gather_eligible = True\n"
                     "            elif padded_total_tokens > max_local_capacity:\n"
                     "                logger.info_once(\n"
                     '                    "v12: CKV context-capacity fallback: padded rank tokens=%d capacity=%d",\n'
                     "                    padded_total_tokens, max_local_capacity,\n"
                     "                )\n")
    # Flags are frozen at backend setup, not re-read per layer invocation.
    source = replace(source, "        self._ckv_gather_enabled = (\n",
                     '        self._v12_fused_ckv = os.getenv("VLLM_GLM53_FUSED_CKV_METADATA", "0") == "1"\n'
                     '        self._v12_borrow_query = os.getenv("VLLM_GLM53_BORROW_MLA_QUERY", "0") == "1"\n'
                     "        self._ckv_gather_enabled = (\n")
    old = "            if not exact_workspace_alias:\n                q_all.copy_(q)\n"
    source = replace(source, old,
                     "            if _v12_can_borrow_query(\n"
                     "                q, scratch, self._v12_borrow_query, num_tokens, input_num_heads, self._q_head_dim\n"
                     "            ):\n                q_all = q\n"
                     '                logger.info_once("v12: borrowed contiguous MLA query active")\n'
                     "            elif not exact_workspace_alias:\n                q_all.copy_(q)\n")
    begin = source.index("            _map_global_topk_to_gathered_ckv(\n", source.index("    def forward_mqa("))
    end = source.index("            _mask_page_table_after_nsa_len(selected_indices, active_counts)\n", begin)
    end += len("            _mask_page_table_after_nsa_len(selected_indices, active_counts)\n")
    legacy = source[begin:end]
    source = source[:begin] + (
        "            if self._is_glm_dsa and self._v12_fused_ckv:\n"
        "                assert attn_metadata.ckv_causal_lens is not None\n"
        "                assert attn_metadata.global_cache_seq_lens_per_req is not None\n"
        "                cache_seq_lens = attn_metadata.ckv_causal_lens[:num_tokens]\n"
        "                _v12_prepare_ckv_metadata(\n"
        "                    attn_metadata.req_id_per_token[:num_tokens], topk_indices,\n"
        "                    attn_metadata.dcp_rank_req_starts, attn_metadata.dcp_rank_req_lens,\n"
        "                    attn_metadata.global_cache_seq_lens_per_req, attn_metadata.query_start_loc,\n"
        "                    selected_indices, active_counts, cache_seq_lens,\n"
        "                    dcp_size=self.dcp_world_size,\n"
        "                    interleave=attn_metadata.cp_kv_cache_interleave_size,\n"
        "                    padded_tokens=attn_metadata.dcp_padded_total_tokens,\n"
        "                )\n"
        '                logger.info_once("v12: fused GLM-DSA CKV metadata active")\n'
        "            else:\n" + "".join("    " + line for line in legacy.splitlines(keepends=True))
    ) + source[end:]
    return source


def patch(root, check=False):
    path = Path(root) / TARGET
    source = path.read_text(encoding="utf-8")
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest == OUTPUT_HASH:
        return
    if check or digest != INPUT_HASH:
        raise RuntimeError(f"unexpected v12 source: {path} ({digest})")
    helper = Path(__file__).with_name("r22_v12_ckv.py").read_text(encoding="utf-8")
    if hashlib.sha256(helper.encode()).hexdigest() != HELPER_HASH:
        raise RuntimeError("v12 helper hash mismatch")
    result = transform(source, helper)
    compile(result, str(path), "exec")
    if hashlib.sha256(result.encode()).hexdigest() != OUTPUT_HASH:
        raise RuntimeError("v12 result hash mismatch")
    path.write_text(result, encoding="utf-8", newline="\n")
    print(f"{VERSION}: applied to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    patch(args.root, args.check)
