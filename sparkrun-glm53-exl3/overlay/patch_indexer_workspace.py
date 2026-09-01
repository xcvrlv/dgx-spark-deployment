#!/usr/bin/env python3
"""Right-size GLM-5.3's locked sparse-indexer prefill workspace.

Adapted for the pinned SparkRing R7 vLLM tree from MiaAI-Lab's Apache-2.0
indexer workspace overlay (commit 2022ce5).  This patch is deliberately
independent of PR 77's uniform-K4 EXL3 fat-expert kernel.

The stock indexer reserves ``max_model_len * 40`` entries.  GLM-5.3 presents
compressed k-pool lengths to the only consumer, so the legal per-step maximum
is bounded by the number of requests the scheduler can admit times one
compressed maximum-length request.  The result is clamped to stock and is
enabled only with ``GLM53_INDEXER_WORKSPACE=rightsize``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path


ENV_NAME = "GLM53_INDEXER_WORKSPACE"
DEFAULT_TARGET = Path(
    "/opt/venv/lib/python3.12/site-packages/vllm/"
    "v1/attention/backends/mla/indexer.py"
)
INPUT_SHA256 = "9757e1763137bf25886b1ea90d44d928b310cfba875dc64705596d2134815ecf"
# Filled after applying the deterministic replacements below.  Keeping both
# hashes makes image rebuilds idempotent while rejecting an unknown vLLM ABI.
OUTPUT_SHA256 = "e71694e06125a30593b5eb73903ee698d4b270b1be9500d24199b94e614f4d37"

MARKER = "# GLM53_INDEXER_WORKSPACE_RIGHTSIZE_V1"

FUNCTION_ANCHOR = '''def get_max_prefill_buffer_size(vllm_config: VllmConfig):
    max_model_len = vllm_config.model_config.max_model_len
    # NOTE(Chen): 40 is a magic number for controlling the prefill buffer size.
    # Each entry is 128 fp8 bytes and 4 scale bytes for a total of 132 bytes.
    # The flashmla_sparse backend uses a workspace size of 5 * max_model_len.
    # The memory usage of the workspace there is 576 * 2 bytes; so we size this as
    # (576 * 2 // 132) * 5 = 40 to maximize this workspace size while still fitting
    # within the flashmla_sparse workspace.
    # For DeepSeek-V3.2, the max_model_len is 163840.
    #   40 * 163840 * 132 = 865075200 bytes = 825 MB
    return max_model_len * 40
'''

FUNCTION_REPLACEMENT = '''# GLM53_INDEXER_WORKSPACE_RIGHTSIZE_V1
_GLM53_INDEXER_WORKSPACE_ENV = "GLM53_INDEXER_WORKSPACE"
_GLM53_INDEXER_STOCK_MULTIPLIER = 40


def _glm53_indexer_workspace_mode() -> str:
    mode = os.environ.get(_GLM53_INDEXER_WORKSPACE_ENV, "stock")
    if mode not in ("stock", "rightsize"):
        raise ValueError(
            f"{_GLM53_INDEXER_WORKSPACE_ENV} must be 'stock' or 'rightsize', "
            f"got {mode!r}"
        )
    return mode


def _glm53_indexer_compress_ratio(vllm_config: VllmConfig) -> int:
    text_config = vllm_config.model_config.hf_text_config
    try:
        ratio = int(getattr(text_config, "index_kpool", 1) or 1)
    except (TypeError, ValueError):
        return 1
    return max(1, ratio)


def _glm53_rightsized_workspace_entries(vllm_config: VllmConfig) -> int:
    max_model_len = int(vllm_config.model_config.max_model_len)
    stock = max_model_len * _GLM53_INDEXER_STOCK_MULTIPLIER
    ratio = _glm53_indexer_compress_ratio(vllm_config)
    if ratio <= 1:
        return stock

    scheduler = vllm_config.scheduler_config
    max_requests = max(
        1,
        min(
            int(scheduler.max_num_seqs),
            int(scheduler.max_num_batched_tokens),
        ),
    )
    speculative = vllm_config.speculative_config
    spec_tokens = (
        int(speculative.num_speculative_tokens) if speculative is not None else 0
    )
    per_request = cdiv(max_model_len + spec_tokens, ratio)
    return min(stock, max_requests * per_request)


def get_max_prefill_buffer_size(vllm_config: VllmConfig):
    max_model_len = vllm_config.model_config.max_model_len
    stock = max_model_len * _GLM53_INDEXER_STOCK_MULTIPLIER
    if _glm53_indexer_workspace_mode() == "stock":
        return stock

    entries = _glm53_rightsized_workspace_entries(vllm_config)
    ratio = _glm53_indexer_compress_ratio(vllm_config)
    if ratio <= 1:
        logger.warning(
            "[%s] rightsize requested without index_kpool compression; "
            "retaining stock workspace (%d entries)",
            _GLM53_INDEXER_WORKSPACE_ENV,
            stock,
        )
    elif entries >= stock:
        logger.warning(
            "[%s] legal maximum reaches stock; retaining %d entries",
            _GLM53_INDEXER_WORKSPACE_ENV,
            stock,
        )
    else:
        logger.info(
            "[%s] rightsize: %d -> %d entries (ratio=%d)",
            _GLM53_INDEXER_WORKSPACE_ENV,
            stock,
            entries,
            ratio,
        )
    return entries
'''

BUILDER_ANCHOR = '''        # Get compress_ratio for DeepseekV4 support
        if isinstance(self.kv_cache_spec, MLAAttentionSpec):
            self.compress_ratio = self.kv_cache_spec.compress_ratio

        # DCP writes the indexer cache through rank-local pages. DeepSeek V4
'''

BUILDER_REPLACEMENT = '''        # Get compress_ratio for DeepseekV4 support
        if isinstance(self.kv_cache_spec, MLAAttentionSpec):
            self.compress_ratio = self.kv_cache_spec.compress_ratio

        if _glm53_indexer_workspace_mode() == "rightsize":
            configured_ratio = _glm53_indexer_compress_ratio(self.vllm_config)
            if configured_ratio != self.compress_ratio:
                raise ValueError(
                    f"{_GLM53_INDEXER_WORKSPACE_ENV}=rightsize requires matching "
                    f"index_kpool and runtime compression ratios; "
                    f"index_kpool={configured_ratio}, "
                    f"compress_ratio={self.compress_ratio}"
                )
            legal_entries = _glm53_rightsized_workspace_entries(self.vllm_config)
            if self.max_prefill_buffer_size < legal_entries:
                raise ValueError(
                    f"{_GLM53_INDEXER_WORKSPACE_ENV}=rightsize produced an "
                    f"undersized workspace: {self.max_prefill_buffer_size} < "
                    f"{legal_entries} entries"
                )
            logger.info(
                "[%s] builder verified %d entries (index_kpool=%d, "
                "compress_ratio=%d)",
                _GLM53_INDEXER_WORKSPACE_ENV,
                self.max_prefill_buffer_size,
                configured_ratio,
                self.compress_ratio,
            )

        # DCP writes the indexer cache through rank-local pages. DeepSeek V4
'''


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prepare(text: str) -> tuple[str, str]:
    observed = sha256_text(text)
    if observed == OUTPUT_SHA256 and MARKER in text:
        return text, "already present"
    if observed != INPUT_SHA256:
        raise RuntimeError(
            "refusing unknown indexer.py state: "
            f"expected {INPUT_SHA256} or {OUTPUT_SHA256}, got {observed}"
        )
    if text.count(FUNCTION_ANCHOR) != 1:
        raise RuntimeError("prefill workspace function anchor drifted")
    if text.count(BUILDER_ANCHOR) != 1:
        raise RuntimeError("metadata builder compression anchor drifted")
    patched = text.replace(FUNCTION_ANCHOR, FUNCTION_REPLACEMENT, 1)
    patched = patched.replace(BUILDER_ANCHOR, BUILDER_REPLACEMENT, 1)
    produced = sha256_text(patched)
    if OUTPUT_SHA256 != "__OUTPUT_SHA256__" and produced != OUTPUT_SHA256:
        raise RuntimeError(
            f"patched indexer hash mismatch: expected {OUTPUT_SHA256}, got {produced}"
        )
    compile(patched, "indexer.py", "exec")
    return patched, "patched"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        type=Path,
        default=Path(os.environ.get("GLM53_INDEXER_BACKEND_PY", DEFAULT_TARGET)),
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="validate the target and deterministic output without writing",
    )
    args = parser.parse_args()
    target = args.target.resolve()
    if not target.is_file():
        raise RuntimeError(f"indexer target is missing: {target}")
    source = target.read_text(encoding="utf-8")
    patched, action = prepare(source)
    if not args.preflight and action == "patched":
        target.write_text(patched, encoding="utf-8", newline="\n")
    verb = "preflight OK" if args.preflight else action
    print(f"{verb}: {target} ({sha256_text(patched)})")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"indexer workspace patch failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
