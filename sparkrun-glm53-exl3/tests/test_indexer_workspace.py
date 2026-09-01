#!/usr/bin/env python3
"""Host tests for the SparkRing sparse-indexer workspace overlay."""

from __future__ import annotations

import importlib.util
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "overlay/patch_indexer_workspace.py"
SPEC = importlib.util.spec_from_file_location("patch_indexer_workspace", PATCH)
assert SPEC is not None and SPEC.loader is not None
overlay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(overlay)


FIXTURE = f'''from __future__ import annotations
import os

def cdiv(a, b):
    return (a + b - 1) // b

class VllmConfig:
    pass

class MLAAttentionSpec:
    pass

class Logger:
    def info(self, *args):
        pass
    def warning(self, *args):
        pass

logger = Logger()

{overlay.FUNCTION_ANCHOR}

class Builder:
    def configure(self):
        self.compress_ratio = 1
{overlay.BUILDER_ANCHOR}            pass
'''


def patched_fixture() -> str:
    old_input = overlay.INPUT_SHA256
    old_output = overlay.OUTPUT_SHA256
    try:
        overlay.INPUT_SHA256 = overlay.sha256_text(FIXTURE)
        overlay.OUTPUT_SHA256 = "__OUTPUT_SHA256__"
        patched, action = overlay.prepare(FIXTURE)
        assert action == "patched"
        return patched
    finally:
        overlay.INPUT_SHA256 = old_input
        overlay.OUTPUT_SHA256 = old_output


def config(*, max_len=1_048_576, seqs=16, batched=4096, ratio=4, spec=3):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            max_model_len=max_len,
            hf_text_config=SimpleNamespace(index_kpool=ratio),
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=seqs,
            max_num_batched_tokens=batched,
        ),
        speculative_config=(
            SimpleNamespace(num_speculative_tokens=spec) if spec is not None else None
        ),
    )


def test_exact_sparkring_source_contract() -> None:
    source = os.environ.get("GLM53_INDEXER_TEST_SOURCE")
    if not source:
        return
    text = Path(source).read_text(encoding="utf-8")
    assert overlay.sha256_text(text) == overlay.INPUT_SHA256
    patched, action = overlay.prepare(text)
    assert action == "patched"
    assert overlay.sha256_text(patched) == overlay.OUTPUT_SHA256


def test_formula_and_dispatch() -> None:
    namespace: dict[str, object] = {}
    exec(patched_fixture(), namespace)
    get_size = namespace["get_max_prefill_buffer_size"]
    rightsize = namespace["_glm53_rightsized_workspace_entries"]
    cfg = config()
    stock = 1_048_576 * 40
    expected = 16 * 262_145
    assert rightsize(cfg) == expected == 4_194_320

    saved = os.environ.get(overlay.ENV_NAME)
    try:
        os.environ[overlay.ENV_NAME] = "stock"
        assert get_size(cfg) == stock
        os.environ[overlay.ENV_NAME] = "rightsize"
        assert get_size(cfg) == expected
        # MNBT cannot admit more requests than its token budget.
        assert rightsize(config(seqs=16, batched=8)) == 8 * 262_145
        # A non-compressed indexer retains the exact stock expression.
        assert rightsize(config(ratio=1)) == stock
        os.environ[overlay.ENV_NAME] = "typo"
        try:
            get_size(cfg)
        except ValueError as exc:
            assert overlay.ENV_NAME in str(exc)
        else:
            raise AssertionError("unknown mode must fail closed")
    finally:
        if saved is None:
            os.environ.pop(overlay.ENV_NAME, None)
        else:
            os.environ[overlay.ENV_NAME] = saved


def test_patch_is_idempotent_and_rejects_drift() -> None:
    patched = patched_fixture()
    old_input = overlay.INPUT_SHA256
    old_output = overlay.OUTPUT_SHA256
    try:
        overlay.INPUT_SHA256 = overlay.sha256_text(FIXTURE)
        overlay.OUTPUT_SHA256 = overlay.sha256_text(patched)
        again, action = overlay.prepare(patched)
        assert action == "already present"
        assert again == patched
        try:
            overlay.prepare(FIXTURE + "# drift\n")
        except RuntimeError as exc:
            assert "unknown indexer.py state" in str(exc)
        else:
            raise AssertionError("source drift must fail closed")
    finally:
        overlay.INPUT_SHA256 = old_input
        overlay.OUTPUT_SHA256 = old_output


def test_cli_preflight_does_not_write() -> None:
    old_input = overlay.INPUT_SHA256
    old_output = overlay.OUTPUT_SHA256
    patched = patched_fixture()
    try:
        overlay.INPUT_SHA256 = overlay.sha256_text(FIXTURE)
        overlay.OUTPUT_SHA256 = overlay.sha256_text(patched)
        with tempfile.TemporaryDirectory() as raw:
            target = Path(raw) / "indexer.py"
            target.write_text(FIXTURE, encoding="utf-8", newline="\n")
            before = target.read_bytes()
            source, action = overlay.prepare(target.read_text(encoding="utf-8"))
            assert action == "patched"
            compile(source, str(target), "exec")
            assert target.read_bytes() == before
    finally:
        overlay.INPUT_SHA256 = old_input
        overlay.OUTPUT_SHA256 = old_output


def test_recipe_and_image_wiring() -> None:
    dockerfile = (ROOT / "Dockerfile.prefill").read_text(encoding="utf-8")
    recipe = (ROOT / "recipes/glm53-exl3-4x-safe.yaml").read_text(encoding="utf-8")
    builder = (ROOT / "scripts/build-prefill-image.sh").read_text(encoding="utf-8")
    assert "FROM spark-vllm-glm52-exl3:sparkring-switch-v1" in dockerfile
    assert dockerfile.count("patch_indexer_workspace.py --preflight") == 2
    assert "container: spark-vllm-glm52-exl3:sparkring-switch-prefill-v2" in recipe
    assert "GLM53_INDEXER_WORKSPACE: rightsize" in recipe
    assert 'VLLM_EXL3_PREFILL_CAPACITY: "4096"' in recipe
    assert "kv_cache_memory_bytes: 18000000000" in recipe
    assert "--kv-cache-memory-bytes {kv_cache_memory_bytes}" in recipe
    assert "docker save" in builder and "docker load" in builder


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"indexer workspace overlay OK ({len(tests)} tests)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
