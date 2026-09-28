#!/usr/bin/env python3
"""Opt-in 8192-token DeepSeek V4.1 single-request piecewise graph."""
import argparse
import hashlib
from pathlib import Path

RELATIVE = 'models/deepseek_v4_1/nvidia/model_state.py'
SOURCE_SHA256 = '7004e7fc0834627a2975a138f7b997806e5a86bab11c80b18997292c59b5aa02'
REPLACEMENTS = (
    (b'from typing import Any\n', b'import os\nfrom typing import Any\n'),
    (b'                and self.max_num_tokens == 4096\n',
     b'                and (\n'
     b'                    self.max_num_tokens == 4096\n'
     b'                    or (\n'
     b'                        self.max_num_tokens == 8192\n'
     b'                        and os.environ.get("DS41_PREFILL_8192_GRAPH") == "1"\n'
     b'                    )\n'
     b'                )\n'),
    (b'                self.single_request_prefill_cudagraph_tokens = 4096\n',
     b'                self.single_request_prefill_cudagraph_tokens = self.max_num_tokens\n'),
    (b'    def can_use_single_request_prefill_graph(self, num_reqs, num_tokens, req_ids):\n'
     b'        return (\n'
     b'            self.single_request_prefill_cudagraph_tokens > 0\n'
     b'            and num_reqs == 1\n'
     b'            and num_tokens == self.single_request_prefill_cudagraph_tokens\n'
     b'            and not any(req in self._ced_prompt_logprobs for req in req_ids)\n'
     b'        )\n',
     b'    def can_use_single_request_prefill_graph(self, num_reqs, num_tokens, req_ids):\n'
     b'        eligible = (\n'
     b'            self.single_request_prefill_cudagraph_tokens > 0\n'
     b'            and num_reqs == 1\n'
     b'            and num_tokens == self.single_request_prefill_cudagraph_tokens\n'
     b'            and not any(req in self._ced_prompt_logprobs for req in req_ids)\n'
     b'        )\n'
     b'        if eligible and num_tokens == 8192 and not getattr(self, "_ds41_8192_reported", False):\n'
     b'            print("DS41 runtime 8192-token prefill graph eligible", flush=True)\n'
     b'            self._ds41_8192_reported = True\n'
     b'        return eligible\n'),
)
RUNNER_RELATIVE = 'v1/worker/gpu/model_runner.py'
# The pinned runner after the existing ds41-dspark-prefill-v1 overlay.
RUNNER_SHA256 = '4c55159cf4d40cca5472a4bd00e9a255fc31fa5b9a2886278206771b7048983a'
RUNNER_REPLACEMENTS = (
    (b'                    assert self.cudagraph_manager is not None\n'
     b'                    model_output = self.cudagraph_manager.run_pw_graph(\n',
     b'                    assert self.cudagraph_manager is not None\n'
     b'                    if (\n'
     b'                        not dummy_run\n'
     b'                        and batch_desc.num_tokens == 8192\n'
     b'                        and self.model_state.single_request_prefill_cudagraph_tokens == 8192\n'
     b'                        and not getattr(self, "_ds41_8192_replay_reported", False)\n'
     b'                    ):\n'
     b'                        print("DS41 serving 8192-token PIECEWISE graph replay active", flush=True)\n'
     b'                        self._ds41_8192_replay_reported = True\n'
     b'                    model_output = self.cudagraph_manager.run_pw_graph(\n'),
)
PATCHES = {
    RELATIVE: (SOURCE_SHA256, REPLACEMENTS),
    RUNNER_RELATIVE: (RUNNER_SHA256, RUNNER_REPLACEMENTS),
}


def transform(data, reverse=False, replacements=REPLACEMENTS):
    for original, changed in reversed(replacements) if reverse else replacements:
        before, after = (changed, original) if reverse else (original, changed)
        if data.count(before) != 1:
            raise RuntimeError('8192 graph source anchor absent or ambiguous')
        data = data.replace(before, after, 1)
    return data


def patch(root, *, check=False, revert=False):
    pending = []
    for relative, (digest, replacements) in PATCHES.items():
        path = Path(root) / relative
        current = path.read_bytes()
        original = current if hashlib.sha256(current).hexdigest() == digest else transform(current, True, replacements)
        if hashlib.sha256(original).hexdigest() != digest:
            raise RuntimeError(f'Unexpected pinned graph source: {relative}; re-audit before patching')
        expected = original if revert else transform(original, replacements=replacements)
        compile(expected, str(path), 'exec')
        if check and current != expected:
            raise RuntimeError('8192 graph patch state differs')
        pending.append((path, current, expected))
    if not check:
        for path, current, expected in pending:
            if current != expected:
                path.write_bytes(expected)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('vllm', type=Path)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--revert', action='store_true')
    args = parser.parse_args()
    patch(args.vllm, check=args.check, revert=args.revert)
