#!/usr/bin/env python3
"""Hash-guarded, opt-in DSpark prefill overlay for Karmic 1794dcf.

Upstream checked 2026-09-27: KK 953a636, JJ 8e1f1e5, b12x d44247b.
See DSPARK-PREFILL.md. All inputs are validated before any file is written.
"""
import argparse
import hashlib
from pathlib import Path

RUNTIME = 'v1/worker/gpu/spec_decode/ds41_prefill.py'
RUNTIME_SOURCE = Path(__file__).with_name('dspark_prefill_runtime.py')
PATCHES = {
    'v1/worker/gpu/spec_decode/dflash/speculator.py': (
        '1ba0642931d7e9b1b7b04fef0e0eaf73d85b4ffb87025a1cfbfc74b8b28b62f7', [
        ('from vllm.v1.worker.gpu.spec_decode.dflash.utils import load_dflash_model\n',
         'from vllm.v1.worker.gpu.spec_decode.dflash.utils import load_dflash_model\n'
         'from vllm.v1.worker.gpu.spec_decode.ds41_prefill import (\n'
         '    skip_prefill_draft, use_compact_context_graph,\n'
         ')\n'),
        ('        # Drop graphs before detaching the KV cache tensors they reference.\n'
         '        self._context_preparer = None\n',
         '        # Drop graphs before detaching the KV cache tensors they reference.\n'
         '        if self._context_preparer is not None:\n'
         '            self._context_preparer.close()\n'
         '        self._context_preparer = None\n'),
        ('            and ced_indices is None\n',
         '            and (ced_indices is None or use_compact_context_graph(\n'
         '                self, input_batch, num_context_tokens\n'
         '            ))\n'),
        ('            and self._context_preparer.can_run(num_target_tokens)\n',
         '            and self._context_preparer.can_run(num_context_tokens)\n'),
        ('        if not context_kv_is_restored and not use_context_graph:\n'
         '            if ced_indices is not None:\n'
         '                # Target aux keeps the original-row ABI. Pack before the expensive\n'
         '                # combine projection; discarded decoder rows have no context KV.\n'
         '                if aux_hidden_states:\n'
         '                    aux_hidden_states = [\n'
         '                        gather_rows(hidden, ced_indices) for hidden in aux_hidden_states\n'
         '                    ]\n'
         '                else:\n'
         '                    last_hidden_states = gather_rows(last_hidden_states, ced_indices)\n',
         '        if not context_kv_is_restored and ced_indices is not None:\n'
         '            # Pack aux before either eager projection or the compact graph.\n'
         '            if aux_hidden_states:\n'
         '                aux_hidden_states = [\n'
         '                    gather_rows(hidden, ced_indices) for hidden in aux_hidden_states\n'
         '                ]\n'
         '            else:\n'
         '                last_hidden_states = gather_rows(last_hidden_states, ced_indices)\n'
         '        if not context_kv_is_restored and not use_context_graph:\n'),
        ('        # replay bounded decode context graphs; compact prefills use eager projection.\n',
         '        # replay bounded context graphs; unsupported compact batches stay eager.\n'),
        ('        if use_context_graph:\n'
         '            self._context_preparer.run(aux_hidden_states, num_target_tokens)\n',
         '        if use_context_graph:\n'
         '            if ced_indices is not None:\n'
         '                logger.info_once("DS41 compact-prefill context graph replay active")\n'
         '                # These context buffers are separate from the query input\n'
         '                # positions and BlockTables slots. Pack only AFTER anchors\n'
         '                # and rejection masks were prepared in original coordinates.\n'
         '                self.context_positions[:num_context_tokens].copy_(context_positions)\n'
         '                for destination, source in zip(\n'
         '                    self._context_slot_mappings, packed_context_slots\n'
         '                ):\n'
         '                    destination[:num_context_tokens].copy_(source)\n'
         '            self._context_preparer.run(aux_hidden_states, num_context_tokens)\n'),
        ('        batch_sync, num_batch_tokens = (\n',
         '        # Context KV is required by later chunks even when the scheduler\n'
         '        # cannot consume this chunk\'s draft tokens. Skip only the query\n'
         '        # backbone and sampling, returning the existing empty-width ABI.\n'
         '        if skip_prefill_draft(\n'
         '            self, input_batch, dummy_run=dummy_run, is_profile=is_profile,\n'
         '            dp_sync=dp_sync, context_kv_is_restored=context_kv_is_restored,\n'
         '        ):\n'
         '            logger.info_once("DS41 skipping unused intermediate-prefill drafts")\n'
         '            return self.draft_tokens[:num_reqs, :0]\n\n'
         '        batch_sync, num_batch_tokens = (\n'),
    ]),
    'models/deepseek_v4_1/nvidia/dspark.py': (
        'b11eee974ebe44d7e63acd09ce902da5520d2574e2c82c1bca719a293a97a1d0', [
        ('from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager\n',
         'from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager\n'
         'from vllm.v1.worker.gpu.spec_decode.ds41_prefill import (\n'
         '    COMPACT_ROWS, compact_context_enabled,\n'
         ')\n'),
        ('    Only bounded decode capacities are captured. Live source tensors are copied,\n',
         '    Bounded decode and opt-in 128-row context capacities are captured.\n'
         '    Live source tensors are copied,\n'),
        ('        self.aux = torch.zeros(\n',
         '        # Preserve EVERY original decode bucket, then add at most one\n'
         '        # compact-prefill bucket. This runs inside startup graph memory\n'
         '        # profiling, never on the first live request. Native CED projection\n'
         '        # plans already include the 128-row sliding-window capacity.\n'
         '        if compact_context_enabled(vllm_config) and hidden_states.shape[0] >= COMPACT_ROWS:\n'
         '            capacities = sorted(set(capacities) | {COMPACT_ROWS})\n'
         '            limit = max(limit, COMPACT_ROWS)\n'
         '        self.aux = torch.zeros(\n'),
    ]),
    'v1/worker/gpu/model_runner.py': (
        '9322af06b06072d2aa75aaa8aa8068dbca6218b51a21c0cc0bcfe8d72b883ba9', [
        ('            if self.adaptive_verification is not None:\n'
         '                self.adaptive_verification.record_confidences(\n',
         '            # An empty draft (intermediate prefill) has no fresh confidence.\n'
         '            if self.adaptive_verification is not None and num_draft_tokens > 0:\n'
         '                self.adaptive_verification.record_confidences(\n'),
    ]),
}


def transform(data, replacements, *, reverse=False):
    for old, new in reversed(replacements) if reverse else replacements:
        before, after = (new, old) if reverse else (old, new)
        before, after = before.encode(), after.encode()
        if data.count(before) != 1:
            raise RuntimeError('Source anchor absent or ambiguous; re-audit upstream')
        data = data.replace(before, after, 1)
    return data


def patch(root, *, revert=False, check=False):
    root = Path(root)
    runtime = root / RUNTIME
    helper = RUNTIME_SOURCE.read_bytes().replace(b'\r\n', b'\n')
    if runtime.exists() and runtime.read_bytes() != helper:
        raise RuntimeError(f'Unexpected installed helper: {runtime}')
    pending = []
    for relative, (sha, replacements) in PATCHES.items():
        path = root / relative
        data = path.read_bytes()
        original = data
        if hashlib.sha256(original).hexdigest() != sha:
            original = transform(data, replacements, reverse=True)
        if hashlib.sha256(original).hexdigest() != sha:
            raise RuntimeError(f'Unexpected upstream source: {path}; re-audit upstream')
        expected = original if revert else transform(original, replacements)
        compile(expected, str(path), 'exec')
        if check and data != expected:
            raise RuntimeError(f'Unexpected patch state: {path}')
        pending.append((path, expected))
    if check:
        if runtime.exists() == revert:
            raise RuntimeError(f'Unexpected helper state: {runtime}')
        return
    for path, expected in pending:
        if path.read_bytes() != expected:
            path.write_bytes(expected)
    if revert:
        runtime.unlink(missing_ok=True)
    else:
        runtime.write_bytes(helper)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package', type=Path)
    parser.add_argument('--revert', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    patch(args.package, revert=args.revert, check=args.check)
