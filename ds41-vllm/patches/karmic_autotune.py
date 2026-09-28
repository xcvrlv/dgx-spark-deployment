#!/usr/bin/env python3
"""Bounded preparation for audited Karmic 1794dcf/502d6cb and b12x a7d7d29/d44247b."""
import argparse
import hashlib
from pathlib import Path

HELPER = 'preparation/ds41_tuning.py'
PATCHES = {
    ('b12x', 'preparation/_memory.py'): (
        '81ec9916a1dc3c8801ee0ff8987ccdc57914264b0687c3560aa304fd3cdd6e49', [
            ('def _counter():\n',
             'def _counter():\n'
             '    from . import ds41_tuning\n'
             '    if ds41_tuning.enabled():\n'
             '        return ds41_tuning.memory_counter()\n'),
        ]),
    ('vllm', 'model_executor/warmup/b12x_prepare.py'): (
        '2d730fc5327f0b3cc2159b7ad3b903d675f9d1d1c82fd2b2cef222cb7e09a9e9', [
            ('    session = PreparationSession(\n',
             '    from b12x.preparation import ds41_tuning\n'
             '    if ds41_tuning.enabled():\n'
             '        namespace["ds41_tuning_profile"] = ds41_tuning.PROFILE\n'
             '        ds41_tuning.require_headroom()\n'
             '        logger.info("DS41 bounded autotune: one candidate plus champion; "\n'
             '                    "limits=%s", ds41_tuning.session_options())\n'
             '    session = PreparationSession(\n'),
            ('        namespace=namespace,\n',
             '        namespace=namespace,\n        **ds41_tuning.session_options(),\n'),
        ]),
    ('b12x', 'preparation/session.py'): (
        'e729ef26952d485de719a3359d36ed768ebe06facab102002e53f5b32a9d530c', [
            ('    def _race_budget(self):\n',
             '    def _race_budget(self):\n'
             '        from . import ds41_tuning\n'
             '        if ds41_tuning.enabled():\n'
             '            import torch\n'
             '            free, _ = torch.cuda.mem_get_info(self.device.ordinal)\n'
             '            return ds41_tuning.bounded_budget(\n'
             '                self.race_budget or ds41_tuning.session_options()["race_budget"],\n'
             '                int(free), ds41_tuning.available_bytes(),\n'
             '                ds41_tuning.reserve_bytes())\n'),
            ('    def advance(self, *, collective_key=None, tuning=None, cache=None):\n',
             '    def advance(self, *, collective_key=None, tuning=None, cache=None):\n'
             '        from . import ds41_tuning\n'
             '        ds41_tuning.require_headroom()\n'
             '        ds41_tuning.write_progress(self)\n'),
            ('            self._last_advance_end = time.perf_counter()\n',
             '            self._last_advance_end = time.perf_counter()\n'
             '            ds41_tuning.write_progress(self, force=self._result is not None)\n'),
            ('                    trial = yield from self._trial(obligation, index, assignment, config)\n',
             '                    from . import ds41_tuning\n'
             '                    ds41_tuning.require_headroom()\n'
             '                    trial = yield from self._trial(obligation, index, assignment, config)\n'),
            ('                    live.append(trial)\n',
             '                    live.append(trial)\n'
             '                    ds41_tuning.require_headroom()\n'),
        ]),
}

# At 502d6cb this file changes only two local-variable type annotations.
# Checked against live Karmic, JJ and b12x heads on 2026-09-28.
ADMITTED_SHAS = {
    ('vllm', 'model_executor/warmup/b12x_prepare.py'): {
        PATCHES[('vllm', 'model_executor/warmup/b12x_prepare.py')][0],
        '203210fc97234af2892fe4ce35191fc5ec7ddc958265042e212b541a377627a0',
    },
}


def transform(data, replacements, reverse=False):
    for old, new in reversed(replacements) if reverse else replacements:
        before, after = (new, old) if reverse else (old, new)
        if data.count(before.encode()) != 1:
            raise RuntimeError('Source anchor absent or ambiguous; re-audit upstream')
        data = data.replace(before.encode(), after.encode(), 1)
    return data


def patch(vllm, b12x, *, check=False, revert=False):
    roots = {'vllm': Path(vllm), 'b12x': Path(b12x)}
    helper = Path(__file__).with_name('karmic_autotune_runtime.py').read_bytes().replace(b'\r\n', b'\n')
    destination = roots['b12x'] / HELPER
    if destination.exists() and destination.read_bytes() != helper:
        raise RuntimeError('Unexpected installed autotune helper')
    pending = []
    for (tree, relative), (digest, replacements) in PATCHES.items():
        path = roots[tree] / relative
        data = path.read_bytes()
        admitted = ADMITTED_SHAS.get((tree, relative), {digest})
        original = data if hashlib.sha256(data).hexdigest() in admitted else transform(data, replacements, True)
        if hashlib.sha256(original).hexdigest() not in admitted:
            raise RuntimeError(f'Unexpected upstream source: {path}')
        expected = original if revert else transform(original, replacements)
        compile(expected, str(path), 'exec')
        if check and data != expected:
            raise RuntimeError(f'Unexpected patch state: {path}')
        pending.append((path, expected))
    if check:
        if destination.exists() == revert:
            raise RuntimeError('Unexpected autotune helper state')
        return
    for path, data in pending:
        path.write_bytes(data)
    if revert:
        destination.unlink(missing_ok=True)
    else:
        destination.write_bytes(helper)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('vllm', type=Path)
    parser.add_argument('b12x', type=Path)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--revert', action='store_true')
    args = parser.parse_args()
    patch(args.vllm, args.b12x, check=args.check, revert=args.revert)
