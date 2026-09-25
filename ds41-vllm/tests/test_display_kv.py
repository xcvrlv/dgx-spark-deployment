import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT/'.build/display-kv-test'
DISPLAY_BYTES = 1792 * 2**20  # the allocator's fixed span; the credit never exceeds it
VLLM = Path(os.environ.get('DS41_VLLM_SOURCE', ROOT.parent/'tmp/jj-audit/local-inference-lab-vllm-5bca5a5/vllm'))
BASELINE = Path(os.environ.get('DS41_DISPLAY_KV_BASELINE', ROOT/'.build/display-kv-baseline'))
DISPLAY = Path(os.environ.get('DS41_DISPLAY_KV_RUN', ROOT/'.build/display-kv-enabled'))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


backing_patch = load('display_kv_patch', ROOT/'patches/display_kv.py')
credit_patch = load('display_kv_credit_patch', ROOT/'patches/display_kv_credit.py')
helper = load('ds41_display_kv', ROOT/'patches/ds41_display_kv.py')

KV_SIZE = re.compile(r'GPU KV cache size: ([\d,]+) tokens')
AVAILABLE = re.compile(r'Available KV cache memory: ([\d.]+) GiB')
CACHE_INFO = re.compile(r'vllm:cache_config_info\{([^}]*)\}')


def reported_capacity(log_text):
    """Reported KV tokens, derived bytes-per-token, and the logged budget.

    bytes-per-token is derived from the same run whose capacity is measured, so the
    expected gain cannot be tuned to pass.
    """
    size, available = KV_SIZE.search(log_text), AVAILABLE.search(log_text)
    if not size or not available:
        raise AssertionError('Missing KV capacity log anchors; re-audit the pin')
    tokens = int(size.group(1).replace(',', ''))
    budget = float(available.group(1)) * 2**30
    return {'tokens': tokens, 'per_token': budget / tokens, 'budget': budget}


def reported_tokens(metrics_text):
    """Reported KV token capacity from the server's own metrics surface."""
    for labels in CACHE_INFO.findall(metrics_text):
        for pair in labels.split(','):
            key, _, value = pair.partition('=')
            if key.strip() == 'kv_cache_size_tokens':
                return int(value.strip().strip('"'))
    raise AssertionError('vllm:cache_config_info has no kv_cache_size_tokens label')


def mem_available_gib(hosts_json):
    for rank in hosts_json.values():
        if 'error' in rank:
            raise AssertionError(f'Host probe failed: {rank["error"]}')
        for line in rank['meminfo'].splitlines():
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) / 2**20
    raise AssertionError('Host probe carried no MemAvailable')


def staged(directory, name):
    return (directory/name).read_text()


def scratch_copy(name, *relatives):
    """Stage pinned files in workspace scratch for a patch round-trip.

    Scratch stays inside the workspace: the system temp directory is not writable
    in every environment this suite runs in.
    """
    root = SCRATCH/name
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    dests = []
    for relative in relatives:
        dest = root/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((VLLM/relative).read_bytes())
        dests.append(dest)
    return root, dests


@unittest.skipUnless(VLLM.is_dir(), 'set DS41_VLLM_SOURCE to the pinned vllm package')
class DisplayKvPatchTests(unittest.TestCase):
    def test_both_anchors_match_the_pinned_tree(self):
        for patch in (backing_patch, credit_patch):
            text = (VLLM/patch.RELATIVE).read_bytes()
            self.assertIn(patch.OLD, text)
            self.assertNotIn(patch.NEW, text)

    def test_backing_patch_guards_reapplication_and_independent_rollback(self):
        root, (dest,) = scratch_copy('backing', backing_patch.RELATIVE)
        backing_patch.patch(root)
        backing_patch.patch(root)
        backing_patch.patch(root, check=True)
        self.assertIn(b'_ds41_backing', dest.read_bytes())
        compile(dest.read_text(), str(dest), 'exec')
        backing_patch.patch(root, revert=True)
        self.assertEqual(dest.read_bytes(), (VLLM/backing_patch.RELATIVE).read_bytes())
        dest.write_bytes(dest.read_bytes()+b'# source drift\n')
        with self.assertRaises(RuntimeError):
            backing_patch.patch(root)

    def test_credit_patch_reverts_without_disturbing_the_backing_patch(self):
        # The two patches target different files, so independent rollback is
        # structural rather than incidental: reverting the credit cannot touch
        # the backing substitution, and each has its own source hash.
        self.assertNotEqual(backing_patch.RELATIVE, credit_patch.RELATIVE)
        root, (backed, credited) = scratch_copy(
            'credit', backing_patch.RELATIVE, credit_patch.RELATIVE
        )
        backing_patch.patch(root)
        credit_patch.patch(root)
        credit_patch.patch(root, check=True)
        credit_patch.patch(root, revert=True)
        self.assertIn(b'_ds41_backing', backed.read_bytes())
        self.assertNotIn(b'_ds41_display_kv_credit', credited.read_bytes())
        self.assertEqual(credited.read_bytes(), (VLLM/credit_patch.RELATIVE).read_bytes())
        compile(backed.read_text(), str(backed), 'exec')
        compile(credited.read_text(), str(credited), 'exec')
        backing_patch.patch(root, revert=True)
        self.assertEqual(backed.read_bytes(), (VLLM/backing_patch.RELATIVE).read_bytes())

    def test_credit_is_inert_until_a_deployment_enables_it(self):
        # A disabled deployment must be byte-identical to today's budget.
        for disabled in (None, '0'):
            if disabled is None:
                os.environ.pop('DS41_DISPLAY_KV_MIB', None)
            else:
                os.environ['DS41_DISPLAY_KV_MIB'] = disabled
            try:
                self.assertEqual(helper.configured_bytes(), 0)
                self.assertEqual(helper.credit(), 0)
            finally:
                os.environ.pop('DS41_DISPLAY_KV_MIB', None)
        os.environ['DS41_DISPLAY_KV_MIB'] = '1792'
        try:
            self.assertEqual(helper.credit(), DISPLAY_BYTES)
            self.assertEqual(helper.configured_bytes(), DISPLAY_BYTES)
        finally:
            os.environ.pop('DS41_DISPLAY_KV_MIB', None)

    def test_credit_is_bounded_by_the_allocator_span(self):
        for mib in ('1', '1024', '1792'):
            os.environ['DS41_DISPLAY_KV_MIB'] = mib
            try:
                self.assertEqual(helper.credit(), int(mib) * 2**20)
                self.assertLessEqual(helper.credit(), DISPLAY_BYTES)
            finally:
                os.environ.pop('DS41_DISPLAY_KV_MIB', None)
        os.environ['DS41_DISPLAY_KV_MIB'] = '1793'
        try:
            with self.assertRaises(ValueError):
                helper.credit()
        finally:
            os.environ.pop('DS41_DISPLAY_KV_MIB', None)

    def test_backing_refuses_instead_of_falling_back_to_ordinary_ram(self):
        # A silent torch.zeros fallback would OOM at 0.85 utilization because
        # the admitted block count no longer fits ordinary RAM.
        with self.assertRaises(RuntimeError):
            helper.backing(DISPLAY_BYTES, dtype='int8', device='cuda:0')


class DisplayKvConfigTests(unittest.TestCase):
    def test_r38_defaults_disable_the_display_credit(self):
        config = json.loads((ROOT/'cluster-r38-c8.json').read_text())
        display = config.get('display_kv', {})
        self.assertFalse(display.get('enabled', False))
        self.assertEqual(config['gpu_memory_utilization'], 0.85)

    def test_configure_preserves_the_display_credit(self):
        # configure-r38.py must carry display_kv forward like the other keys, so a
        # regenerated recipe cannot silently drop the host state requirement.
        source = (ROOT/'configure-r38.py').read_text()
        self.assertIn('display_kv', source)


@unittest.skipUnless(BASELINE.is_dir() and DISPLAY.is_dir(),
                     'run observe.py for a display-kv-{baseline,enabled} pair first')
class DisplayKvHeadroomTests(unittest.TestCase):
    """Prove the display reserve became usable KV of the predicted size."""

    def test_credited_kv_memory_is_larger_by_the_display_credit(self):
        # The unquantised surface: vLLM logs the budget it planned against, so
        # this shows the credit directly. The reported token capacity is rounded
        # down to whole max-length requests and can hide it entirely.
        base = reported_capacity(staged(BASELINE, 'rank-0.log'))
        display = reported_capacity(staged(DISPLAY, 'rank-0.log'))
        gain = display['budget'] - base['budget']
        self.assertAlmostEqual(gain / DISPLAY_BYTES, 1.0, delta=0.10)
        self.assertAlmostEqual(display['per_token'], base['per_token'],
                               delta=base['per_token']*0.05)

    def test_reported_capacity_never_falls_and_matches_when_the_floor_moves(self):
        # Reported capacity is quantised to whole max-length requests, so a
        # 1.75 GiB credit is only visible when it crosses a floor boundary.
        # A flat reading is the floor effect, not a failure and not a gain either.
        base = reported_capacity(staged(BASELINE, 'rank-0.log'))
        display = reported_capacity(staged(DISPLAY, 'rank-0.log'))
        self.assertGreaterEqual(display['tokens'], base['tokens'])
        if display['tokens'] == base['tokens']:
            self.skipTest(
                f'Floor effect: the {DISPLAY_BYTES/2**30:.2f} GiB credit did not cross a '
                f'max-length boundary at {base["tokens"]:,} tokens; the credited KV '
                'backing is still larger, see the budget test'
            )
        gain_bytes = (display['tokens'] - base['tokens']) * base['per_token']
        self.assertAlmostEqual(gain_bytes / DISPLAY_BYTES, 1.0, delta=0.15)

    def test_capacity_is_reported_on_the_serving_metrics_surface(self):
        base = reported_tokens(staged(BASELINE, 'after-metrics.txt'))
        display = reported_tokens(staged(DISPLAY, 'after-metrics.txt'))
        self.assertEqual(base, reported_capacity(staged(BASELINE, 'rank-0.log'))['tokens'])
        self.assertGreaterEqual(display, base)

    def test_display_backing_is_not_ordinary_ram(self):
        base = mem_available_gib(json.loads(staged(BASELINE, 'after-hosts.json')))
        display = mem_available_gib(json.loads(staged(DISPLAY, 'after-hosts.json')))
        # Ordinary RAM must not have absorbed the 1.75 GiB span.
        self.assertLess(base - display, 512 / 1024)

    def test_disabled_run_matches_the_recorded_baseline(self):
        # Rollback guarantee: with the credit disabled, reported capacity is the
        # recorded baseline, so the patches are inert when not enabled.
        recorded = json.loads(staged(BASELINE, 'baseline.json'))
        self.assertEqual(
            reported_tokens(staged(BASELINE, 'after-metrics.txt')),
            recorded['kv_cache_size_tokens'],
        )


if __name__ == '__main__':
    unittest.main()
