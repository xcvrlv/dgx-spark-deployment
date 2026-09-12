import importlib.util
import json
import os
from pathlib import Path
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('topk_policy', ROOT / 'upstream/adapter/topk_policy.py')
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


class PolicyTests(unittest.TestCase):
    def test_both_profiles_emit_explicit_override(self):
        for k in (512, 2048):
            args = policy.serving_args('--watchdog-timeout 1800', k)
            self.assertEqual(json.loads(args[args.index('--json-model-override-args') + 1]), {'index_topk': k})
            self.assertEqual(args[args.index('--dsa-topk-backend') + 1], 'sgl-kernel')

    def test_conflicting_overrides_rejected(self):
        for value in ('--json-model-override-args={"index_topk":512}',
                      '--dsa-topk-backend torch',
                      '--speculative-dsa-topk-backend=flashinfer',
                      '--model-path /other', '--attention-backend triton', '--tp 3'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                policy.serving_args(value, 2048)

    def test_quoted_extra_argument_survives(self):
        args = policy.serving_args('--some-label "two words"', 2048)
        self.assertEqual(args[:2], ['--some-label', 'two words'])

    def test_unsupported_topk_rejected(self):
        with self.assertRaises(ValueError):
            policy.serving_args('', 1024)

    def test_model_config_cannot_silently_lower_topk(self):
        class ModelConfig:
            def __init__(self, actual, **kwargs):
                self.hf_text_config = types.SimpleNamespace(index_topk=actual)
        module = types.SimpleNamespace(ModelConfig=ModelConfig)
        policy.install(module)
        with patch.dict(os.environ, {'DSV41_INDEX_TOPK': '2048'}):
            module.ModelConfig(2048)
            module.ModelConfig(2048, is_draft_model=True)
            for actual in (512, 1024, None):
                with self.assertRaises(RuntimeError):
                    module.ModelConfig(actual)


if __name__ == '__main__':
    unittest.main()
