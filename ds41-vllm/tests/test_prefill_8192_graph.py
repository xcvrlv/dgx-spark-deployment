import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fleet

spec = importlib.util.spec_from_file_location('prefill_8192_patch', ROOT / 'patches/prefill_8192_graph.py')
overlay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(overlay)
runner_spec = importlib.util.spec_from_file_location('prefill_guarded_runner', ROOT / 'run-autotune.py')
runner = importlib.util.module_from_spec(runner_spec)
runner_spec.loader.exec_module(runner)
dspark_spec = importlib.util.spec_from_file_location('existing_dspark_patch', ROOT / 'patches/dspark_prefill.py')
dspark = importlib.util.module_from_spec(dspark_spec)
dspark_spec.loader.exec_module(dspark)
SOURCE = ROOT / '.build/upstream/vllm-1794dcf18454900263e0c66711af8ea4a1283ac1/vllm'


class Prefill8192GraphTests(unittest.TestCase):
    def test_route_requires_all_runtime_steps_and_rank_zero_capture(self):
        records = {rank: {'environment': '1', 'command_budget': '8192',
                          'captured_8192': rank == 0, 'runtime_eligible_8192': True,
                          'serving_replay_8192': True}
                   for rank in range(4)}
        def read(config, rank, command, timeout):
            return json.dumps(records[rank])
        with patch.object(runner.fleet, 'remote', side_effect=read), \
                patch.object(runner, 'serving_batch_histogram', return_value={
                    'observations_above_4096_up_to_8192': 0,
                    'observations_above_8192': 5}):
            self.assertEqual(len(runner.batch_route_evidence({'prefill_8192_graph': True})['ranks']), 4)
            records[2]['runtime_eligible_8192'] = False
            with self.assertRaisesRegex(RuntimeError, 'rank 2'):
                runner.batch_route_evidence({'prefill_8192_graph': True})
            records[2]['runtime_eligible_8192'] = True
            records[2]['serving_replay_8192'] = False
            with self.assertRaisesRegex(RuntimeError, 'rank 2'):
                runner.batch_route_evidence({'prefill_8192_graph': True})
            records[2]['serving_replay_8192'] = True
            records[0]['captured_8192'] = False
            with self.assertRaisesRegex(RuntimeError, 'rank 0'):
                runner.batch_route_evidence({'prefill_8192_graph': True})

    def test_serving_histogram_records_aggregated_observations(self):
        text = '\n'.join(f'vllm:iteration_tokens_total_bucket{{engine="0",le="{ceiling}"}} {count}'
                         for ceiling, count in ((4096, 200), (8192, 248), ('+Inf', 248)))
        self.assertEqual(runner.parse_iteration_histogram(text), {
            'observations_above_4096_up_to_8192': 48,
            'observations_above_8192': 0})
        with self.assertRaisesRegex(RuntimeError, 'Missing'):
            runner.parse_iteration_histogram(text.splitlines()[0])

    @unittest.skipUnless((SOURCE / overlay.RELATIVE).is_file(), 'pinned source snapshot required')
    def test_hash_guard_roundtrip_flag_and_nonmatching_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / overlay.RELATIVE
            path.parent.mkdir(parents=True)
            original = (SOURCE / overlay.RELATIVE).read_bytes()
            path.write_bytes(original)
            runner_path = Path(directory) / overlay.RUNNER_RELATIVE
            runner_path.parent.mkdir(parents=True)
            existing_runner = dspark.transform(
                (SOURCE / overlay.RUNNER_RELATIVE).read_bytes(),
                dspark.PATCHES[overlay.RUNNER_RELATIVE][1])
            runner_path.write_bytes(existing_runner)
            overlay.patch(directory)
            overlay.patch(directory, check=True)
            overlay.patch(directory)
            patched = path.read_bytes()
            replay_tree = ast.parse(runner_path.read_bytes())
            replay_condition = next(node.test for node in ast.walk(replay_tree)
                                    if isinstance(node, ast.If) and any(
                                        isinstance(child, ast.Assign) and any(
                                            isinstance(target, ast.Attribute) and
                                            target.attr == '_ds41_8192_replay_reported'
                                            for target in child.targets)
                                        for child in node.body))
            replay_expression = compile(ast.Expression(replay_condition), '<serving replay>', 'eval')
            for dummy, tokens, expected in ((True, 8192, False), (False, 4096, False), (False, 8192, True)):
                self.assertEqual(eval(replay_expression, {
                    'dummy_run': dummy, 'batch_desc': NS(num_tokens=tokens),
                    'self': NS(model_state=NS(single_request_prefill_cudagraph_tokens=8192))}), expected)
            tree = ast.parse(patched)
            condition = next(node.test for node in ast.walk(tree)
                             if isinstance(node, ast.If)
                             and any(isinstance(child, ast.Assign) and
                                     any(isinstance(target, ast.Attribute) and
                                         target.attr == 'single_request_prefill_cudagraph_tokens'
                                         for target in child.targets)
                                     for child in node.body))
            expression = compile(ast.Expression(condition), '<graph condition>', 'eval')
            for tokens, flag, expected in ((4096, '0', True), (8192, '0', False),
                                           (8192, '1', True), (16384, '1', False)):
                with patch.dict(os.environ, {'DS41_PREFILL_8192_GRAPH': flag}):
                    scope = {'self': NS(max_num_tokens=tokens), 'os': os,
                             'parallel': NS(decode_context_parallel_size=1,
                                            pipeline_parallel_size=1, data_parallel_size=1),
                             'vllm_config': NS(lora_config=None),
                             'is_breakable_cudagraph_enabled': lambda: True}
                    self.assertEqual(eval(expression, scope), expected)
            overlay.patch(directory, revert=True)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(runner_path.read_bytes(), existing_runner)
            runner_path.write_bytes(existing_runner + b'\n# source drift\n')
            with self.assertRaises(RuntimeError):
                overlay.patch(directory)
            self.assertEqual(path.read_bytes(), original)
            runner_path.write_bytes(existing_runner)
            path.write_bytes(original + b'\n# source drift\n')
            with self.assertRaises(RuntimeError):
                overlay.patch(directory)

    def test_recipe_passes_8192_and_opt_in_to_all_ranks(self):
        import importlib.util
        builder_spec = importlib.util.spec_from_file_location('prefill_builder', ROOT / 'build-prefill-8192.py')
        builder = importlib.util.module_from_spec(builder_spec)
        builder_spec.loader.exec_module(builder)
        with tempfile.TemporaryDirectory() as directory:
            source = json.loads((ROOT / 'cluster-karmic-20260925.json').read_text())
            source.update(b12x_autotune=True, b12x_bounded_autotune=True,
                          b12x_compile_workers=1, b12x_preparation_trace=True,
                          b12x_hang_dump=True, max_num_batched_tokens=4096)
            path = Path(directory) / 'source.json'
            path.write_text(json.dumps(source))
            c = builder.candidate(path, 'candidate:8192')
            candidate = Path(directory) / 'candidate.json'
            candidate.write_text(json.dumps(c))
            fleet.load_config(candidate)
            for rank in range(4):
                self.assertEqual(fleet.environment(c, rank)['DS41_PREFILL_8192_GRAPH'], '1')
                args = fleet.serve_args(c, rank)
                self.assertEqual(args[args.index('--max-num-batched-tokens') + 1], '8192')
            c['max_num_batched_tokens'] = 4096
            candidate.write_text(json.dumps(c))
            with self.assertRaises(AssertionError):
                fleet.load_config(candidate)


if __name__ == '__main__':
    unittest.main()
