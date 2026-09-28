import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('performance_benchmark', ROOT/'benchmark.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class BenchmarkTests(unittest.TestCase):
    def test_streaming_retains_usage_text_and_sampling_parameters(self):
        events = [
            {'choices': [{'text': '', 'finish_reason': None}]},
            {'choices': [{'text': 'The answer', 'finish_reason': None}]},
            {'choices': [{'text': ' is 4.', 'finish_reason': 'length'}]},
            {'usage': {'prompt_tokens': 20, 'completion_tokens': 4}, 'choices': []},
        ]
        stream = b''.join(b'data: '+json.dumps(e).encode()+b'\n\n' for e in events)
        stream += b'data: [DONE]\n\n'
        with patch.object(benchmark.urllib.request, 'urlopen', return_value=io.BytesIO(stream)) as open_, \
                patch.object(benchmark.time, 'monotonic', side_effect=[10, 11, 13]):
            result = benchmark.measure({'nodes': [{'ip': '127.0.0.1'}], 'port': 1234},
                                       [1, 2], 4, temperature=1, top_p=0.95, seed=87)
        body = json.loads(open_.call_args.args[0].data)
        self.assertEqual((body['temperature'], body['top_p'], body['seed']), (1, 0.95, 87))
        self.assertEqual(result['text'], 'The answer is 4.')
        self.assertEqual(result['finish_reason'], 'length')
        self.assertEqual(result['ttft_seconds'], 1)
        self.assertEqual(result['decode_tokens_per_second'], 1.5)

    def test_http_only_run_never_contacts_nodes_and_never_repeats_real_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory)/'corpus.txt'
            corpus.write_text('A varied document about code, weather and architecture.')
            args = NS(config=ROOT/'cluster-karmic-c16.json', concurrency=1, requests=2,
                      input_tokens=32, max_tokens=4, temperature=1, top_p=0.95, seed=100,
                      prompt_file=corpus, http_only=True, output=Path(directory)/'results.json')
            prompts, draws = [], []

            def tokenize(config, url, payload):
                prompts.append(payload['prompt'])
                return {'tokens': list(range(64))}

            def measure(config, tokens, limit, **options):
                draws.append(options)
                return dict(ttft_seconds=0.1, completion_tokens=4, text='ok')

            with patch.object(benchmark.fleet, 'remote', side_effect=AssertionError('SSH forbidden')) as remote, \
                    patch.object(benchmark.fleet, 'request', side_effect=tokenize), \
                    patch.object(benchmark, 'measure', side_effect=measure), \
                    patch.object(benchmark.urllib.request, 'urlopen', side_effect=lambda *a, **k: io.BytesIO(b'counter 3\n')), \
                    patch('sys.stdout', new=io.StringIO()):
                report = benchmark.run(args)
            remote.assert_not_called()
            self.assertNotEqual(prompts[0], prompts[1])
            self.assertTrue(all(p.count(corpus.read_text()) == 1 for p in prompts))
            self.assertEqual([d['seed'] for d in draws], [100, 101])
            self.assertIsNone(report['image_id'])
            self.assertEqual(report['prompt_kind'], 'file')
            self.assertEqual(report['metrics_before'], 'counter 3\n')
            self.assertEqual(json.loads(args.output.read_text())['temperature'], 1)
            with patch.object(benchmark.fleet, 'request', return_value={'tokens': [1]}), \
                    patch.object(benchmark.fleet, 'remote') as remote:
                with self.assertRaisesRegex(AssertionError, 'fewer tokens'):
                    benchmark.run(args)
                remote.assert_not_called()


if __name__ == '__main__':
    unittest.main()
