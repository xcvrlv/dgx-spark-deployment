"""Ensure missing recipes and failed builds cannot cascade into distribution."""
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('build_dspark_prefill', ROOT / 'build-dspark-prefill.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class BuildDsparkPrefillTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.source = self.directory / 'actual launch.json'
        self.output = self.directory / 'candidate.json'
        self.config = json.loads((ROOT / 'cluster-karmic-c16.json').read_text())
        self.config.update(image='operator:working-prefill', engram_resident_scales=True)
        self.source.write_text(json.dumps(self.config))
        self.quiet = patch('sys.stdout', new_callable=io.StringIO)
        self.quiet.start()
        self.addCleanup(self.quiet.stop)

    def test_missing_source_empty_image_and_conflicting_output_never_invoke_docker(self):
        with patch.object(m.subprocess, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'Source recipe does not exist'):
                m.build(self.directory / 'fleet.prefill-scales.json', self.output, share=True)
            for image in ('', ' '):
                self.source.write_text(json.dumps(dict(self.config, image=image)))
                with self.assertRaisesRegex(ValueError, 'no usable image'):
                    m.build(self.source, self.output, share=True)
            self.source.write_text(json.dumps(self.config))
            with self.assertRaisesRegex(ValueError, 'Source and output must differ'):
                m.build(self.source, self.source)
            self.output.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'different recipe'):
                m.build(self.source, self.output)
            run.assert_not_called()

    def test_failed_inspect_or_build_does_not_publish_or_share(self):
        original = self.source.read_bytes()
        for fail_at in (0, 1):
            with self.subTest(fail_at=fail_at):
                calls = []
                def run(args, **kwargs):
                    calls.append(args)
                    if len(calls) - 1 == fail_at:
                        raise subprocess.CalledProcessError(1, args)
                    return subprocess.CompletedProcess(args, 0)
                with patch.object(m.subprocess, 'run', side_effect=run):
                    with self.assertRaises(subprocess.CalledProcessError):
                        m.build(self.source, self.output, share=True)
                self.assertEqual(len(calls), fail_at + 1)
                self.assertFalse(self.output.exists())
                self.assertEqual(self.source.read_bytes(), original)

    def test_success_preserves_working_recipe_and_orders_build_before_share(self):
        before = self.source.read_bytes()
        calls = []
        def run(args, **kwargs):
            self.assertTrue(kwargs['check'])
            calls.append(args)
            if args[-1] == 'share':
                self.assertTrue(self.output.is_file())
            else:
                self.assertFalse(self.output.exists())
            return subprocess.CompletedProcess(args, 0)
        with patch.object(m.subprocess, 'run', side_effect=run):
            m.build(self.source, self.output, share=True)
        self.assertEqual(calls[0][-1], self.config['image'])
        self.assertEqual(calls[1][:2], ['docker', 'build'])
        self.assertIn('BASE_IMAGE=operator:working-prefill', calls[1])
        self.assertEqual(calls[2][-1], 'share')
        candidate = json.loads(self.output.read_text())
        self.assertEqual(candidate['model_path'], self.config['model_path'])
        self.assertTrue(candidate['engram_resident_scales'])
        self.assertTrue(candidate['dspark_skip_prefill_draft'])
        self.assertTrue(candidate['dspark_compact_context_graph'])
        self.assertEqual(self.source.read_bytes(), before)
        # An identical published recipe can be retried after distribution fails.
        with patch.object(m.subprocess, 'run') as run:
            m.build(self.source, self.output, share=True)
            self.assertEqual(run.call_count, 3)

    def test_discovery_ignores_nonrecipes_and_noninteractive_selection_never_guesses(self):
        (self.directory / 'not-a-recipe.json').write_text('{}')
        build_dir = self.directory / '.build'
        build_dir.mkdir()
        other = build_dir / 'working.json'
        other.write_text(json.dumps(self.config))
        ignored = self.directory / 'candidate.json'
        ignored.write_text(json.dumps(dict(self.config, image=m.DEFAULT_IMAGE)))
        candidates = m.discover(self.directory)
        self.assertEqual({path for path, _ in candidates}, {self.source, other})
        with patch.object(m, 'discover', return_value=candidates), patch.object(sys.stdin, 'isatty', return_value=False):
            with self.assertRaisesRegex(ValueError, 'Pass --from-config'):
                m.select_source()
        with patch.object(m, 'discover', return_value=candidates), patch.object(sys.stdin, 'isatty', return_value=True), patch('builtins.input', return_value='2'):
            self.assertEqual(m.select_source(), candidates[1][0])


if __name__ == '__main__':
    unittest.main()
