import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('build_tags', ROOT/'prepare-build-tags.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class BuildTagsTests(unittest.TestCase):
    def test_filters_only_artifact_refs_and_restores_without_changing_source(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            repo=root/'.build'/'source'
            repo.mkdir(parents=True)
            original_file=m.__file__
            m.__file__=str(root/'prepare-build-tags.py')
            def g(*args):
                return subprocess.check_output(['git','-C',str(repo),*args],text=True,stderr=subprocess.DEVNULL).strip()
            try:
                g('init')
                g('-c','user.name=Test','-c','user.email=test@example.invalid','commit','--allow-empty','-m','version')
                g('tag','v0.21.0')
                g('-c','user.name=Test','-c','user.email=test@example.invalid','commit','--allow-empty','-m','build')
                sha=g('rev-parse','HEAD')
                artifact='vllm-jovian-cu134-beta-'+sha
                g('tag',artifact)
                self.assertEqual(g('describe','--tags','--abbrev=0'),artifact)
                m.prepare(repo,sha)
                m.prepare(repo,sha)
                self.assertEqual(g('describe','--tags','--abbrev=0'),'v0.21.0')
                self.assertEqual(g('rev-parse','HEAD'),sha)
                self.assertEqual(g('status','--porcelain'),'')
                m.prepare(repo,sha,restore=True)
                self.assertEqual(g('describe','--tags','--abbrev=0'),artifact)
                with self.assertRaisesRegex(ValueError,'expected pin'):
                    m.prepare(repo,'0'*40)
                with self.assertRaisesRegex(ValueError,'disposable clones'):
                    m.prepare(root,sha)
            finally:
                m.__file__=original_file

if __name__=='__main__': unittest.main()
