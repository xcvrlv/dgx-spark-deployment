"""Fleet policy applied by boot and checked after SGLang loads each model config."""
import functools
import json
import os
import shlex


def serving_args(extra, topk):
    if topk not in (512, 2048):
        raise ValueError('DSV41_INDEX_TOPK must be 512 or 2048')
    args = shlex.split(extra)
    # Refuse alternate config sources/backends rather than allow argparse's
    # last-value-wins behavior to undermine the fixed serving profile.
    reserved = ('--json-model-override-args', '--dsa-topk-backend',
                '--speculative-dsa-topk-backend', '--model-path',
                '--attention-backend', '--speculative-draft-model-path',
                '--tp', '--tp-size', '--tensor-parallel-size', '--ep-size',
                '--nnodes', '--node-rank', '--dist-init-addr')
    for arg in args:
        if any(arg == flag or arg.startswith(flag + '=') for flag in reserved):
            raise ValueError(f'{arg}: controlled by the fleet top-k profile')
    # V4.1's SGLang config normalizer flattens text_config before applying
    # model overrides. A top-level index_topk is therefore intentional.
    args += ['--json-model-override-args', json.dumps({'index_topk': topk}),
             '--dsa-topk-backend', 'sgl-kernel',
             '--speculative-dsa-topk-backend', 'sgl-kernel']
    return args


def install(module):
    original = module.ModelConfig.__init__

    @functools.wraps(original)
    def checked(self, *args, **kwargs):
        original(self, *args, **kwargs)
        expected = int(os.environ['DSV41_INDEX_TOPK'])
        actual = getattr(self.hf_text_config, 'index_topk', None)
        if actual != expected:
            raise RuntimeError(f'index_topk mismatch: requested {expected}, loaded {actual}')
        print(f'[topk-policy] rank={os.environ.get("NODE_RANK")} '
              f'draft={kwargs.get("is_draft_model", False)} index_topk={actual}', flush=True)

    module.ModelConfig.__init__ = checked
