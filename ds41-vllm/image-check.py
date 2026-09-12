"""Driver-free build checks, plus required post-build single-Spark GPU checks."""
import argparse
import platform
import subprocess
from importlib.metadata import distribution, version
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', action='store_true')
args = parser.parse_args()
assert platform.machine() == 'aarch64', platform.machine()
from b12x.comm.roce._proxy import load
from b12x.loader._native import _build as build_storage

# Docker build has no host libcuda. Check the wheel contents without importing
# vLLM (including adapters that can transitively import its CUDA extensions).
package = Path(distribution('vllm').locate_file('vllm'))
assert list(package.glob('_C_stable_libtorch*.so')), package
assert version('nvidia-cutlass-dsl') == '4.6.2'
subprocess.run(['pkg-config', '--exists', 'liburing'], check=True)
print('RoCEnante proxy:', load()._name)
print('Compiled Engram native storage:', build_storage())
if args.gpu:
    import torch
    import vllm._C_stable_libtorch
    from b12x.comm import roce
    from b12x.loader._native import load as load_storage
    from vllm.distributed.device_communicators.b12x_roce_all_reduce import REQUIRED_B12X_ROCE_API_VERSION
    from vllm.models.deepseek_v4_1 import DeepseekV41ForCausalLM
    assert roce.API_VERSION == REQUIRED_B12X_ROCE_API_VERSION
    print('Loaded Engram native storage:', load_storage())
    assert torch.cuda.device_count() == 1
    assert torch.cuda.get_device_capability() == (12, 1)
    x = torch.arange(128, device='cuda', dtype=torch.float32)
    assert (x + 1).sum().item() == 8256
    torch.cuda.synchronize()
    print('GPU import/ABI checks passed', torch.__version__, torch.version.cuda)
else:
    print('Driver-free build checks passed; CUDA imports and ABI checks require --gpu.')
