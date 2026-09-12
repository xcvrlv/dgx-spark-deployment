"""Build-time native imports, plus optional single-Spark CUDA capability check."""
import argparse
import platform
import subprocess
from importlib.metadata import version

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', action='store_true')
args = parser.parse_args()
assert platform.machine() == 'aarch64', platform.machine()
import torch
import vllm._C_stable_libtorch
from b12x.comm import roce
from b12x.comm.roce._proxy import load
from b12x.loader._native import load as load_storage
from vllm.distributed.device_communicators.b12x_roce_all_reduce import REQUIRED_B12X_ROCE_API_VERSION

assert roce.API_VERSION == REQUIRED_B12X_ROCE_API_VERSION
assert version('nvidia-cutlass-dsl') == '4.6.2'
subprocess.run(['pkg-config', '--exists', 'liburing'], check=True)
print('RoCEnante proxy:', load()._name)
print('Engram native storage:', load_storage())
if args.gpu:
    from vllm.models.deepseek_v4_1 import DeepseekV41ForCausalLM
    assert torch.cuda.device_count() == 1
    assert torch.cuda.get_device_capability() == (12, 1)
    x = torch.arange(128, device='cuda', dtype=torch.float32)
    assert (x + 1).sum().item() == 8256
    torch.cuda.synchronize()
print('Image import/ABI checks passed', torch.__version__, torch.version.cuda)
