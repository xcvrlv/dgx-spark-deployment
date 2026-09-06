"""Numerical probe for the actual mixed-K sigmoid implementation."""
import cuda.bindings.driver as cuda
import cutlass.cute as cute
from b12x.moe._shared.kernels.w4a16.kernel import W4A16FusedMoeKernel


class ActivationProbe(W4A16FusedMoeKernel):
    def __init__(self, size, fast):
        self.size, self.gb10_sigmoid, self.fast_math = size, fast, True

    @property
    def __cache_key__(self):
        return ("v15_activation_probe", self.size, self.gb10_sigmoid)

    @cute.jit
    def __call__(self, x, y, stream: cuda.CUstream):
        self.probe(x, y).launch(grid=((self.size + 127)//128, 1, 1),
                                block=(128, 1, 1), stream=stream)

    @cute.kernel
    def probe(self, x, y):
        idx = cute.arch.block_idx()[0] * 128 + cute.arch.thread_idx()[0]
        if idx < self.size:
            y[idx] = self._sigmoid_f32(x[idx])
