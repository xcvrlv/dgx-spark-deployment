"""Compile the actual legacy/new cooperative-grid input-rotation methods."""
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32
from b12x.moe._shared.kernels.w4a16.kernel import W4A16FusedMoeKernel


class RotationProbe(W4A16FusedMoeKernel):
    def __init__(self, rows, hidden, topk, shared, broadcast=True):
        self.rows, self.hidden_size, self.top_k = rows, hidden, topk
        self.gb10_shared_input = shared
        self.broadcast_suh = broadcast
        self.direct_topk_routes = False
        self.moe_block_size = 8
        self.cta_threads = 128

    @property
    def __cache_key__(self):
        return ("v14_rotation_probe", self.rows, self.hidden_size, self.top_k,
                self.gb10_shared_input, self.broadcast_suh)

    @cute.jit
    def __call__(self, x, gate, up, sg, su, routes, experts, count, mapping,
                 stream: cuda.CUstream):
        self.probe(x, gate, up, sg, su, routes, experts, count, mapping).launch(
            grid=(96, 1, 1), block=(128, 1, 1), stream=stream)

    @cute.kernel
    def probe(self, x, gate, up, sg, su, routes, experts, count, mapping):
        tid = cute.arch.thread_idx()[0]
        cta = cute.arch.block_idx()[0]
        self._run_input_rotation(x, gate, up, sg, su, routes, experts, count, mapping,
                                 Int32(2), Int32(2), tid, cta, Int32(96), Int32(self.rows))
