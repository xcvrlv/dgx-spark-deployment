#!/usr/bin/env python3
"""Validate v11 hashes and compare fused stores with FP32 + torch conversion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def check_gpu():
    import torch
    import cutlass
    import cutlass.cute as cute
    from cutlass.cutlass_dsl import Int32
    from b12x._lib.utils import current_cuda_stream, make_ptr
    from b12x.moe._shared.kernels.w4a16.kernel import compile_w4a16_topk_sum

    assert torch.cuda.get_device_capability() == (12, 1)
    torch.manual_seed(711)
    hidden, topk, experts = 6144, 8, 6
    dtype_names = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    pointer_types = {torch.float32: cutlass.Float32, torch.bfloat16: cutlass.BFloat16,
                     torch.float16: cutlass.Float16, torch.int32: cutlass.Int32,
                     torch.int64: cutlass.Int64}

    def ptr(t):
        return make_ptr(pointer_types[t.dtype], t.data_ptr(), cute.AddressSpace.gmem,
                        assumed_align=4 if t.dtype == torch.int32 else 8 if t.dtype == torch.int64 else 16)

    cases = 0
    for m in (1, 4, 32, 128, 2048):
        for shared in (False, True):
            # Same reduction receives either two- or three-tier projection
            # results; use permuted global ids and missing routes to exercise
            # the combined-expert map, including stale FC2 entries.
            ids_dtype = torch.int64 if shared else torch.int32
            values = torch.randn(m * topk, hidden, device="cuda", dtype=torch.float16)
            ids = torch.randint(-1, experts + 2, (m, topk), device="cuda", dtype=ids_dtype)
            mapping = torch.tensor([4, 1, 5, 0, 3, 2, -1, -1, -1], device="cuda", dtype=torch.int32)
            weights = torch.rand(m, topk, device="cuda", dtype=torch.float32)
            weights /= weights.sum(dim=-1, keepdim=True)
            svh = torch.randn(1 if shared else experts, hidden, device="cuda", dtype=torch.float16)
            outputs = {name: torch.empty(m, hidden, device="cuda", dtype=dtype)
                       for name, dtype in dtype_names.items()}
            launches = {name: compile_w4a16_topk_sum(
                m=m, topk=topk, hidden_size=hidden, element_dtype="fp16",
                output_element_dtype=name, full_rotation=True,
                num_experts=experts, route_num_experts=experts + 2,
                route_ids_dtype=ids_dtype, use_expert_map=True, broadcast_svh=shared,
            ) for name in outputs}
            assert len({id(x.compiled) for x in launches.values()}) == 3

            def run(name):
                launches[name].compiled(ptr(values), ptr(outputs[name]), ptr(weights),
                    ptr(ids), ptr(mapping), ptr(svh), Int32(experts), Int32(experts + 2),
                    m, current_cuda_stream())

            for name in outputs:
                run(name)
            torch.cuda.synchronize()
            for name in ("bf16", "fp16"):
                torch.testing.assert_close(outputs[name], outputs["fp32"].to(dtype_names[name]),
                                           rtol=0, atol=0)
            # Capture all three specializations, then change inputs each replay.
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for name in outputs:
                    run(name)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                for name in outputs:
                    run(name)
            for _ in range(3):
                values.normal_()
                torch.cuda.synchronize()
                graph.replay()
                torch.cuda.synchronize()
                for name in ("bf16", "fp16"):
                    torch.testing.assert_close(outputs[name], outputs["fp32"].to(dtype_names[name]),
                                               rtol=0, atol=0)
            cases += 1
    return {"fused_output_eager_and_graph_cases": cases}


def main():
    import b12x
    from patch_r22_v11 import patch, VERSION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--distributed", action="store_true",
                        help="Run four-node v10 DCP/RoCE graph tests against v11")
    args = parser.parse_args()
    patch(Path(b12x.__file__).parent, check=True)
    result = {"overlay": VERSION}
    if args.gpu:
        result.update(check_gpu())
    if args.distributed:
        from smoke_r22_performance import check_distributed_roce
        result.update(check_distributed_roce())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
