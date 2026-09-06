#!/usr/bin/env python3
"""SM121 rotation and full K6-path checks against the retained native kernels."""
import argparse
import json
from pathlib import Path


def gpu():
    import torch
    import exllamav3_ext
    from b12x.moe._shared.kernels.w4a16 import gb10_rotations as rot
    from b12x.gemm import trellis_linear as api
    from smoke_r22_v11 import check_gpu
    inherited = check_gpu()
    assert torch.cuda.get_device_capability() == (12, 1)
    torch.manual_seed(5313)
    cases = 0
    for rows, cols in ((1, 128), (1, 6144), (4, 6144), (32, 512), (128, 6144), (4096, 6144)):
        for dtype in (torch.float16, torch.bfloat16):
            x = torch.randn(rows, cols, device="cuda", dtype=dtype) * 0.25
            scales = (torch.randn(cols, device="cuda") * 0.5).half()
            temp = torch.empty_like(x, dtype=torch.float16)
            actual = torch.empty_like(x)
            for pre in (True, False):
                exllamav3_ext.had_r_128(x.half(), temp, scales if pre else None,
                                       None if pre else scales, 1.0)
                expected = temp.to(dtype)
                rot.rotate(x, scales, actual, pre=pre)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                cases += 1
    timings = []
    original = rot.ENABLED
    try:
        for rows, k, n in ((1, 6144, 512), (4, 6144, 512), (32, 512, 1536), (128, 512, 512)):
            payload = torch.randint(-32768, 32768, (k // 16, n // 16, 96), device="cuda", dtype=torch.int16)
            suh = (torch.randn(k, device="cuda") * 0.1).half()
            svh = (torch.randn(n, device="cuda") * 0.1).half()
            weight = api.prepare_weight(payload, suh, svh, codebook="mcg", params_dtype=torch.float16)
            x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16) * 0.1
            old_x = torch.empty_like(x, dtype=torch.float16)
            old_out = torch.empty(rows, n, device="cuda", dtype=torch.float16)
            new_out = torch.empty(rows, n, device="cuda", dtype=torch.bfloat16)
            expected = torch.empty_like(new_out)
            padded = max(((rows + 47) // 48) * 48, ((rows + 63) // 64) * 64)
            def buffers():
                return dict(gemm_output=torch.empty_like(old_out),
                            c_tmp=torch.empty(n * padded, device="cuda", dtype=torch.float32),
                            rotated_f16=torch.empty_like(old_x))
            old_buffers, new_buffers = buffers(), buffers()
            def legacy():
                rot.ENABLED = False
                old_x.copy_(x)
                api.run(old_x, weight, output=old_out, **old_buffers)
                expected.copy_(old_out)
            def fused():
                rot.ENABLED = True
                api.run(x, weight, output=new_out, **new_buffers)
            legacy()
            fused()
            torch.cuda.synchronize()
            torch.testing.assert_close(new_out, expected, rtol=0, atol=0)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                fused()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                fused()
            for _ in range(3):
                x.normal_(0, 0.1)
                legacy()
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(new_out, expected, rtol=0, atol=0)
            def ms(fn):
                for _ in range(3):
                    fn()
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                for _ in range(20):
                    fn()
                b.record()
                b.synchronize()
                return a.elapsed_time(b) / 20
            timings.append(dict(m=rows, k=k, n=n, legacy_ms=ms(legacy), v13_ms=ms(fused)))
    finally:
        rot.ENABLED = original
    return dict(rotation_cases=cases, dense_k6_timings=timings, inherited_v11=inherited)


def main():
    import b12x
    import vllm
    from patch_r22_v13 import patch, VERSION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    patch(Path(b12x.__file__).parent, Path(vllm.__file__).parent, check=True)
    result = dict(overlay=VERSION)
    if args.gpu:
        result.update(gpu())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
