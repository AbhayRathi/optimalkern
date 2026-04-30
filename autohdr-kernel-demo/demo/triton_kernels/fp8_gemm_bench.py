"""
Real FP8 GEMM benchmark using torch._scaled_mm on H100 (SM90 required).
Run this directly on an H100 instance to replace the projected numbers in fp8_demo.py.
"""
import torch
import json
from pathlib import Path


def benchmark_fp8_vs_fp16(M: int = 4096, N: int = 4096, K: int = 4096) -> dict:
    assert torch.cuda.is_available(), "CUDA required"
    cap = torch.cuda.get_device_capability()
    assert cap[0] >= 9, f"H100 (SM90) required, got SM{cap[0]}{cap[1]}"

    def _bench(fn, warmup=10, reps=100):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(reps):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / reps

    a16 = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b16 = torch.randn(K, N, device="cuda", dtype=torch.float16)
    fp16_ms = _bench(lambda: torch.mm(a16, b16))

    a8  = a16.to(torch.float8_e4m3fn)
    b8  = b16.to(torch.float8_e4m3fn)
    s   = torch.ones(1, device="cuda", dtype=torch.float32)
    fp8_ms = _bench(lambda: torch._scaled_mm(a8, b8, s, s, out_dtype=torch.float16))

    result = {
        "M": M, "N": N, "K": K,
        "fp16_ms": round(fp16_ms, 4),
        "fp8_ms":  round(fp8_ms, 4),
        "speedup": round(fp16_ms / fp8_ms, 3),
        "measured": True,
        "device": torch.cuda.get_device_name(),
    }
    print(json.dumps(result, indent=2))
    Path("fp8_gemm_results.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    benchmark_fp8_vs_fp16()
