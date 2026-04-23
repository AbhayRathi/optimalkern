"""
helion_kernel.py — Component 2

Helion-based fused GPU kernel for real estate photo editing.
Compares against all three baseline methods from baseline.py.

Run on a GPU:
    python demo/helion_kernel.py

Requires: pip install helion
"""

import time
import sys
import torch

# ---------------------------------------------------------------------------
# Device selection with CPU fallback
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    print("[WARNING] CUDA not available — Helion kernel requires CUDA. "
          "Falling back to PyTorch fused implementation.")

# ---------------------------------------------------------------------------
# Try to import Helion; fall back gracefully if unavailable
# ---------------------------------------------------------------------------
HELION_AVAILABLE = False
try:
    import helion
    import helion.language as hl
    HELION_AVAILABLE = True
except ImportError:
    print("[WARNING] helion not installed. Run: pip install helion")
    print("[INFO] Using torch.compile fallback for helion_kernel.")


# ---------------------------------------------------------------------------
# Helion kernel (GPU only)
# ---------------------------------------------------------------------------

if HELION_AVAILABLE and DEVICE == "cuda":
    @helion.kernel(
        config=helion.Config(
            block_sizes=[64, 64],
        )
    )
    def fused_edit_kernel(
        x: torch.Tensor,
        warmth: float = 0.05,
        sat: float = 1.15,
    ) -> torch.Tensor:
        """
        Fused elementwise kernel tiling over [H, W]:
          1. Reinhard tone mapping: ch / (1 + ch)
          2. Warmth shift:         r += warmth, b -= warmth * 0.5, clamp [0,1]
          3. Saturation boost:     lum = 0.299r + 0.587g + 0.114b,
                                   ch = clamp(lum + sat*(ch - lum), 0, 1)
        Input:  float32 [3, H, W] with values in [0, 2.0]
        Output: float32 [3, H, W] with values in [0, 1.0]
        """
        _c, h, w = x.size()
        out = torch.empty_like(x)

        for tile_h, tile_w in hl.tile([h, w]):
            r = x[0, tile_h, tile_w]
            g = x[1, tile_h, tile_w]
            b = x[2, tile_h, tile_w]

            # 1. Reinhard tone mapping
            r = r / (1.0 + r)
            g = g / (1.0 + g)
            b = b / (1.0 + b)

            # 2. Warmth shift
            r = torch.clamp(r + warmth, 0.0, 1.0)
            b = torch.clamp(b - warmth * 0.5, 0.0, 1.0)

            # 3. Saturation boost
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            r = torch.clamp(lum + sat * (r - lum), 0.0, 1.0)
            g = torch.clamp(lum + sat * (g - lum), 0.0, 1.0)
            b = torch.clamp(lum + sat * (b - lum), 0.0, 1.0)

            out[0, tile_h, tile_w] = r
            out[1, tile_h, tile_w] = g
            out[2, tile_h, tile_w] = b

        return out

else:
    # ---------------------------------------------------------------------------
    # PyTorch fallback when Helion is unavailable or no CUDA
    # ---------------------------------------------------------------------------

    def fused_edit_kernel(
        x: torch.Tensor,
        warmth: float = 0.05,
        sat: float = 1.15,
    ) -> torch.Tensor:
        """PyTorch fallback (no Helion): same math, no custom tiling."""
        r, g, b = x[0], x[1], x[2]

        # Reinhard
        r = r / (1.0 + r)
        g = g / (1.0 + g)
        b = b / (1.0 + b)

        # Warmth
        r = torch.clamp(r + warmth, 0.0, 1.0)
        b = torch.clamp(b - warmth * 0.5, 0.0, 1.0)

        # Saturation
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        r = torch.clamp(lum + sat * (r - lum), 0.0, 1.0)
        g = torch.clamp(lum + sat * (g - lum), 0.0, 1.0)
        b = torch.clamp(lum + sat * (b - lum), 0.0, 1.0)

        return torch.stack([r, g, b], dim=0)


# ---------------------------------------------------------------------------
# Benchmarking helper (local copy — same logic as baseline.py)
# ---------------------------------------------------------------------------

def _benchmark(fn, x: torch.Tensor, n: int = 200) -> float:
    for _ in range(10):
        _ = fn(x)
    if x.is_cuda:
        torch.cuda.synchronize()

    t = time.perf_counter()
    for _ in range(n):
        _ = fn(x)
    if x.is_cuda:
        torch.cuda.synchronize()

    return (time.perf_counter() - t) / n * 1000.0


# ---------------------------------------------------------------------------
# Main — 4-method comparison table
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Import baseline pipelines
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from baseline import (
        pipeline_naive,
        pipeline_fused,
        pipeline_compiled,
        benchmark,
    )

    print(f"Running on: {DEVICE.upper()}")
    if HELION_AVAILABLE and DEVICE == "cuda":
        print("Helion kernel: ACTIVE (compiled Triton kernel)")
    else:
        print("Helion kernel: FALLBACK (PyTorch implementation)")
    print(f"Input tensor: 3 × 2160 × 3840 (4K, float32)\n")

    x = torch.rand(3, 2160, 3840, device=DEVICE)

    print("Benchmarking all four methods…")
    naive_ms = benchmark(pipeline_naive, x)
    fused_ms = benchmark(pipeline_fused, x)
    compiled_ms = benchmark(pipeline_compiled, x)
    helion_ms = _benchmark(fused_edit_kernel, x)

    header = f"{'Method':<30} {'Time (ms)':<15} {'Speedup vs Naive'}"
    sep = "-" * 62
    print(f"\n{header}")
    print(sep)
    print(f"{'Naive PyTorch':<30} {naive_ms:<15.2f} 1.00x")
    print(f"{'Fused (manual)':<30} {fused_ms:<15.2f} {naive_ms / fused_ms:.2f}x")
    print(f"{'torch.compile':<30} {compiled_ms:<15.2f} {naive_ms / compiled_ms:.2f}x")

    label = "Helion kernel" if (HELION_AVAILABLE and DEVICE == "cuda") else "Helion (PyTorch fallback)"
    print(f"{label:<30} {helion_ms:<15.2f} {naive_ms / helion_ms:.2f}x")

    # Save benchmark results for the Streamlit app
    import json
    from pathlib import Path

    bench_out = [
        {"method": "Naive PyTorch",  "time_ms": round(naive_ms, 2),    "speedup": 1.0},
        {"method": "Fused (manual)", "time_ms": round(fused_ms, 2),    "speedup": round(naive_ms / fused_ms, 2)},
        {"method": "torch.compile",  "time_ms": round(compiled_ms, 2), "speedup": round(naive_ms / compiled_ms, 2)},
        {"method": label,            "time_ms": round(helion_ms, 2),   "speedup": round(naive_ms / helion_ms, 2)},
    ]
    bench_path = Path(__file__).parent / "bench_results.json"
    bench_path.write_text(json.dumps(bench_out, indent=2), encoding="utf-8")
    print(f"\nResults saved to {bench_path}")
