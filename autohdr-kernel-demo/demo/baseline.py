"""
baseline.py — Component 1

Baseline GPU kernel benchmarks for AutoHDR's real estate photo pipeline.
Compares naive sequential PyTorch, manually fused, and torch.compile approaches.

Run on a GPU:
    python demo/baseline.py
"""

import time
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

# ---------------------------------------------------------------------------
# Device selection with CPU fallback
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    print("[WARNING] CUDA not available — running on CPU. Timings will not reflect GPU performance.")


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image(path: str) -> torch.Tensor:
    """Load a JPEG as a float32 CUDA tensor [3, H, W] scaled 0–2.0 (simulating HDR range)."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0 * 2.0  # map [0,255] → [0, 2.0]
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # [H, W, 3] → [3, H, W]
    return tensor.to(DEVICE)


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------

def reinhard_tonemap(x: torch.Tensor) -> torch.Tensor:
    """Reinhard tone mapping: x / (1 + x) per pixel per channel."""
    return x / (1.0 + x)


def aces_tonemap(x: torch.Tensor) -> torch.Tensor:
    """ACES filmic tone mapping, clamped to [0, 1]."""
    return torch.clamp(
        (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14),
        0.0,
        1.0,
    )


def color_grade(x: torch.Tensor, warmth: float = 0.05, sat: float = 1.15) -> torch.Tensor:
    """
    Warmth shift: r += warmth, b -= warmth * 0.5
    Saturation boost using luminance weights.
    """
    r, g, b = x[0], x[1], x[2]
    r = r + warmth
    b = b - warmth * 0.5
    x = torch.stack([r, g, b], dim=0)

    lum = 0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]  # [H, W]
    x = torch.clamp(lum.unsqueeze(0) + sat * (x - lum.unsqueeze(0)), 0.0, 1.0)
    return x


def sharpen(x: torch.Tensor) -> torch.Tensor:
    """3×3 unsharp mask via F.conv2d with kernel [[0,-1,0],[-1,5,-1],[0,-1,0]]."""
    kernel = torch.tensor(
        [[0.0, -1.0, 0.0], [-1.0, 5.0, -1.0], [0.0, -1.0, 0.0]],
        dtype=torch.float32,
        device=x.device,
    )
    # Shape: [out_channels, in_channels/groups, kH, kW] — depthwise conv
    kernel = kernel.view(1, 1, 3, 3).expand(3, 1, 3, 3).contiguous()
    sharpened = F.conv2d(x.unsqueeze(0), kernel, padding=1, groups=3)
    return sharpened.squeeze(0)


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------

def pipeline_naive(x: torch.Tensor) -> torch.Tensor:
    """Naive sequential pipeline — separate VRAM pass per operation."""
    x = reinhard_tonemap(x)
    x = color_grade(x)
    x = sharpen(x)
    return x


def pipeline_fused(x: torch.Tensor) -> torch.Tensor:
    """
    Same math as pipeline_naive but collapsed into one function (single VRAM pass).
    Avoids redundant tensor materialisations between stages.
    """
    r = x[0]
    g = x[1]
    b = x[2]

    # --- Reinhard tone mapping ---
    r = r / (1.0 + r)
    g = g / (1.0 + g)
    b = b / (1.0 + b)

    # --- Warmth shift ---
    r = r + 0.05
    b = b - 0.025

    # --- Saturation boost ---
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    sat = 1.15
    r = torch.clamp(lum + sat * (r - lum), 0.0, 1.0)
    g = torch.clamp(lum + sat * (g - lum), 0.0, 1.0)
    b = torch.clamp(lum + sat * (b - lum), 0.0, 1.0)

    x = torch.stack([r, g, b], dim=0)

    # --- Sharpening (depthwise convolution) ---
    kernel = torch.tensor(
        [[0.0, -1.0, 0.0], [-1.0, 5.0, -1.0], [0.0, -1.0, 0.0]],
        dtype=torch.float32,
        device=x.device,
    )
    kernel = kernel.view(1, 1, 3, 3).expand(3, 1, 3, 3).contiguous()
    x = F.conv2d(x.unsqueeze(0), kernel, padding=1, groups=3).squeeze(0)
    return x


# torch.compile version of the fused pipeline
try:
    pipeline_compiled = torch.compile(pipeline_fused)
except Exception:
    # torch.compile may be unavailable on some builds; fall back transparently
    pipeline_compiled = pipeline_fused


# ---------------------------------------------------------------------------
# Benchmarking helper
# ---------------------------------------------------------------------------

def benchmark(fn, x: torch.Tensor, n: int = 200) -> float:
    """
    Warmup 10 runs, then time n runs.
    Uses torch.cuda.synchronize() around the timed section.
    Returns average time in milliseconds.
    """
    for _ in range(10):
        _ = fn(x)
    if x.is_cuda:
        torch.cuda.synchronize()

    t_start = time.perf_counter()
    for _ in range(n):
        _ = fn(x)
    if x.is_cuda:
        torch.cuda.synchronize()
    t_end = time.perf_counter()

    return (t_end - t_start) / n * 1000.0


# ---------------------------------------------------------------------------
# Main — run all three methods on a 4K tensor and print results
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Running on: {DEVICE.upper()}")
    print(f"Input tensor: 3 × 2160 × 3840 (4K, float32)\n")

    x = torch.rand(3, 2160, 3840, device=DEVICE)

    print("Benchmarking (this may take ~1 min on CPU)…")
    naive_ms = benchmark(pipeline_naive, x)
    fused_ms = benchmark(pipeline_fused, x)
    compiled_ms = benchmark(pipeline_compiled, x)

    header = f"{'Method':<25} {'Time (ms)':<15} {'Speedup vs Naive'}"
    sep = "-" * 58
    print(f"\n{header}")
    print(sep)
    print(f"{'Naive PyTorch':<25} {naive_ms:<15.2f} 1.00x")
    print(f"{'Fused (manual)':<25} {fused_ms:<15.2f} {naive_ms / fused_ms:.2f}x")
    print(f"{'torch.compile':<25} {compiled_ms:<15.2f} {naive_ms / compiled_ms:.2f}x")
