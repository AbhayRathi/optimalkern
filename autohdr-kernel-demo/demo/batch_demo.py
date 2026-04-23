"""
batch_demo.py — Component 4

Demonstrates the throughput ceiling problem at AutoHDR scale.
Compares sequential vs batched processing across different batch sizes.

Run on a GPU:
    python demo/batch_demo.py
"""

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Device selection with CPU fallback
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    print("[WARNING] CUDA not available — running on CPU. "
          "Timings will not reflect production GPU performance.")


# ---------------------------------------------------------------------------
# Per-image processing (fused, no sharpen for batch compat)
# ---------------------------------------------------------------------------

def _process_single(img: torch.Tensor, warmth: float = 0.05, sat: float = 1.15) -> torch.Tensor:
    """Fused tone-map + color-grade on a single [3, H, W] tensor."""
    r, g, b = img[0], img[1], img[2]

    # Reinhard
    r = r / (1.0 + r)
    g = g / (1.0 + g)
    b = b / (1.0 + b)

    # Warmth shift
    r = torch.clamp(r + warmth, 0.0, 1.0)
    b = torch.clamp(b - warmth * 0.5, 0.0, 1.0)

    # Saturation boost
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    r = torch.clamp(lum + sat * (r - lum), 0.0, 1.0)
    g = torch.clamp(lum + sat * (g - lum), 0.0, 1.0)
    b = torch.clamp(lum + sat * (b - lum), 0.0, 1.0)

    return torch.stack([r, g, b], dim=0)


def _process_batch(batch: torch.Tensor, warmth: float = 0.05, sat: float = 1.15) -> torch.Tensor:
    """Fused tone-map + color-grade on a batched [N, 3, H, W] tensor."""
    r = batch[:, 0, :, :]
    g = batch[:, 1, :, :]
    b = batch[:, 2, :, :]

    # Reinhard
    r = r / (1.0 + r)
    g = g / (1.0 + g)
    b = b / (1.0 + b)

    # Warmth shift
    r = torch.clamp(r + warmth, 0.0, 1.0)
    b = torch.clamp(b - warmth * 0.5, 0.0, 1.0)

    # Saturation boost
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    r = torch.clamp(lum + sat * (r - lum), 0.0, 1.0)
    g = torch.clamp(lum + sat * (g - lum), 0.0, 1.0)
    b = torch.clamp(lum + sat * (b - lum), 0.0, 1.0)

    return torch.stack([r, g, b], dim=1)  # [N, 3, H, W]


# ---------------------------------------------------------------------------
# Sequential vs batched API
# ---------------------------------------------------------------------------

def sequential_process(images: list[torch.Tensor]) -> tuple[list[torch.Tensor], float]:
    """
    Process each [3, H, W] tensor one at a time.
    Returns (results, total_ms).
    """
    if images[0].is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    results = [_process_single(img) for img in images]
    if images[0].is_cuda:
        torch.cuda.synchronize()
    total_ms = (time.perf_counter() - t0) * 1000.0
    return results, total_ms


def batched_process(images: list[torch.Tensor]) -> tuple[list[torch.Tensor], float]:
    """
    Stack into [N, 3, H, W], process the entire batch simultaneously, then unstack.
    Returns (results, total_ms).
    """
    batch = torch.stack(images, dim=0)  # [N, 3, H, W]
    if batch.is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out_batch = _process_batch(batch)
    if batch.is_cuda:
        torch.cuda.synchronize()
    total_ms = (time.perf_counter() - t0) * 1000.0
    results = [out_batch[i] for i in range(out_batch.shape[0])]
    return results, total_ms


# torch.compile version of the batched pipeline
try:
    compiled_batch = torch.compile(batched_process)
except (RuntimeError, AttributeError, ImportError) as exc:
    print(f"[WARNING] torch.compile not available ({exc}); using eager batched_process.")
    compiled_batch = batched_process


# ---------------------------------------------------------------------------
# Main — sweep over batch sizes
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    BATCH_SIZES = [1, 4, 8, 16, 32]
    N_WARMUP = 3
    N_RUNS = 10  # repetitions per batch size for stable timing

    print(f"Running on: {DEVICE.upper()}")
    print(f"Photo resolution: 1080 × 1920 (FHD, float32)\n")

    results_data = []

    for n in BATCH_SIZES:
        images = [torch.rand(3, 1080, 1920, device=DEVICE) for _ in range(n)]

        # Warmup
        for _ in range(N_WARMUP):
            sequential_process(images)
            batched_process(images)

        # Timed runs
        seq_times = []
        bat_times = []
        for _ in range(N_RUNS):
            _, t_seq = sequential_process(images)
            _, t_bat = batched_process(images)
            seq_times.append(t_seq)
            bat_times.append(t_bat)

        avg_seq = sum(seq_times) / N_RUNS
        avg_bat = sum(bat_times) / N_RUNS
        throughput_gain = avg_seq / avg_bat if avg_bat > 0 else float("inf")

        results_data.append({
            "n_photos": n,
            "sequential_ms": round(avg_seq, 2),
            "batched_ms": round(avg_bat, 2),
            "per_photo_sequential_ms": round(avg_seq / n, 2),
            "per_photo_batched_ms": round(avg_bat / n, 2),
            "throughput_gain": round(throughput_gain, 2),
        })

    # Print table
    header = (
        f"{'N Photos':<10} {'Sequential (ms)':<18} {'Batched (ms)':<15} "
        f"{'Per-photo Seq':<16} {'Per-photo Bat':<16} {'Throughput Gain'}"
    )
    sep = "-" * 95
    print(header)
    print(sep)
    for r in results_data:
        print(
            f"{r['n_photos']:<10} {r['sequential_ms']:<18.2f} {r['batched_ms']:<15.2f} "
            f"{r['per_photo_sequential_ms']:<16.2f} {r['per_photo_batched_ms']:<16.2f} "
            f"{r['throughput_gain']:.2f}x"
        )

    # Save for Streamlit
    out_path = Path(__file__).parent / "batch_results.json"
    out_path.write_text(json.dumps(results_data, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out_path}")
