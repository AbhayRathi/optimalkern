"""Layer 0 profiler demo for AutoHDR-style pipeline bottleneck detection."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_PATH = Path(__file__).parent / "profiler_results.json"


def reinhard_tonemap(x: torch.Tensor) -> torch.Tensor:
    return x / (1.0 + x)


def color_grade(x: torch.Tensor) -> torch.Tensor:
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    r = r + 0.05
    b = b - 0.025
    x = torch.stack([r, g, b], dim=1)
    lum = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
    return torch.clamp(lum.unsqueeze(1) + 1.15 * (x - lum.unsqueeze(1)), 0.0, 1.0)


def sharpen(x: torch.Tensor) -> torch.Tensor:
    kernel = torch.tensor(
        [[0.0, -1.0, 0.0], [-1.0, 5.0, -1.0], [0.0, -1.0, 0.0]],
        device=x.device,
        dtype=torch.float32,
    ).view(1, 1, 3, 3)
    kernel = kernel.expand(3, 1, 3, 3).contiguous()
    return F.conv2d(x, kernel, padding=1, groups=3)


def simulated_attention(x: torch.Tensor) -> torch.Tensor:
    b, _, h, w = x.shape
    seq = h * w
    tokens = x.view(b, 3, seq).transpose(1, 2).contiguous()
    proj = torch.randn(3, 128, device=x.device, dtype=x.dtype)
    q = tokens @ proj
    k = tokens @ proj
    v = tokens @ proj
    scale = q.shape[-1] ** -0.5
    attn = torch.bmm(q, k.transpose(1, 2)) * scale
    probs = torch.softmax(attn, dim=-1)
    out = torch.bmm(probs, v)
    out = out.transpose(1, 2).view(b, 128, h, w)
    return out.mean(dim=1, keepdim=True)


def pipeline(x: torch.Tensor) -> torch.Tensor:
    x = reinhard_tonemap(x)
    x = color_grade(x)
    x = sharpen(x)
    _ = simulated_attention(x)
    return x


def main() -> None:
    if DEVICE == "cpu":
        print("[WARNING] CUDA not available. Profiling CPU only; run on GPU for CUDA timing.")

    x = torch.rand(1, 3, 64, 64, device=DEVICE)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.insert(0, torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
        for _ in range(8):
            pipeline(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    sort_key = "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
    table = prof.key_averages().table(sort_by=sort_key, row_limit=10)
    print("\nLAYER 0 — AUTODHR PIPELINE PROFILER")
    print("====================================")
    print("Top 10 operations sorted by", sort_key)
    print(table)

    events = prof.key_averages()
    top = []
    for evt in sorted(events, key=lambda e: getattr(e, sort_key, 0.0), reverse=True)[:10]:
        top.append(
            {
                "op": evt.key,
                "cuda_time_total_us": float(getattr(evt, "cuda_time_total", 0.0)),
                "cpu_time_total_us": float(getattr(evt, "cpu_time_total", 0.0)),
                "self_cuda_time_total_us": float(getattr(evt, "self_cuda_time_total", 0.0)),
                "calls": int(evt.count),
            }
        )

    dominant = top[0]["op"] if top else "N/A"
    print(f"\nDominant bottleneck operation: {dominant}")

    payload = {
        "device": DEVICE,
        "sort_key": sort_key,
        "dominant_op": dominant,
        "top_ops": top,
        "note": "Measured with torch.profiler before optimization.",
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved results to {OUT_PATH}")


if __name__ == "__main__":
    main()
