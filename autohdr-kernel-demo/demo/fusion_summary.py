"""Layer 3 fusion summary from baseline/helion benchmarks."""

from __future__ import annotations

import json
from pathlib import Path

import torch

DEMO_DIR = Path(__file__).parent
BENCH_PATH = DEMO_DIR / "bench_results.json"
OUT_PATH = DEMO_DIR / "fusion_results.json"

DAILY_EDITS = 50000
H100_COST_PER_HOUR = 33.0


def _load_or_benchmark() -> list[dict]:
    if BENCH_PATH.exists():
        return json.loads(BENCH_PATH.read_text(encoding="utf-8"))

    print("bench_results.json not found; running local benchmark fallback.")
    try:
        from baseline import pipeline_naive, pipeline_fused, pipeline_compiled, benchmark
        from helion_kernel import fused_edit_kernel
    except ImportError as exc:
        raise ImportError(
            "Fallback benchmarking requires local demo modules baseline.py and helion_kernel.py."
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.rand(3, 720, 1280, device=device)
    n = 60 if device == "cuda" else 5

    naive_ms = benchmark(pipeline_naive, x, n=n)
    fused_ms = benchmark(pipeline_fused, x, n=n)
    compiled_ms = benchmark(pipeline_compiled, x, n=n)
    helion_ms = benchmark(fused_edit_kernel, x, n=n)

    results = [
        {"method": "Naive PyTorch", "time_ms": round(naive_ms, 3), "speedup": 1.0},
        {"method": "Fused (manual)", "time_ms": round(fused_ms, 3), "speedup": round(naive_ms / fused_ms, 3)},
        {"method": "torch.compile", "time_ms": round(compiled_ms, 3), "speedup": round(naive_ms / compiled_ms, 3)},
        {"method": "Helion kernel", "time_ms": round(helion_ms, 3), "speedup": round(naive_ms / helion_ms, 3)},
    ]
    BENCH_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    had_existing_results = BENCH_PATH.exists()
    rows = _load_or_benchmark()

    naive = next((r for r in rows if "Naive" in r["method"]), rows[0])
    fused = min(rows[1:], key=lambda r: r["time_ms"]) if len(rows) > 1 else rows[0]

    naive_daily_ms = DAILY_EDITS * naive["time_ms"]
    fused_daily_ms = DAILY_EDITS * fused["time_ms"]

    naive_gpu_hours = naive_daily_ms / 3_600_000.0
    fused_gpu_hours = fused_daily_ms / 3_600_000.0
    saved_gpu_hours = naive_gpu_hours - fused_gpu_hours

    naive_daily_cost = naive_gpu_hours * H100_COST_PER_HOUR
    fused_daily_cost = fused_gpu_hours * H100_COST_PER_HOUR
    saved_daily_cost = naive_daily_cost - fused_daily_cost

    summary = {
        "daily_edits": DAILY_EDITS,
        "h100_cost_per_hour": H100_COST_PER_HOUR,
        "benchmark_rows": rows,
        "naive_method": naive,
        "fused_method": fused,
        "naive_gpu_hours_per_day": round(naive_gpu_hours, 3),
        "fused_gpu_hours_per_day": round(fused_gpu_hours, 3),
        "saved_gpu_hours_per_day": round(saved_gpu_hours, 3),
        "naive_daily_cost_usd": round(naive_daily_cost, 2),
        "fused_daily_cost_usd": round(fused_daily_cost, 2),
        "saved_daily_cost_usd": round(saved_daily_cost, 2),
        "measured": had_existing_results,
    }

    print("LAYER 3 — FUSION SUMMARY")
    print("=========================")
    print(f"Naive: {naive['method']} @ {naive['time_ms']:.3f} ms")
    print(f"Best fused: {fused['method']} @ {fused['time_ms']:.3f} ms")
    print(f"Daily GPU-hours (naive): {naive_gpu_hours:.3f}")
    print(f"Daily GPU-hours (fused): {fused_gpu_hours:.3f}")
    print(f"Daily GPU-hours saved:   {saved_gpu_hours:.3f}")
    print(f"Daily $ saved @ ${H100_COST_PER_HOUR:.2f}/hr: ${saved_daily_cost:.2f}")

    OUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved results to {OUT_PATH}")


if __name__ == "__main__":
    main()
