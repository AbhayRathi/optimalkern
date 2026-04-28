"""Layer 2 job router simulation for cost-aware GPU tier routing."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

OUT_PATH = Path(__file__).parent / "router_results.json"

GPU_COST_PER_HOUR = {"T4": 0.35, "A100": 3.50, "H100": 33.0}
JOB_TYPES = {
    "tonemap": {"h100_ms": 12, "cost_tier": "T4"},
    "color_grade": {"h100_ms": 15, "cost_tier": "T4"},
    "sky_replace": {"h100_ms": 180, "cost_tier": "A100"},
    "virtual_staging": {"h100_ms": 1800, "cost_tier": "H100"},
    "day_to_dusk": {"h100_ms": 1400, "cost_tier": "H100"},
}
JOB_WEIGHTS = {
    "tonemap": 0.25,
    "color_grade": 0.25,
    "sky_replace": 0.15,
    "virtual_staging": 0.20,
    "day_to_dusk": 0.15,
}


def simulate_queue(n_jobs: int = 10000, seed: int = 42) -> dict:
    random.seed(seed)
    job_names = list(JOB_WEIGHTS.keys())
    weights = [JOB_WEIGHTS[j] for j in job_names]

    jobs = random.choices(job_names, weights=weights, k=n_jobs)
    counts = Counter(jobs)

    naive_cost = 0.0
    routed_cost = 0.0
    per_job_rows = []

    for job_name, count in counts.items():
        profile = JOB_TYPES[job_name]
        gpu_seconds = profile["h100_ms"] / 1000.0

        naive_job_cost = gpu_seconds / 3600.0 * GPU_COST_PER_HOUR["H100"]
        routed_job_cost = gpu_seconds / 3600.0 * GPU_COST_PER_HOUR[profile["cost_tier"]]

        naive_cost += naive_job_cost * count
        routed_cost += routed_job_cost * count

        per_job_rows.append(
            {
                "job_type": job_name,
                "count": count,
                "h100_ms": profile["h100_ms"],
                "routed_tier": profile["cost_tier"],
                "naive_cost_usd": round(naive_job_cost * count, 2),
                "routed_cost_usd": round(routed_job_cost * count, 2),
            }
        )

    savings = naive_cost - routed_cost
    savings_pct = (savings / naive_cost * 100.0) if naive_cost else 0.0

    result = {
        "n_jobs": n_jobs,
        "gpu_cost_per_hour": GPU_COST_PER_HOUR,
        "job_distribution": dict(counts),
        "jobs": sorted(per_job_rows, key=lambda x: x["job_type"]),
        "naive_cost_usd": round(naive_cost, 2),
        "routed_cost_usd": round(routed_cost, 2),
        "savings_usd": round(savings, 2),
        "savings_pct": round(savings_pct, 2),
        "measured": False,
        "note": "Cost simulation using provided per-job H100 runtimes and GPU hourly pricing.",
    }
    return result


def main() -> None:
    result = simulate_queue(n_jobs=10000)

    print("LAYER 2 — SMART JOB ROUTER")
    print("===========================")
    print("Job Type         Count   H100 ms   Tier")
    print("------------------------------------------")
    for row in result["jobs"]:
        print(f"{row['job_type']:<16} {row['count']:<7} {row['h100_ms']:<9} {row['routed_tier']}")

    print("\nCost comparison (10,000 jobs):")
    print(f"Naive (all on H100): ${result['naive_cost_usd']:.2f}")
    print(f"Routed by tier:      ${result['routed_cost_usd']:.2f}")
    print(f"Savings:             ${result['savings_usd']:.2f} ({result['savings_pct']:.2f}%)")

    OUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved results to {OUT_PATH}")


if __name__ == "__main__":
    main()
