"""Layer 6 predictive scheduling simulation over 30 days."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT_PATH = Path(__file__).parent / "scheduler_results.json"
H100_COST_PER_HOUR = 33.0
DAYS = 30
HOURS = 24 * DAYS

JOB_TYPES = ["tonemap", "color_grade", "sky_replace", "virtual_staging", "day_to_dusk"]
JOB_WEIGHTS = [0.25, 0.25, 0.15, 0.20, 0.15]

# Estimated end-to-end GPU-minutes per job in production-scale SD pipelines.
GPU_MINUTES_PER_JOB = {
    "tonemap": 2.0,
    "color_grade": 3.0,
    "sky_replace": 40.0,
    "virtual_staging": 800.0,
    "day_to_dusk": 700.0,
}

NON_URGENT = {"tonemap", "color_grade"}


def jobs_per_hour(hour_of_day: int) -> float:
    if 6 <= hour_of_day <= 14:
        return 20.0
    if 15 <= hour_of_day <= 18:
        return 200.0
    return 5.0


def sample_hourly_demand(seed: int = 7) -> tuple[list[int], list[float], list[int]]:
    rng = np.random.default_rng(seed)
    arrivals = []
    demand_gpu_minutes = []
    non_urgent_jobs = []

    for h in range(HOURS):
        hour = h % 24
        count = int(rng.poisson(jobs_per_hour(hour)))
        if count == 0:
            arrivals.append(0)
            demand_gpu_minutes.append(0.0)
            non_urgent_jobs.append(0)
            continue

        sampled_types = rng.choice(JOB_TYPES, size=count, p=JOB_WEIGHTS)
        demand = sum(GPU_MINUTES_PER_JOB[t] for t in sampled_types)
        non_urgent = int(sum(1 for t in sampled_types if t in NON_URGENT))

        arrivals.append(count)
        demand_gpu_minutes.append(float(demand))
        non_urgent_jobs.append(non_urgent)

    return arrivals, demand_gpu_minutes, non_urgent_jobs


def simulate_naive(arrivals: list[int], demand: list[float]) -> dict:
    backlog_gpu_minutes = 0.0
    prev_gpus = 0
    total_gpu_hours = 0.0
    wait_weighted_minutes = 0.0
    peak_gpu = 0
    hourly_gpu = []

    for i in range(HOURS):
        total_demand = backlog_gpu_minutes + demand[i]
        needed_gpu = math.ceil(total_demand / 60.0)

        # Reactive ramp constraints and spin-up overhead.
        if needed_gpu > prev_gpus:
            gpu_count = min(needed_gpu, prev_gpus + 350)
            spinup_overhead_gpu_hours = (gpu_count - prev_gpus) * 0.2
        else:
            gpu_count = max(needed_gpu, max(0, prev_gpus - 250))
            spinup_overhead_gpu_hours = 0.0

        capacity = gpu_count * 60.0
        served = min(total_demand, capacity)
        backlog_gpu_minutes = max(0.0, total_demand - served)

        utilization_wait = (backlog_gpu_minutes / max(1.0, capacity)) * 60.0
        wait_weighted_minutes += arrivals[i] * utilization_wait

        total_gpu_hours += gpu_count + spinup_overhead_gpu_hours
        peak_gpu = max(peak_gpu, gpu_count)
        hourly_gpu.append(gpu_count)
        prev_gpus = gpu_count

    avg_wait = wait_weighted_minutes / max(1, sum(arrivals))
    return {
        "strategy": "naive_reactive",
        "peak_gpu_count": int(peak_gpu),
        "total_gpu_hours": round(total_gpu_hours, 2),
        "total_cost_usd": round(total_gpu_hours * H100_COST_PER_HOUR, 2),
        "avg_job_wait_minutes": round(avg_wait, 2),
        "hourly_gpu_count": hourly_gpu,
    }


def simulate_predictive(arrivals: list[int], hourly_demand: list[float], non_urgent_jobs: list[int]) -> dict:
    # Work on a local mutable copy so deferred-load reshaping stays side-effect free for callers.
    demand_work = hourly_demand.copy()
    backlog_gpu_minutes = 0.0
    deferred = defaultdict(float)
    total_gpu_hours = 0.0
    wait_weighted_minutes = 0.0
    peak_gpu = 0
    hourly_gpu = []

    for i in range(HOURS):
        hour = i % 24

        # During peak, shift 70% of non-urgent jobs to next 2 hours.
        peak_period = 15 <= hour <= 18
        if peak_period and arrivals[i] > 0 and non_urgent_jobs[i] > 0:
            fraction_non_urgent = non_urgent_jobs[i] / arrivals[i]
            defer_amount = demand_work[i] * fraction_non_urgent * 0.70
            demand_work[i] -= defer_amount
            deferred[i + 1] += defer_amount * 0.55
            deferred[i + 2] += defer_amount * 0.45

        total_demand = backlog_gpu_minutes + demand_work[i] + deferred[i]
        needed_gpu = math.ceil(total_demand / 60.0)

        # Predictive policy: pre-warm before 3pm and cap peak fleet near 450-600.
        if hour == 14:
            gpu_count = max(450, needed_gpu)
        elif 15 <= hour <= 18:
            gpu_count = min(max(500, needed_gpu), 600)
        else:
            gpu_count = min(max(80, needed_gpu), 420)

        capacity = gpu_count * 60.0
        served = min(total_demand, capacity)
        backlog_gpu_minutes = max(0.0, total_demand - served)

        utilization_wait = (backlog_gpu_minutes / max(1.0, capacity)) * 60.0
        wait_weighted_minutes += arrivals[i] * utilization_wait

        total_gpu_hours += gpu_count
        peak_gpu = max(peak_gpu, gpu_count)
        hourly_gpu.append(gpu_count)

    avg_wait = wait_weighted_minutes / max(1, sum(arrivals))
    return {
        "strategy": "predictive",
        "peak_gpu_count": int(peak_gpu),
        "total_gpu_hours": round(total_gpu_hours, 2),
        "total_cost_usd": round(total_gpu_hours * H100_COST_PER_HOUR, 2),
        "avg_job_wait_minutes": round(avg_wait, 2),
        "hourly_gpu_count": hourly_gpu,
    }


def main() -> None:
    arrivals, demand, non_urgent = sample_hourly_demand()
    naive = simulate_naive(arrivals, demand.copy())
    predictive = simulate_predictive(arrivals, demand.copy(), non_urgent)

    result = {
        "days": DAYS,
        "h100_cost_per_hour": H100_COST_PER_HOUR,
        "arrival_pattern": {
            "off_peak_6_14_jobs_per_hour": 20,
            "peak_15_18_jobs_per_hour": 200,
            "night_19_6_jobs_per_hour": 5,
        },
        "demand_model_note": "GPU-min/job values are estimated to match observed peak-fleet scale.",
        "strategies": [naive, predictive],
        "peak_gpu_reduction": naive["peak_gpu_count"] - predictive["peak_gpu_count"],
        "peak_gpu_reduction_pct": round(
            (naive["peak_gpu_count"] - predictive["peak_gpu_count"]) / max(1, naive["peak_gpu_count"]) * 100.0,
            2,
        ),
        "cost_savings_usd": round(naive["total_cost_usd"] - predictive["total_cost_usd"], 2),
    }

    print("LAYER 6 — PEAK LOAD SCHEDULER")
    print("==============================")
    print("Strategy          Peak GPUs   GPU-hours(30d)   Cost(USD)      Avg wait (min)")
    print("--------------------------------------------------------------------------")
    for s in result["strategies"]:
        print(
            f"{s['strategy']:<17} {s['peak_gpu_count']:<11} {s['total_gpu_hours']:<15.2f} "
            f"${s['total_cost_usd']:<12.2f} {s['avg_job_wait_minutes']:.2f}"
        )

    print(
        f"\nPeak reduction: {result['peak_gpu_reduction']} GPUs "
        f"({result['peak_gpu_reduction_pct']:.2f}%)"
    )

    OUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved results to {OUT_PATH}")


if __name__ == "__main__":
    main()
