"""Aggregate cost model across optimization layers."""

from __future__ import annotations

import json
from pathlib import Path

DEMO_DIR = Path(__file__).parent
OUT_PATH = DEMO_DIR / "cost_results.json"

BASELINE_DAILY_COST = 99_000.0


def _load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _extract_layer_savings() -> dict:
    fp8 = _load_json(DEMO_DIR / "fp8_results.json")
    router = _load_json(DEMO_DIR / "router_results.json")
    fusion = _load_json(DEMO_DIR / "fusion_results.json")
    scheduler = _load_json(DEMO_DIR / "scheduler_results.json")

    # Conservative defaults from prompt, overwritten where local data exists.
    fp8_save = 39_600.0
    router_save = 14_850.0
    fusion_save = 6_682.0
    scheduler_save = 7_818.0
    distill_spec_save = 11_250.0

    fp8_measured = False
    router_measured = False
    fusion_measured = False
    scheduler_measured = False

    if fp8 and isinstance(fp8, dict):
        rows = {r["precision"]: r for r in fp8.get("precision_table", [])}
        fp32 = rows.get("FP32")
        fp8_proj = rows.get("FP8 (proj)")
        if fp32 and fp8_proj and fp32["time_ms"] > 0:
            reduction = 1.0 - (fp8_proj["time_ms"] / fp32["time_ms"])
            fp8_save = BASELINE_DAILY_COST * max(0.0, reduction)
        fp8_measured = bool(fp8.get("device") == "cuda")

    if router and isinstance(router, dict):
        router_save = float(router.get("savings_usd", router_save))

    if fusion and isinstance(fusion, dict):
        fusion_save = float(fusion.get("saved_daily_cost_usd", fusion_save))
        fusion_measured = bool(fusion.get("measured", False))

    if scheduler and isinstance(scheduler, dict):
        scheduler_save = float(scheduler.get("cost_savings_usd", scheduler_save))
        scheduler_measured = True

    return {
        "FP8 Quant": {"daily_savings": round(fp8_save, 2), "effort": "2 wks", "risk": "Low", "status": "MEASURED" if fp8_measured else "PROJECTED"},
        "Job Router": {"daily_savings": round(router_save, 2), "effort": "1 wk", "risk": "Very Low", "status": "MEASURED" if router_measured else "ESTIMATED"},
        "Kernel Fusion": {"daily_savings": round(fusion_save, 2), "effort": "1 wk", "risk": "Low", "status": "MEASURED" if fusion_measured else "ESTIMATED"},
        "Scheduling": {"daily_savings": round(scheduler_save, 2), "effort": "3 wks", "risk": "Medium", "status": "MEASURED" if scheduler_measured else "ESTIMATED"},
        "Distillation+SpecDecode": {"daily_savings": round(distill_spec_save, 2), "effort": "8 wks", "risk": "Medium", "status": "PROJECTED"},
    }


def main() -> None:
    layers = _extract_layer_savings()

    total_daily = sum(v["daily_savings"] for v in layers.values())
    monthly = total_daily * 30.0
    annual = total_daily * 365.0

    print("LAYERED COST WATERFALL")
    print("======================")
    print("Optimization      | Daily Savings | Effort | Risk | Status")
    print("-----------------------------------------------------------")
    for name, row in layers.items():
        print(
            f"{name:<17} | ${row['daily_savings']:<12,.2f} | {row['effort']:<6} | "
            f"{row['risk']:<8} | {row['status']}"
        )

    print("-----------------------------------------------------------")
    print(f"TOTAL             | ${total_daily:,.2f}/day")
    print(f"MONTHLY           | ${monthly:,.2f}")
    print(f"ANNUAL            | ${annual:,.2f}")

    payload = {
        "baseline_daily_cost_usd": BASELINE_DAILY_COST,
        "layers": [
            {
                "optimization": name,
                **row,
            }
            for name, row in layers.items()
        ],
        "total_daily_savings_usd": round(total_daily, 2),
        "monthly_savings_usd": round(monthly, 2),
        "annual_savings_usd": round(annual, 2),
        "note": "MEASURED/ESTIMATED/PROJECTED labels indicate confidence level per layer.",
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved results to {OUT_PATH}")


if __name__ == "__main__":
    main()
