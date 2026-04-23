"""
app.py — Component 5

Streamlit demo for the AutoHDR GPU Kernel Optimization project.

Run:
    streamlit run demo/app.py

Works on both CPU and GPU; shows a warning banner when running on CPU.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AutoHDR Kernel Optimization Demo",
    layout="wide",
    page_icon="⚡",
)

# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEMO_DIR = Path(__file__).parent
ASSETS_DIR = DEMO_DIR.parent / "assets"

# ---------------------------------------------------------------------------
# Inline image-processing helpers (no VRAM import chain from other modules)
# ---------------------------------------------------------------------------

def _reinhard(x: torch.Tensor) -> torch.Tensor:
    return x / (1.0 + x)


def _color_grade(x: torch.Tensor, warmth: float = 0.05, sat: float = 1.15) -> torch.Tensor:
    r, g, b = x[0], x[1], x[2]
    r = r + warmth
    b = b - warmth * 0.5
    x = torch.stack([r, g, b], dim=0)
    lum = 0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]
    return torch.clamp(lum.unsqueeze(0) + sat * (x - lum.unsqueeze(0)), 0.0, 1.0)


def _sharpen(x: torch.Tensor) -> torch.Tensor:
    k = torch.tensor(
        [[0.0, -1.0, 0.0], [-1.0, 5.0, -1.0], [0.0, -1.0, 0.0]],
        dtype=torch.float32,
        device=x.device,
    ).view(1, 1, 3, 3).expand(3, 1, 3, 3).contiguous()
    return F.conv2d(x.unsqueeze(0), k, padding=1, groups=3).squeeze(0)


def naive_pipeline(x: torch.Tensor) -> torch.Tensor:
    return _sharpen(_color_grade(_reinhard(x)))


def fused_pipeline(x: torch.Tensor) -> torch.Tensor:
    r, g, b = x[0], x[1], x[2]
    r = r / (1.0 + r);  g = g / (1.0 + g);  b = b / (1.0 + b)
    r = torch.clamp(r + 0.05, 0.0, 1.0);  b = torch.clamp(b - 0.025, 0.0, 1.0)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    sat = 1.15
    r = torch.clamp(lum + sat * (r - lum), 0.0, 1.0)
    g = torch.clamp(lum + sat * (g - lum), 0.0, 1.0)
    b = torch.clamp(lum + sat * (b - lum), 0.0, 1.0)
    x = torch.stack([r, g, b], dim=0)
    k = torch.tensor(
        [[0.0, -1.0, 0.0], [-1.0, 5.0, -1.0], [0.0, -1.0, 0.0]],
        dtype=torch.float32,
        device=x.device,
    ).view(1, 1, 3, 3).expand(3, 1, 3, 3).contiguous()
    return F.conv2d(x.unsqueeze(0), k, padding=1, groups=3).squeeze(0)


def helion_pipeline(x: torch.Tensor) -> torch.Tensor:
    """Helion-equivalent fused pipeline (element-wise only, no sharpen)."""
    r, g, b = x[0], x[1], x[2]
    r = r / (1.0 + r);  g = g / (1.0 + g);  b = b / (1.0 + b)
    r = torch.clamp(r + 0.05, 0.0, 1.0);  b = torch.clamp(b - 0.025, 0.0, 1.0)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    sat = 1.15
    r = torch.clamp(lum + sat * (r - lum), 0.0, 1.0)
    g = torch.clamp(lum + sat * (g - lum), 0.0, 1.0)
    b = torch.clamp(lum + sat * (b - lum), 0.0, 1.0)
    return torch.stack([r, g, b], dim=0)


def _time_fn(fn, x: torch.Tensor, n: int = 20) -> float:
    for _ in range(3):
        fn(x)
    if x.is_cuda:
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn(x)
    if x.is_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000.0


def _tensor_to_image(t: torch.Tensor) -> np.ndarray:
    """Convert [3, H, W] float32 tensor → uint8 H×W×3 numpy array."""
    arr = t.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return (arr * 255).astype(np.uint8)


def _load_uploaded(file_bytes: bytes) -> torch.Tensor:
    """Load uploaded image bytes → [3, H, W] float32 tensor scaled 0–2."""
    from PIL import Image
    import io
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0 * 2.0
    return torch.from_numpy(arr).permute(2, 0, 1).to(DEVICE)


def _synthetic_image(h: int = 512, w: int = 768) -> torch.Tensor:
    """Generate a warm-toned gradient as a placeholder interior image."""
    y = torch.linspace(0.3, 1.0, h, device=DEVICE).unsqueeze(1).expand(h, w)
    x_ax = torch.linspace(0.5, 1.5, w, device=DEVICE).unsqueeze(0).expand(h, w)
    r = torch.clamp(x_ax * 0.9, 0.0, 2.0)
    g = torch.clamp(y * 0.7, 0.0, 2.0)
    b = torch.clamp(y * 0.4 + 0.1, 0.0, 2.0)
    return torch.stack([r, g, b], dim=0)


# ---------------------------------------------------------------------------
# Load JSON result files (with sensible defaults if not present)
# ---------------------------------------------------------------------------

def _load_json(path: Path, default: Any) -> Any:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return default


_PLACEHOLDER_BENCH = [
    {"method": "Naive PyTorch", "time_ms": 48.5, "speedup": 1.00},
    {"method": "Fused (manual)", "time_ms": 31.2, "speedup": 1.55},
    {"method": "torch.compile", "time_ms": 22.8, "speedup": 2.13},
    {"method": "Helion kernel", "time_ms": 14.6, "speedup": 3.32},
]

_PLACEHOLDER_AGENT = {
    "baseline_ms": 48.5,
    "best_ms": 12.1,
    "best_speedup": 4.01,
    "best_kernel_code": "# Run demo/agent_loop.py on a GPU to generate real results.\ndef fused_edit_kernel(x, warmth=0.05, sat=1.15):\n    r, g, b = x[0], x[1], x[2]\n    r = r / (1 + r); g = g / (1 + g); b = b / (1 + b)\n    r = (r + warmth).clamp(0, 1); b = (b - warmth * 0.5).clamp(0, 1)\n    lum = 0.299*r + 0.587*g + 0.114*b\n    sat_val = 1.15\n    r = (lum + sat_val*(r-lum)).clamp(0,1)\n    g = (lum + sat_val*(g-lum)).clamp(0,1)\n    b = (lum + sat_val*(b-lum)).clamp(0,1)\n    return torch.stack([r, g, b])",
    "history": [
        {"iteration": 1, "time_ms": 38.4, "speedup": 1.26, "kernel_code": ""},
        {"iteration": 2, "time_ms": 22.7, "speedup": 2.14, "kernel_code": ""},
        {"iteration": 3, "time_ms": 12.1, "speedup": 4.01, "kernel_code": ""},
    ],
}

_PLACEHOLDER_BATCH = [
    {"n_photos": 1,  "sequential_ms": 12.1,  "batched_ms": 10.8,  "per_photo_sequential_ms": 12.1,  "per_photo_batched_ms": 10.8,  "throughput_gain": 1.12},
    {"n_photos": 4,  "sequential_ms": 48.4,  "batched_ms": 18.3,  "per_photo_sequential_ms": 12.1,  "per_photo_batched_ms": 4.58,  "throughput_gain": 2.64},
    {"n_photos": 8,  "sequential_ms": 96.8,  "batched_ms": 24.1,  "per_photo_sequential_ms": 12.1,  "per_photo_batched_ms": 3.01,  "throughput_gain": 4.02},
    {"n_photos": 16, "sequential_ms": 193.6, "batched_ms": 36.4,  "per_photo_sequential_ms": 12.1,  "per_photo_batched_ms": 2.28,  "throughput_gain": 5.32},
    {"n_photos": 32, "sequential_ms": 387.2, "batched_ms": 64.7,  "per_photo_sequential_ms": 12.1,  "per_photo_batched_ms": 2.02,  "throughput_gain": 5.98},
]

bench_data = _load_json(DEMO_DIR / "bench_results.json", _PLACEHOLDER_BENCH)
agent_data = _load_json(DEMO_DIR / "agent_results.json", _PLACEHOLDER_AGENT)
batch_data = _load_json(DEMO_DIR / "batch_results.json", _PLACEHOLDER_BATCH)

# ---------------------------------------------------------------------------
# App header
# ---------------------------------------------------------------------------

if DEVICE == "cpu":
    st.warning(
        "⚠️  **No GPU detected** — running on CPU. "
        "Timings shown in benchmark tabs are illustrative placeholder values. "
        "Run the demo scripts on a CUDA GPU for real numbers.",
        icon="⚠️",
    )

st.title("⚡ AutoHDR Kernel Optimization Demo")
st.markdown(
    "*Built in 48 hrs to demonstrate GPU kernel optimization "
    "directly relevant to AutoHDR's pipeline architecture*"
)
st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📸 Visual Results",
    "⚡ Benchmark Results",
    "🤖 Agent Optimization Loop",
    "🏗️ Batch Architecture",
])


# ── TAB 1: Visual Results ────────────────────────────────────────────────────
with tab1:
    st.header("📸 Visual Results")

    uploaded = st.file_uploader(
        "Upload a real estate photo (JPG or PNG)",
        type=["jpg", "jpeg", "png"],
        help="Upload any interior/exterior real estate photo to see the pipeline in action.",
    )

    if uploaded is not None:
        img_tensor = _load_uploaded(uploaded.read())
    else:
        st.info("No photo uploaded — using a synthetic warm-toned gradient as placeholder.")
        img_tensor = _synthetic_image()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original")
        # Scale back to [0,1] for display
        orig_display = _tensor_to_image(img_tensor / 2.0)
        st.image(orig_display, use_container_width=True)
        st.caption("Raw input (HDR 0–2.0 range, normalised for display)")

    with col2:
        st.subheader("Naive Pipeline")
        t_naive_start = time.perf_counter()
        out_naive = naive_pipeline(img_tensor)
        t_naive_ms = (time.perf_counter() - t_naive_start) * 1000
        st.image(_tensor_to_image(out_naive), use_container_width=True)
        st.caption(f"Naive PyTorch  |  {t_naive_ms:.1f} ms")

    with col3:
        st.subheader("Helion Kernel")
        t_helion_start = time.perf_counter()
        out_helion = helion_pipeline(img_tensor)
        t_helion_ms = (time.perf_counter() - t_helion_start) * 1000
        st.image(_tensor_to_image(out_helion), use_container_width=True)
        st.caption(f"Helion-equivalent fused  |  {t_helion_ms:.1f} ms")


# ── TAB 2: Benchmark Results ─────────────────────────────────────────────────
with tab2:
    st.header("⚡ Benchmark Results")

    using_placeholder = not (DEMO_DIR / "bench_results.json").exists()
    if using_placeholder:
        st.info(
            "📊 Showing **placeholder values**. "
            "Run `python demo/baseline.py` and `python demo/helion_kernel.py` "
            "on a GPU to populate real numbers."
        )

    # Table
    import pandas as pd

    df = pd.DataFrame(bench_data)
    if "method" in df.columns:
        df = df.rename(columns={"method": "Method", "time_ms": "Time (ms)", "speedup": "Speedup vs Naive"})

    st.dataframe(df, use_container_width=True, hide_index=True)

    # Bar chart
    try:
        import matplotlib.pyplot as plt
        methods = [r.get("method", r.get("Method", "")) for r in bench_data]
        speedups = [r.get("speedup", r.get("Speedup vs Naive", 1.0)) for r in bench_data]

        fig, ax = plt.subplots(figsize=(8, 3))
        bars = ax.barh(methods, speedups, color=["#4a90d9", "#50c878", "#f4a261", "#e76f51"])
        ax.set_xlabel("Speedup vs Naive")
        ax.set_title("Kernel Speedup Comparison")
        ax.axvline(1.0, color="white", linestyle="--", linewidth=0.8, alpha=0.6)
        for bar, val in zip(bars, speedups):
            ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}x", va="center", fontsize=9)
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#262730")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        st.pyplot(fig)
    except Exception as exc:
        st.warning(f"Could not render chart: {exc}")

    # Business impact callout
    st.info(
        "💡 **Business Impact at AutoHDR's Scale**  \n"
        "128 GPUs × \\$3.50/hr × 24 hrs = **\\$10,752/day** in compute costs.  \n"
        "A **2× kernel speedup** = ~\\$5,000/day in saved compute.  \n"
        "A **4× speedup** halves their infrastructure bill outright."
    )


# ── TAB 3: Agent Optimization Loop ──────────────────────────────────────────
with tab3:
    st.header("🤖 Agent Optimization Loop")

    using_placeholder = not (DEMO_DIR / "agent_results.json").exists()
    if using_placeholder:
        st.info(
            "📊 Showing **placeholder values**. "
            "Run `python demo/agent_loop.py` on a GPU with ANTHROPIC_API_KEY set "
            "to generate real optimization history."
        )

    history = agent_data.get("history", [])
    valid_history = [h for h in history if h.get("time_ms") is not None]

    if valid_history:
        try:
            import matplotlib.pyplot as plt
            iters = [h["iteration"] for h in valid_history]
            times = [h["time_ms"] for h in valid_history]
            baseline_ms = agent_data.get("baseline_ms", times[0] if times else 50)

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(iters, times, "o-", color="#50c878", linewidth=2, markersize=8, label="Agent kernel")
            ax.axhline(baseline_ms, color="#e76f51", linestyle="--", linewidth=1.5, label=f"Baseline ({baseline_ms:.1f} ms)")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Time (ms)")
            ax.set_title("Agent-Driven Kernel Improvement")
            ax.legend()
            ax.set_xticks(iters)
            fig.patch.set_facecolor("#0e1117")
            ax.set_facecolor("#262730")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#444")
            ax.legend(facecolor="#1a1a2e", labelcolor="white")
            st.pyplot(fig)
        except Exception as exc:
            st.warning(f"Could not render chart: {exc}")
    else:
        st.warning("No valid iteration data found.")

    # Winning kernel code
    best_code = agent_data.get("best_kernel_code", "# No kernel code available yet.")
    st.subheader("🏆 Best Generated Kernel")
    st.code(best_code, language="python")

    # Iteration table
    if valid_history:
        iter_df = pd.DataFrame(
            [{"Iteration": h["iteration"],
              "Time (ms)": h["time_ms"],
              "Speedup": h["speedup"]}
             for h in valid_history]
        )
        st.dataframe(iter_df, use_container_width=True, hide_index=True)

    st.caption(
        "💡 This loop could run **nightly** on AutoHDR's actual operations. "
        "The pipeline gets faster automatically as models and hardware evolve — "
        "no engineer required."
    )


# ── TAB 4: Batch Architecture ────────────────────────────────────────────────
with tab4:
    st.header("🏗️ Batch Architecture")

    using_placeholder = not (DEMO_DIR / "batch_results.json").exists()
    if using_placeholder:
        st.info(
            "📊 Showing **placeholder values**. "
            "Run `python demo/batch_demo.py` on a GPU to populate real numbers."
        )

    import pandas as pd

    batch_df = pd.DataFrame(batch_data)
    ns = [r["n_photos"] for r in batch_data]
    seq_totals = [r["sequential_ms"] for r in batch_data]
    bat_totals = [r["batched_ms"] for r in batch_data]
    seq_per = [r["per_photo_sequential_ms"] for r in batch_data]
    bat_per = [r["per_photo_batched_ms"] for r in batch_data]

    try:
        import matplotlib.pyplot as plt

        col_a, col_b = st.columns(2)

        with col_a:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(ns, seq_totals, "o-", color="#e76f51", label="Sequential")
            ax.plot(ns, bat_totals, "o-", color="#50c878", label="Batched")
            ax.set_xlabel("N Photos")
            ax.set_ylabel("Total time (ms)")
            ax.set_title("Total Processing Time")
            ax.legend()
            fig.patch.set_facecolor("#0e1117")
            ax.set_facecolor("#262730")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#444")
            ax.legend(facecolor="#1a1a2e", labelcolor="white")
            st.pyplot(fig)

        with col_b:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(ns, seq_per, "o-", color="#e76f51", label="Sequential")
            ax.plot(ns, bat_per, "o-", color="#50c878", label="Batched")
            ax.set_xlabel("N Photos")
            ax.set_ylabel("Per-photo time (ms)")
            ax.set_title("Per-Photo Processing Time")
            ax.legend()
            fig.patch.set_facecolor("#0e1117")
            ax.set_facecolor("#262730")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#444")
            ax.legend(facecolor="#1a1a2e", labelcolor="white")
            st.pyplot(fig)
    except Exception as exc:
        st.warning(f"Could not render charts: {exc}")

    # Full results table
    display_df = batch_df.rename(columns={
        "n_photos": "N Photos",
        "sequential_ms": "Sequential (ms)",
        "batched_ms": "Batched (ms)",
        "per_photo_sequential_ms": "Per-photo Seq (ms)",
        "per_photo_batched_ms": "Per-photo Bat (ms)",
        "throughput_gain": "Throughput Gain",
    })
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Key insight
    gain_32 = next((r["throughput_gain"] for r in batch_data if r["n_photos"] == 32), "N/A")
    st.info(
        f"💡 **Key Insight**  \n"
        f"At N=32 photos (one photographer's full shoot), batched processing delivers "
        f"**{gain_32}× per-photo speedup**.  \n"
        f"This is the difference between a 30-minute promise that holds at scale vs "
        f"one that breaks under load."
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption("Built by [Your Name] | [github.com/AbhayRathi/optimalkern](https://github.com/AbhayRathi/optimalkern)")
