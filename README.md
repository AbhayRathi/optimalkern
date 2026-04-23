# AutoHDR Kernel Optimization Demo

> Built in 48 hours to demonstrate GPU pipeline optimization
> directly relevant to AutoHDR's architecture.

## The Business Problem
- 128 GPUs × $3.50/hr × 24 hrs = ~$10,700/day in compute costs
- At $0.37/edit, every efficiency gain is direct margin
- 30-minute turnaround promise breaks under batch load without
  smart scheduling
- Premium $3–5/edit market requires faster, higher-quality diffusion inference

## What This Demonstrates

### 1. Kernel Fusion
AutoHDR's standard edit pipeline (tone map → color grade → sharpen →
composite) likely runs as separate GPU passes today. Each pass = one full
read + write from VRAM. Kernel fusion collapses N passes into 1 — same
math, dramatically less memory traffic. Benchmarked across 4 methods
from naive PyTorch to Helion.

### 2. Agent-Driven Kernel Optimization
An LLM agent iteratively writes, benchmarks, and improves GPU kernels
automatically. Each iteration feeds profiler output back to Claude, which
rewrites with targeted improvements. This loop could run nightly on
AutoHDR's actual operations — the pipeline gets faster without anyone
touching the code.

### 3. Batch Architecture
Demonstrates the throughput difference between sequential vs batched
processing across N photos. Shows exactly where the 30-minute promise
breaks at scale and how batching fixes it.

## Project Structure

```
autohdr-kernel-demo/
├── demo/
│   ├── baseline.py        # Component 1 — naive / fused / compiled pipelines
│   ├── helion_kernel.py   # Component 2 — Helion GPU kernel + 4-method bench
│   ├── agent_loop.py      # Component 3 — Claude-driven iterative optimization
│   ├── batch_demo.py      # Component 4 — sequential vs batched throughput
│   └── app.py             # Component 5 — Streamlit demo app (4 tabs)
└── assets/
    ├── sample.jpg          # Synthetic interior placeholder image
    └── generate_sample.py  # Script to regenerate sample.jpg
```

## Benchmark Results
*(Placeholder — run on a GPU to fill with real numbers)*

| Method          | Time (ms) | Speedup |
|-----------------|-----------|---------|
| Naive PyTorch   | [X]       | 1.0x    |
| Fused (manual)  | [X]       | [X]x    |
| torch.compile   | [X]       | [X]x    |
| Helion kernel   | [X]       | [X]x    |

## How To Run

### Install dependencies
```bash
pip install torch torchvision helion streamlit matplotlib pandas Pillow anthropic
```

### Run the Streamlit app (works on CPU with placeholder data)
```bash
streamlit run autohdr-kernel-demo/demo/app.py
```

### Generate real benchmark numbers (GPU required)

Run **in order** on a CUDA-enabled machine:

```bash
# 1. Baseline benchmarks
python autohdr-kernel-demo/demo/baseline.py

# 2. Helion kernel benchmark
python autohdr-kernel-demo/demo/helion_kernel.py

# 3. Agent optimization loop (requires ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY="sk-ant-..."
python autohdr-kernel-demo/demo/agent_loop.py

# 4. Batch throughput demo
python autohdr-kernel-demo/demo/batch_demo.py

# 5. Launch the Streamlit app with real data
streamlit run autohdr-kernel-demo/demo/app.py
```

> **GPU Note:** CUDA is required for real benchmark numbers.
> Use Google Colab (free T4) or Lambda Labs ($0.50/hr A10) if no local GPU.

## The Bigger Opportunity

80% of AutoHDR's GPU time lives in their Stable Diffusion pipeline —
virtual staging and day-to-dusk both run 50 denoising steps, each with
~16 attention layers. Each attention layer materialises a
sequence_length × sequence_length matrix in VRAM. For a 512 px image
that's 16M numbers written and read per layer, per step, per image.

FlashAttention-style kernels never write that matrix. They tile the
computation so intermediate results live in fast shared memory.
Direct result: 2–4× throughput on diffusion jobs, which translates
to sub-10-minute turnaround on virtual staging — a product
transformation, not just an optimisation.
