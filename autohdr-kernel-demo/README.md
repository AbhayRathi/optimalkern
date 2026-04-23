# AutoHDR Kernel Optimization Demo

> Built to demonstrate GPU pipeline optimization directly relevant to
> AutoHDR's real estate photo editing architecture.

## The Business Problem

- 128 GPUs × $3.50/hr × 24 hrs = ~$10,700/day in compute costs
- At $0.37/edit, every efficiency gain is direct margin
- 30-minute turnaround promise breaks under batch load without smart scheduling
- Premium $3–5/edit market requires faster, higher-quality diffusion inference

## What This Demonstrates

### 1. Kernel Fusion (`demo/baseline.py`, `demo/helion_kernel.py`)

AutoHDR's standard edit pipeline (tone map → color grade → sharpen →
composite) likely runs as separate GPU passes today. Each pass = one full
read + write from VRAM. Kernel fusion collapses N passes into 1 — same
math, dramatically less memory traffic. Benchmarked across 4 methods:
naive PyTorch → fused manual → `torch.compile` → Helion.

### 2. Agent-Driven Kernel Optimization (`demo/agent_loop.py`)

An LLM agent (Claude `claude-opus-4-5`) iteratively writes, benchmarks,
and improves GPU kernels automatically. Each iteration feeds the profiler
result back to Claude, which rewrites with targeted improvements. The loop
runs 3 iterations by default and saves history to `demo/agent_results.json`
for display in the Streamlit app.

### 3. Batch Architecture (`demo/batch_demo.py`)

Compares sequential, batched, and `torch.compile`d batched processing across
N ∈ {1, 4, 8, 16, 32} photos. Shows exactly where the 30-minute promise
breaks at scale and how batching fixes it. Results saved to
`demo/batch_results.json`.

### 4. Streamlit Demo App (`demo/app.py`)

Four-tab interactive app:
- **📸 Visual Results** — upload a photo or use the synthetic placeholder
- **⚡ Benchmark Results** — 4-method comparison table + bar chart
- **🤖 Agent Optimization Loop** — iteration-by-iteration improvement chart
- **🏗️ Batch Architecture** — total-time and per-photo-time charts

Works on CPU with placeholder data; shows real numbers after running the
benchmark scripts on a GPU.

## Folder Structure

```
autohdr-kernel-demo/
├── README.md               ← you are here
├── requirements.txt
├── demo/
│   ├── baseline.py         # naive / fused / torch.compile pipelines + benchmark
│   ├── helion_kernel.py    # @helion.kernel fused GPU kernel + 4-method comparison
│   ├── agent_loop.py       # Claude iterative kernel optimization loop
│   ├── batch_demo.py       # sequential vs batched vs compiled_batch throughput
│   └── app.py              # Streamlit demo (4 tabs)
└── assets/
    ├── sample.jpg           # synthetic warm-toned interior placeholder
    └── generate_sample.py   # regenerate sample.jpg
```

## Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the Streamlit app (CPU-safe, uses placeholder data)

```bash
streamlit run demo/app.py
```

### Generate real benchmark numbers (CUDA GPU required)

Run **in this order** on a CUDA-enabled machine:

```bash
# 1. Kernel-fusion benchmarks → prints table
python demo/baseline.py

# 2. Helion kernel comparison → saves demo/bench_results.json
python demo/helion_kernel.py

# 3. Agent optimization loop (set API key first)
export ANTHROPIC_API_KEY="sk-ant-..."
python demo/agent_loop.py          # saves demo/agent_results.json

# 4. Batch throughput demo → saves demo/batch_results.json
python demo/batch_demo.py

# 5. Launch app with real numbers
streamlit run demo/app.py
```

> **No GPU?** Use [Google Colab](https://colab.research.google.com) (free T4)
> or [Lambda Labs](https://lambdalabs.com) (~$0.50/hr A10).

## Benchmark Results

*(Placeholder — run the scripts above on a GPU to fill in real numbers)*

| Method          | Time (ms) | Speedup |
|-----------------|-----------|---------|
| Naive PyTorch   | [X]       | 1.00×   |
| Fused (manual)  | [X]       | [X]×    |
| torch.compile   | [X]       | [X]×    |
| Helion kernel   | [X]       | [X]×    |

## The Bigger Opportunity

80% of AutoHDR's GPU time lives in their Stable Diffusion pipeline —
virtual staging and day-to-dusk both run 50 denoising steps, each with
~16 attention layers. Each layer materialises a
`sequence_length × sequence_length` matrix in VRAM. For a 512 px image
that's 16M numbers written and read per layer, per step, per image.

FlashAttention-style kernels never write that matrix — they tile the
computation so intermediate results live in fast shared memory.
Direct result: 2–4× throughput on diffusion jobs, which translates to
sub-10-minute turnaround on virtual staging.

---

Built by Abhay Rathi | [github.com/AbhayRathi/optimalkern](https://github.com/AbhayRathi/optimalkern)
