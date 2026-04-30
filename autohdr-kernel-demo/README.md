# AutoHDR Kernel Optimization Demo

> 6-layer GPU cost-reduction architecture: runnable demos, real Triton kernels,
> FP8 benchmarks, LLM-guided autotuning, and integrated cost modeling.

## The Business Problem

- 128 GPUs × $3.50/hr × 24 hrs = ~$10,700/day in compute costs
- At $0.37/edit, every efficiency gain is direct margin
- 30-minute turnaround promise breaks under batch load without smart scheduling
- Premium $3–5/edit market requires faster, higher-quality diffusion inference

## Architecture Overview

| Layer | File | What it does |
|---|---|---|
| 1. Kernel Fusion | `triton_kernels/fused_ops.py` | Fused tone-map + color-grade + sharpen in one Triton kernel (2 fewer DRAM roundtrips) |
| 2. FP8 Precision | `fp8_demo.py`, `triton_kernels/fp8_gemm_bench.py` | Real `torch._scaled_mm` FP8 measurement on H100 SM90; fallback projection on other hardware |
| 3. Job Routing | `job_router.py` | Simulates 10k jobs routed across T4/A100/H100 by complexity tier |
| 4. Predictive Scheduling | `scheduler.py` | 30-day demand simulation: reactive vs. predictive GPU scheduling |
| 5. Model Compression | `distillation_spec.py`, `speculative_spec.py` | Architecture specs for distillation and speculative decoding |
| 6. Cost Aggregation | `cost_model.py`, `fusion_summary.py` | Daily/monthly/annual savings waterfall across all layers |

> **Note on `helion_kernel.py`:** Helion is a pre-release compiler not yet
> available on PyPI. The file always falls back to a PyTorch stub at runtime.
> It is retained as a future integration point for when Helion stabilizes.

Supporting files: `baseline.py` (PyTorch benchmark + CUDA Graph), `profiler_demo.py`
(torch.profiler pipeline with real SDPA attention), `agent_loop.py` (LLM-guided
kernel autotuning with optional Nsight Compute hardware feedback), `app.py`
(9-tab Streamlit dashboard).

## Folder Structure

```text
autohdr-kernel-demo/
├── demo/
│   ├── triton_kernels/
│   │   ├── __init__.py
│   │   ├── fused_ops.py          # Triton fused kernel (Layer 1)
│   │   └── fp8_gemm_bench.py     # Real FP8 GEMM benchmark (H100 only)
│   ├── baseline.py               # PyTorch benchmark + CUDA Graph
│   ├── profiler_demo.py          # torch.profiler pipeline + SDPA attention
│   ├── fp8_demo.py               # FP8 precision layer (measured or projected)
│   ├── job_router.py             # Job routing simulation
│   ├── scheduler.py              # Predictive scheduling simulation
│   ├── cost_model.py             # Cost aggregation waterfall
│   ├── fusion_summary.py         # Fusion savings summary
│   ├── distillation_spec.py      # Distillation architecture spec
│   ├── speculative_spec.py       # Speculative decoding spec
│   ├── agent_loop.py             # LLM-guided kernel autotuning
│   └── app.py                    # Streamlit dashboard (9 tabs)
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run simulations (no GPU required)
python demo/baseline.py
python demo/job_router.py
python demo/scheduler.py
python demo/cost_model.py

# 3. Run the Streamlit dashboard
streamlit run demo/app.py

# 4. H100 ONLY — real kernel benchmarks
python demo/triton_kernels/fp8_gemm_bench.py   # real FP8 GEMM
python demo/triton_kernels/fused_ops.py        # real Triton fusion speedup
python demo/profiler_demo.py                   # full pipeline profiler trace

# 5. LLM kernel autotuning (requires ANTHROPIC_API_KEY; ncu optional for H100 feedback)
python demo/agent_loop.py
```

## Benchmark Results

| Pipeline | Hardware | Time (ms) | Notes |
|---|---|---|---|
| Naive unfused (FP32) | H100 SXM5 | [GPU_MS] | 3 separate DRAM passes |
| Fused Triton kernel (FP16) | H100 SXM5 | [GPU_MS] | 1 DRAM pass, 2 roundtrips saved |
| torch.compile | H100 SXM5 | [GPU_MS] | Inductor backend |
| FP8 GEMM (E4M3) | H100 SXM5 | [GPU_MS] | torch._scaled_mm measured |
| FP8 GEMM (E4M3) | non-H100 | [PROJECTED] | bandwidth model fallback |

[GPU_MS] = fill in after running on H100. See `fp8_gemm_bench.py` and `fused_ops.py`.

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
