# AutoHDR GPU Cost Reduction Demo (6 Layers)

This repo demonstrates a practical 6-layer optimization architecture for AutoHDR-scale workloads (up to 1000 H100 GPUs at peak, ~$99,000/day baseline).

## 6-Layer Architecture

```text
                +------------------------------+
Incoming Jobs ->| Layer 2: Job Router         |--> T4 / A100 / H100 tiering
                +------------------------------+
                              |
                              v
                +------------------------------+
                | Layer 6: Predictive Scheduler|--> Peak flattening + pre-warm
                +------------------------------+
                              |
                              v
                +------------------------------+
                | Layer 0: Profiler            |--> Find true bottleneck first
                +------------------------------+
                              |
                              v
                +------------------------------+
                | Layer 1: FP8 Quantization    |--> Memory-bandwidth win
                +------------------------------+
                              |
                              v
                +------------------------------+
                | Layer 3: Kernel Fusion       |--> Fewer VRAM passes
                +------------------------------+
                              |
                              v
                +------------------------------+
                | Layers 4+5: Model-level bets |--> Distillation + speculative
                +------------------------------+
```

## Built vs Specced

### BUILT in code
- **Layer 0**: `autohdr-kernel-demo/demo/profiler_demo.py`
- **Layer 1**: `autohdr-kernel-demo/demo/fp8_demo.py`
- **Layer 2**: `autohdr-kernel-demo/demo/job_router.py`
- **Layer 3**: `autohdr-kernel-demo/demo/fusion_summary.py` (+ `baseline.py`, `helion_kernel.py`)
- **Layer 6**: `autohdr-kernel-demo/demo/scheduler.py`
- **Cost model**: `autohdr-kernel-demo/demo/cost_model.py`
- **UI**: `autohdr-kernel-demo/demo/app.py`

### SPECCED (architectural only)
- **Layer 4**: `autohdr-kernel-demo/demo/distillation_spec.py`
- **Layer 5**: `autohdr-kernel-demo/demo/speculative_spec.py`

> Layers 4 and 5 are explicitly **specifications**, not implemented training/inference systems.

## Real Benchmark Table (fill after GPU runs)

| Layer | Metric | Value |
|---|---|---|
| Layer 0 Profiler | Dominant op | `[GPU_BOTTLENECK_OP]` |
| Layer 1 FP8 | FP32 attention time | `[GPU_MS]` |
| Layer 1 FP8 | FP16 attention time | `[GPU_MS]` |
| Layer 1 FP8 | FP16+compile time | `[GPU_MS]` |
| Layer 1 FP8 | FP8 projected time | `[GPU_MS]` |
| Layer 3 Fusion | Naive pipeline time | `[GPU_MS]` |
| Layer 3 Fusion | Best fused time | `[GPU_MS]` |
| Layer 6 Scheduler | Naive peak GPUs | `[GPU_COUNT]` |
| Layer 6 Scheduler | Predictive peak GPUs | `[GPU_COUNT]` |

## Cost Model (measured + projected)

| Optimization | Daily Savings | Effort | Risk | Status |
|---|---:|---|---|---|
| FP8 Quant | $39,600 | 2 wks | Low | projected |
| Job Router | $14,850 | 1 wk | Very Low | estimated/simulated |
| Kernel Fusion | $6,682 | 1 wk | Low | measured when benchmark JSON exists |
| Scheduling | $7,818 | 3 wks | Medium | estimated/simulated |
| Distillation + Speculative | $11,250 | 8 wks | Medium | projected |
| **TOTAL** | **$80,200/day** |  |  | mixed |
| **MONTHLY** | **$2,406,000** |  |  | mixed |
| **ANNUAL** | **$29,273,000** |  |  | mixed |

## How to Run

Install deps:

```bash
pip install -r autohdr-kernel-demo/requirements.txt
```

Run each layer script:

```bash
python autohdr-kernel-demo/demo/profiler_demo.py
python autohdr-kernel-demo/demo/fp8_demo.py
python autohdr-kernel-demo/demo/job_router.py
python autohdr-kernel-demo/demo/baseline.py
python autohdr-kernel-demo/demo/helion_kernel.py
python autohdr-kernel-demo/demo/fusion_summary.py
python autohdr-kernel-demo/demo/distillation_spec.py
python autohdr-kernel-demo/demo/speculative_spec.py
python autohdr-kernel-demo/demo/scheduler.py
python autohdr-kernel-demo/demo/cost_model.py
streamlit run autohdr-kernel-demo/demo/app.py
```

All generated `*_results.json` files are stored in `autohdr-kernel-demo/demo/` and auto-loaded by Streamlit.

## Exact Google Colab Commands

```bash
!git clone https://github.com/AbhayRathi/optimalkern.git
%cd optimalkern
!pip install -r autohdr-kernel-demo/requirements.txt

!python autohdr-kernel-demo/demo/profiler_demo.py
!python autohdr-kernel-demo/demo/fp8_demo.py
!python autohdr-kernel-demo/demo/job_router.py
!python autohdr-kernel-demo/demo/baseline.py
!python autohdr-kernel-demo/demo/helion_kernel.py
!python autohdr-kernel-demo/demo/fusion_summary.py
!python autohdr-kernel-demo/demo/scheduler.py
!python autohdr-kernel-demo/demo/cost_model.py
```

(Optional UI in Colab/local):

```bash
!streamlit run autohdr-kernel-demo/demo/app.py
```
