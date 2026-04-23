"""
agent_loop.py — Component 3

Uses the Anthropic API to iteratively improve a GPU kernel via an LLM agent.
Each iteration: generate kernel → benchmark → feed results back → improve.

Requirements:
    export ANTHROPIC_API_KEY="sk-ant-..."  # Set this environment variable first
    pip install anthropic

Run on a GPU:
    python demo/agent_loop.py
"""

import json
import os
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Optional: Anthropic client
# ---------------------------------------------------------------------------
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("[WARNING] anthropic package not installed. Run: pip install anthropic")

# ---------------------------------------------------------------------------
# Benchmarking block appended to each candidate kernel before execution
# ---------------------------------------------------------------------------
_BENCH_BLOCK = textwrap.dedent("""

import torch, time
_x = torch.rand(3, 2160, 3840, device='cuda' if torch.cuda.is_available() else 'cpu')
for _ in range(10):
    _result = fused_edit_kernel(_x)
if _x.is_cuda:
    torch.cuda.synchronize()
_t = time.perf_counter()
for _ in range(100):
    _result = fused_edit_kernel(_x)
if _x.is_cuda:
    torch.cuda.synchronize()
_ms = (time.perf_counter() - _t) / 100 * 1000
print(f"BENCHMARK_MS:{_ms:.4f}")
""")

# ---------------------------------------------------------------------------
# System prompt skeleton for Claude
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = (
    "You are an expert GPU kernel engineer specializing in PyTorch and Triton. "
    "When asked for a kernel implementation, return ONLY raw Python code. "
    "No markdown, no backtick fences, no explanation — just executable Python source."
)


def _build_user_prompt(
    operation_desc: str,
    previous_code: str | None,
    previous_ms: float | None,
    baseline_ms: float,
    iteration: int,
) -> str:
    """Build the user-facing prompt for Claude."""
    if iteration == 1 or previous_code is None:
        history_section = "This is your FIRST attempt. No previous kernel yet."
    else:
        history_section = (
            f"Previous kernel code:\n\n{previous_code}\n\n"
            f"Previous benchmark result: {previous_ms:.2f} ms"
        )

    return textwrap.dedent(f"""
        Write a Python function called `fused_edit_kernel(x, warmth=0.05, sat=1.15)` that implements:

        {operation_desc}

        Requirements:
        - Input x is a float32 torch.Tensor of shape [3, H, W] with values in [0, 2.0]
        - Output must be a float32 torch.Tensor of shape [3, H, W] with values in [0, 1.0]
        - Must run on CUDA if available, otherwise CPU (use x.device)
        - The function must be named exactly: fused_edit_kernel
        - You may use torch, torch.nn.functional, or triton — but NOT helion
        - Optimise for minimum execution time on a 4K (3×2160×3840) tensor

        Baseline to beat: {baseline_ms:.2f} ms

        {history_section}

        Return ONLY raw Python code — no markdown, no backticks, no explanation.
    """).strip()


def _extract_code(response_text: str) -> str:
    """
    Strip any accidental markdown fences Claude might return despite the instruction.
    """
    text = response_text.strip()

    # Remove ```python ... ``` or ``` ... ``` fences
    fence_pattern = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)
    matches = fence_pattern.findall(text)
    if matches:
        return matches[0].strip()

    return text


def _run_kernel(code: str) -> tuple[float | None, str]:
    """
    Write kernel code + bench block to a temp file, execute via subprocess.
    Returns (benchmark_ms, stdout) or (None, error_message) on failure.
    """
    tmp_path = Path("/tmp/agent_kernel_candidate.py")
    tmp_path.write_text(code + _BENCH_BLOCK, encoding="utf-8")

    try:
        result = subprocess.run(
            [sys.executable, str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        stdout = result.stdout + result.stderr
        for line in stdout.splitlines():
            if line.startswith("BENCHMARK_MS:"):
                ms = float(line.split(":")[1])
                return ms, stdout
        return None, f"No BENCHMARK_MS found in output:\n{stdout}"
    except subprocess.TimeoutExpired:
        return None, "Subprocess timed out after 120 s."
    except (subprocess.SubprocessError, ValueError, OSError) as exc:
        return None, str(exc)


def run_agent_loop(
    operation_desc: str,
    baseline_ms: float,
    n_iterations: int = 3,
) -> tuple[str, float, list[dict[str, Any]]]:
    """
    Run an iterative kernel improvement loop using Claude.

    Parameters
    ----------
    operation_desc : str
        Natural-language description of the kernel to implement.
    baseline_ms : float
        Execution time of the naive baseline to beat (milliseconds).
    n_iterations : int
        Number of LLM-guided improvement cycles.

    Returns
    -------
    best_kernel_code : str
    best_ms : float
    history : list of dicts with keys {iteration, time_ms, speedup, kernel_code}
    """
    if not ANTHROPIC_AVAILABLE:
        raise RuntimeError("anthropic package is required. Run: pip install anthropic")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY environment variable is not set.\n"
            "Export it with:  export ANTHROPIC_API_KEY='sk-ant-...'"
        )

    client = anthropic.Anthropic(api_key=api_key)

    history: list[dict[str, Any]] = []
    best_ms: float = float("inf")
    best_code: str = ""
    prev_code: str | None = None
    prev_ms: float | None = None

    print(f"\n{'='*60}")
    print(f"Agent Kernel Optimization Loop  ({n_iterations} iterations)")
    print(f"Baseline to beat: {baseline_ms:.2f} ms")
    print(f"{'='*60}\n")

    for i in range(1, n_iterations + 1):
        print(f"--- Iteration {i}/{n_iterations} ---")

        # 1. Build prompt and call Claude
        user_prompt = _build_user_prompt(
            operation_desc=operation_desc,
            previous_code=prev_code,
            previous_ms=prev_ms,
            baseline_ms=baseline_ms,
            iteration=i,
        )

        print("  Querying Claude claude-opus-4-5…", end=" ", flush=True)
        t0 = time.perf_counter()
        message = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=2000,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        elapsed_api = time.perf_counter() - t0
        print(f"done ({elapsed_api:.1f}s)")

        raw_code = message.content[0].text
        candidate_code = _extract_code(raw_code)

        # 2. Benchmark the candidate kernel
        print("  Benchmarking candidate kernel…", end=" ", flush=True)
        ms, output = _run_kernel(candidate_code)

        if ms is None:
            print("FAILED")
            print(f"  Error: {output[:300]}")
            record = {
                "iteration": i,
                "time_ms": None,
                "speedup": None,
                "kernel_code": candidate_code,
                "error": output[:500],
            }
        else:
            speedup = baseline_ms / ms
            print(f"{ms:.2f} ms  (speedup {speedup:.2f}x vs naive)")
            record = {
                "iteration": i,
                "time_ms": round(ms, 4),
                "speedup": round(speedup, 4),
                "kernel_code": candidate_code,
            }
            if ms < best_ms:
                best_ms = ms
                best_code = candidate_code

        history.append(record)
        prev_code = candidate_code
        prev_ms = ms

    # Summary
    print(f"\n{'='*60}")
    print("Final Improvement Table")
    print(f"{'='*60}")
    header = f"  {'Iter':<6} {'Time (ms)':<14} {'Speedup':<12} {'Status'}"
    print(header)
    print("  " + "-" * 50)
    for r in history:
        if r["time_ms"] is not None:
            print(f"  {r['iteration']:<6} {r['time_ms']:<14.2f} {r['speedup']:<12.2f} OK")
        else:
            print(f"  {r['iteration']:<6} {'—':<14} {'—':<12} FAILED")

    if best_ms < float("inf"):
        print(f"\nBest result: {best_ms:.2f} ms  ({baseline_ms / best_ms:.2f}x vs baseline)")
    else:
        print("\nNo successful benchmark in this run.")
        best_code = prev_code or ""

    return best_code, best_ms, history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    OPERATION_DESC = (
        "Fused real estate photo kernel: Reinhard tone mapping (x/(1+x) per channel) "
        "+ warmth shift (r+=0.05, b-=0.025) + saturation boost (factor 1.15 using "
        "luminance weights 0.299r/0.587g/0.114b). All fused into single elementwise "
        "pass. Input: float32 [3,H,W] values 0-2.0. Output: float32 [3,H,W] 0-1.0."
    )

    # Representative baseline — replace with a real number from baseline.py
    BASELINE_MS = 50.0

    best_code, best_ms, history = run_agent_loop(
        operation_desc=OPERATION_DESC,
        baseline_ms=BASELINE_MS,
        n_iterations=3,
    )

    # Persist results for the Streamlit app
    out_path = Path(__file__).parent / "agent_results.json"
    payload = {
        "baseline_ms": BASELINE_MS,
        "best_ms": best_ms if best_ms < float("inf") else None,
        "best_speedup": round(BASELINE_MS / best_ms, 4) if best_ms < float("inf") else None,
        "best_kernel_code": best_code,
        "history": history,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out_path}")
