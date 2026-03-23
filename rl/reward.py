"""
reward.py - Continuous reward signal across all outcome tiers.

Full reward spectrum so the model ALWAYS has gradient signal:

  -1.0              = no code extracted (format failure)
  -0.7              = compile failure (at least attempted code)
  -0.50 → -0.05    = wrong output, continuous by how wrong (shape, wrong_frac, bias)
   0.1  →  1.0     = correct but slower than eager (linear ramp by speedup)
   2.0  →  2.99    = beats eager, continuous scaling toward torch.compile speed
   3.0  →  4.0     = beats torch.compile, continuous scaling by how much faster
"""

import math


def calculate_reward(sandbox_result: dict) -> float:
    """
    Reward for a correct kernel with continuous scaling within milestones.

    Returns:
        0.1 to 1.0  — correct but slower than eager (linear by speedup)
        2.0 to 2.99 — beats eager, scales toward torch.compile threshold
        3.0 to 4.0  — beats torch.compile, scales by additional speedup
        0.5         — correct but no timing data
    """
    if not sandbox_result.get("compiles", False):
        return 0.0
    if not sandbox_result.get("correct", False):
        return 0.0

    baseline_ms = sandbox_result.get("baseline_runtime_ms")
    kernel_ms = sandbox_result.get("runtime_ms")
    compile_ms = sandbox_result.get("compile_runtime_ms")

    if baseline_ms is None or kernel_ms is None or kernel_ms <= 0:
        return 0.5  # correct but no timing

    speedup_vs_eager = baseline_ms / kernel_ms

    # Tier 3: Beats torch.compile → 3.0 + log bonus (capped at 4.0)
    if compile_ms is not None and compile_ms > 0 and kernel_ms < compile_ms:
        speedup_vs_compile = compile_ms / kernel_ms
        # log2 scaling: 1x over compile = 3.0, 2x over compile = 3.5, 4x = 4.0
        bonus = min(1.0, 0.5 * math.log2(max(1.0, speedup_vs_compile)))
        return 3.0 + bonus

    # Tier 2: Beats eager → 2.0 + linear scaling toward compile threshold
    if speedup_vs_eager >= 1.0:
        if compile_ms is not None and compile_ms > 0:
            # How close to beating compile? Scale 2.0→2.99
            compile_speedup = baseline_ms / compile_ms  # how fast compile is vs eager
            if compile_speedup > 1.0 and speedup_vs_eager < compile_speedup:
                progress = (speedup_vs_eager - 1.0) / (compile_speedup - 1.0)
                return 2.0 + 0.99 * min(1.0, progress)
        # No compile timing or compile is slower than eager
        return 2.0 + min(0.99, 0.5 * math.log2(max(1.0, speedup_vs_eager)))

    # Tier 1: Correct but slower — linear ramp 0.1 to 1.0
    return max(0.1, min(1.0, speedup_vs_eager))


def calculate_wrong_reward(sandbox_result: dict) -> float:
    """
    Continuous reward for kernels that compile but produce wrong output.

    Uses sandbox signals (wrong_frac, shape_ok, max_abs_error, systematic_bias)
    to create gradient from "total garbage" to "almost correct."

    Range: -0.50 (total garbage) → -0.05 (borderline correct)
    """
    # Shape completely wrong — worst wrong-output tier
    if sandbox_result.get("shape_ok") is False:
        return -0.50

    wrong_frac = sandbox_result.get("wrong_frac")
    if wrong_frac is None:
        wrong_frac = 1.0
    bias = abs(sandbox_result.get("systematic_bias") or 0.0)

    # Nearly all elements wrong — close to compile-fail quality
    if wrong_frac > 0.9:
        return -0.45

    # Most elements wrong but shape is right — linear interpolation
    if wrong_frac > 0.3:
        # 0.9 → -0.40, 0.3 → -0.20
        t = (wrong_frac - 0.3) / 0.6
        return -0.20 - 0.20 * t

    # Few elements wrong — likely boundary or precision issue
    if wrong_frac > 0.0:
        # 0.3 → -0.20, ~0 → -0.05
        t = wrong_frac / 0.3
        return -0.05 - 0.15 * t

    # wrong_frac ≈ 0 but failed correctness check — systematic bias
    if bias > 0.0:
        return max(-0.15, -0.05 - 0.1 * min(1.0, bias))

    return -0.05  # passed element check but failed some other assert


def calculate_opt_reward(sandbox_result: dict, baseline_speedup: float) -> float:
    """
    Reward for optimization turns: base reward + delta bonus/penalty.

    During optimization turns, the model receives a working kernel and must
    make it faster. The base reward alone doesn't differentiate speed well
    (1.23x and 1.33x both get ~3.9), so we add a bonus/penalty based on
    improvement over the baseline speedup they were given.

    Args:
        sandbox_result: eval dict from sandbox
        baseline_speedup: the frozen speedup when optimization phase began

    Returns:
        base_reward + delta_bonus  (bonus can be negative for slowdowns)
    """
    base_reward = calculate_reward(sandbox_result)

    # Only apply delta for correct kernels with timing data
    if base_reward < 0.1 or baseline_speedup <= 0:
        return base_reward

    kernel_ms = sandbox_result.get("runtime_ms")
    baseline_ms = sandbox_result.get("baseline_runtime_ms")
    if not kernel_ms or not baseline_ms or kernel_ms <= 0:
        return base_reward

    kernel_speedup = baseline_ms / kernel_ms
    improvement_ratio = kernel_speedup / baseline_speedup  # >1 = faster

    if improvement_ratio > 1.02:  # at least 2% faster
        # +0.5 per 10% improvement, capped at +2.0
        delta_bonus = min(2.0, 5.0 * (improvement_ratio - 1.0))
        return base_reward + delta_bonus
    elif improvement_ratio < 0.98:  # got slower
        # Penalty proportional to slowdown, capped at -1.0
        slowdown_penalty = min(1.0, 5.0 * (1.0 - improvement_ratio))
        return base_reward - slowdown_penalty
    else:
        return base_reward  # within 2% = no change
