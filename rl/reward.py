"""
reward.py - Continuous reward signal across all outcome tiers.

Full reward spectrum so the model ALWAYS has gradient signal:

  -1.0              = no code extracted (format failure)
  -0.7              = compile failure (at least attempted code)
  -0.50 → -0.05    = wrong output, continuous by how wrong (shape, wrong_frac, bias)
   reward = speedup = correct kernel, reward IS the speedup ratio
                      (0.5x slower → 0.5, 1.0x same → 1.0, 2.0x faster → 2.0)
                      capped at 10.0 to prevent outlier destabilization
"""


def calculate_reward(sandbox_result: dict) -> float:
    """
    Reward = speedup ratio for correct kernels.

    Linear signal: 1.06x → 1.06, 1.34x → 1.34, 2.04x → 2.04
    No tier compression — every 1% speedup counts equally in GRPO advantage.

    Returns:
        0.0         — compile fail or wrong output
        0.5         — correct but no timing data
        speedup     — correct with timing (capped at 10.0)
    """
    if not sandbox_result.get("compiles", False):
        return 0.0
    if not sandbox_result.get("correct", False):
        return 0.0

    baseline_ms = sandbox_result.get("baseline_runtime_ms")
    kernel_ms = sandbox_result.get("runtime_ms")

    if baseline_ms is None or kernel_ms is None or kernel_ms <= 0:
        return 0.5  # correct but no timing

    speedup = baseline_ms / kernel_ms
    return min(10.0, max(0.1, speedup))


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
    Reward for optimization turns: speedup + delta bonus/penalty.

    Base reward is already the speedup ratio, so the model gets clear signal
    from absolute speed. The delta adds extra incentive to improve over the
    starting kernel during OPT turns.

    Args:
        sandbox_result: eval dict from sandbox
        baseline_speedup: the frozen speedup when optimization phase began

    Returns:
        speedup + delta  (delta can be negative for slowdowns)
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
