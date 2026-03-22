"""
reward.py - Milestone reward with linear ramp for correct kernels.

Discrete milestones at key thresholds + linear interpolation between them
so the model always has gradient signal to optimize speed.

  -1.0  = wrong (compile fail, wrong output)
   0.0  = correct but very slow (>= 10x slower than eager)
   1.0  = correct and matches eager speed
   2.0  = beats eager PyTorch
   3.0  = beats torch.compile
"""


def calculate_reward(sandbox_result: dict) -> float:
    """
    Reward for a correct kernel with linear ramp between milestones.

    Returns:
        0.0 to 1.0  — correct, linearly scaled by how close to eager speed
        2.0         — faster than eager PyTorch
        3.0         — faster than torch.compile
        0.0         — not correct or no timing data
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

    speedup = baseline_ms / kernel_ms  # >1 means faster than eager

    # Beats torch.compile?
    if compile_ms is not None and compile_ms > 0 and kernel_ms < compile_ms:
        return 3.0

    # Beats eager PyTorch?
    if speedup >= 1.0:
        return 2.0

    # Correct but slower — linear ramp from 0.0 (10x slower) to 1.0 (matches eager)
    # This gives gradient signal to improve speed even below 1.0x
    # clamp so even very slow correct kernels get a small positive reward
    return max(0.1, min(1.0, speedup))
