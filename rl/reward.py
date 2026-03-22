"""
reward.py - Discrete milestone reward for correct kernels.

Based on CUDA Agent (arXiv 2602.24286) discrete reward scheme:
  r = -1  if wrong
  r =  1  if correct
  r =  2  if beats eager PyTorch
  r =  3  if beats torch.compile

This function handles the correct-kernel milestones (1, 2, 3).
Negative rewards for wrong kernels are handled in train_grpo.py.
"""


def calculate_reward(sandbox_result: dict) -> float:
    """
    Discrete milestone reward for a correct kernel.

    Returns:
        1.0 — correct but slower than or equal to eager PyTorch
        2.0 — faster than eager PyTorch
        3.0 — faster than torch.compile
        0.0 — not correct or no timing data
    """
    if not sandbox_result.get("compiles", False):
        return 0.0
    if not sandbox_result.get("correct", False):
        return 0.0

    baseline_ms = sandbox_result.get("baseline_runtime_ms")
    kernel_ms = sandbox_result.get("runtime_ms")
    compile_ms = sandbox_result.get("compile_runtime_ms")

    if baseline_ms is None or kernel_ms is None or kernel_ms <= 0:
        return 1.0  # correct but no timing — give base correct reward

    # Beats torch.compile?
    if compile_ms is not None and compile_ms > 0 and kernel_ms < compile_ms:
        return 3.0

    # Beats eager PyTorch?
    if kernel_ms < baseline_ms:
        return 2.0

    # Correct but not faster
    return 1.0
