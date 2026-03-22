"""
reward.py - Milestone reward with continuous scaling above each tier.

Milestones at key thresholds + continuous scaling within each tier
so the model ALWAYS has gradient signal to push faster.

  -1.0       = wrong (compile fail, wrong output)
   0.1→1.0   = correct but slower than eager (linear ramp by speedup)
   2.0→2.99  = beats eager, continuous scaling toward torch.compile speed
   3.0→4.0   = beats torch.compile, continuous scaling by how much faster
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
