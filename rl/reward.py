"""
reward.py - Speedup reward component for correct kernels.

NOTE: This function is called ONLY for kernels that are already correct.
It returns the speedup-based reward component added on top of reward_correct_base.

Full graduated reward logic (compile fail, shape mismatch, wrong outputs, regression
penalty, etc.) lives in train_grpo.py:_run_group_episodes() and is used during
RL training. This function is the speedup component only, used by:
  - train_grpo.py: `reward_correct_base + calculate_reward(eval_res)`
  - agent.py: standalone inference reward display
"""

import math


def calculate_reward(sandbox_result: dict, max_reward: float = 3.0) -> float:
    """
    Speedup reward for a correct kernel. Returns 0.0 if not correct or no timing data.

    Scaling:
      speedup <= 2x  → reward = speedup          (linear)
      speedup >  2x  → reward = 2 + log2(s/2)    (log-scaled to avoid gradient spikes)
      capped at max_reward=3.0

    Args:
        sandbox_result: dict from sandbox.evaluate()
        max_reward: cap to prevent gradient explosions from micro-benchmark noise

    Returns:
        float speedup reward (0.0 if kernel is wrong or timing unavailable)
    """
    if not sandbox_result.get("compiles", False):
        return 0.0
    if not sandbox_result.get("correct", False):
        return 0.0

    baseline_ms = sandbox_result.get("baseline_runtime_ms")
    kernel_ms = sandbox_result.get("runtime_ms")

    if baseline_ms is None or kernel_ms is None or kernel_ms <= 0:
        return 0.0

    speedup = baseline_ms / kernel_ms

    if speedup > 2.0:
        reward = 2.0 + math.log2(speedup / 2.0)
    else:
        reward = speedup

    return float(min(reward, max_reward))
