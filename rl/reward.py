"""
reward.py - SP2 Reward Function for RL

Maps the execution sandbox evaluation metrics to a scalar RL reward.
Focuses strictly on correctness (binary 0/1 gate) and speedup (multiplier).
"""

def calculate_reward(sandbox_result: dict, max_reward: float = 10.0) -> float:
    """
    Calculate RL reward from sandbox evaluation result.
    
    Args:
        sandbox_result: Dictionary from sandbox.evaluate() containing
                        compiles, correct, runtime_ms, baseline_runtime_ms
        max_reward: Cap the maximum reward to prevent gradient explosions
                   from tiny measurement anomalies
                   
    Returns:
        float: The continuous scalar reward for PPO
    """
    # 1. Gate: Must compile
    if not sandbox_result.get("compiles", False):
        return 0.0
        
    # 2. Gate: Must produce correct outputs
    if not sandbox_result.get("correct", False):
        return 0.0
        
    baseline_ms = sandbox_result.get("baseline_runtime_ms")
    kernel_ms = sandbox_result.get("runtime_ms")
    
    # 3. Validation: Must have valid timing data
    if baseline_ms is None or kernel_ms is None:
        return 0.0
        
    if kernel_ms <= 0:  # Prevent division by zero or negative anomaly
        return 0.0
        
    # 4. Calculation: Speedup = baseline / kernel
    speedup = baseline_ms / kernel_ms
    
    # Cap reward to prevent wild spikes from micro-benchmarks
    reward = min(speedup, max_reward)
    
    return float(reward)
