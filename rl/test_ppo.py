"""
test_ppo.py - Validates the PPO / GRPO reward function bindings.

This ensures the reward parser correctly translates LLM output
strings into float speedups using our sandbox.
"""

import sys
from unittest.mock import patch, MagicMock

sys.modules['datasets'] = MagicMock()
sys.modules['trl'] = MagicMock()
sys.modules['torch'] = MagicMock()

# Mock the agent initialization so the reward parser doesn't try to load HF models
import agent
agent.KernelForgeAgent = MagicMock()
agent_instance = MagicMock()
agent_instance.extract_code_block.side_effect = lambda x: x.split("```python")[1].split("```")[0].strip() if "```python" in x else None
agent.KernelForgeAgent.return_value = agent_instance

from train_ppo import kernel_reward_func

def test_reward_func_bindings():
    print("Testing GRPO Reward Function Parsing...")

    # Mock the LLM outputting a perfect, already-optimized kernel
    mock_good_completion = """
    ```python
    import torch
    from torch.utils.cpp_extension import load_inline
    cuda = "int i = blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i] = fmaxf(0.f, in[i]);"
    ext = load_inline("t1", "torch::Tensor r(torch::Tensor i);", cuda, ["r"])
    class ModelNew(torch.nn.Module):
        def forward(self, x): return ext.r(x)
    ```
    """
    
    # Mock LLM creating a syntax error or hallucinating format
    mock_bad_completion = "I am an AI, I cannot write code."
    
    dummy_prompt = "import torch\ndef get_inputs(): return [torch.randn(10, device='cuda')]"

    print("\nCase 1: Bad Formatting (Format Collapse)")
    rewards = kernel_reward_func([dummy_prompt], [mock_bad_completion])
    assert rewards[0] == -1.0, f"Expected -1.0 penalty, got {rewards[0]}"
    print("✅ Handled format collapse correctly")

    print("\nCase 2: Good formatting, but evaluates incorrectly (Sandbox Failure)")
    with __import__('unittest').mock.patch("sandbox.evaluate") as mock_eval:
        mock_eval.return_value = {"correct": False, "compiler_error": "Syntax Error"}
        rewards = kernel_reward_func([dummy_prompt], [mock_good_completion])
        assert rewards[0] == -0.5, f"Expected -0.5 penalty, got {rewards[0]}"
        print("✅ Handled sandbox failure correctly")

    print("\nCase 3: Good formatting, compiles, gets 2.5x speedup")
    with __import__('unittest').mock.patch("sandbox.evaluate") as mock_eval:
        with __import__('unittest').mock.patch("reward.calculate_reward") as mock_rew:
            mock_eval.return_value = {"correct": True}
            mock_rew.return_value = 2.5
            
            rewards = kernel_reward_func([dummy_prompt], [mock_good_completion])
            assert rewards[0] == 2.5, f"Expected 2.5x reward, got {rewards[0]}"
            print("✅ Handled speedup reward correctly")

    print("\n✅ PASS | PPO Reward Function integration verified")

if __name__ == "__main__":
    test_reward_func_bindings()
