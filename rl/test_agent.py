"""
test_agent.py - Verifies the KernelForgeAgent ReAct loop state transitions,
including the new C++ extraction and load_inline wrapping flow.
"""
import sys
from unittest.mock import patch, MagicMock

import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent import KernelForgeAgent, build_load_inline_wrapper

TARGET_CODE = """import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.relu(x)

def get_inputs():
    return [torch.randn(1024, 1024, device='cuda')]

def get_init_inputs():
    return []
"""

# Step 1: Model outputs C++ with a syntax error (missing semicolon)
MOCK_RESPONSE_1 = """
#include <cuda_runtime.h>

__global__ void relu_k(float* out, const float* in, int n) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < n) out[i] = fmaxf(0.f, in[i])
}

torch::Tensor run_cuda(torch::Tensor input) {
    auto out = torch::empty_like(input);
    int n = input.numel();
    relu_k<<<(n+255)/256, 256>>>(out.data_ptr<float>(), input.data_ptr<float>(), n);
    return out;
}
```"""

# Step 2: Model fixes the syntax error
MOCK_RESPONSE_2 = """
#include <cuda_runtime.h>

__global__ void relu_k(float* out, const float* in, int n) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < n) out[i] = fmaxf(0.f, in[i]);
}

torch::Tensor run_cuda(torch::Tensor input) {
    auto out = torch::empty_like(input);
    int n = input.numel();
    relu_k<<<(n+255)/256, 256>>>(out.data_ptr<float>(), input.data_ptr<float>(), n);
    return out;
}
```"""


class MockAgent(KernelForgeAgent):
    def __init__(self):
        self.system_prompt = "MOCK SYSTEM PROMPT"
        self.call_count = 0
        
    def generate(self, messages):
        self.call_count += 1
        if self.call_count == 1:
            return MOCK_RESPONSE_1
        else:
            # Verify error feedback was sent
            last_msg = messages[-1]["content"]
            assert "failed" in last_msg.lower() or "error" in last_msg.lower(), \
                f"Expected error feedback, got: {last_msg[:100]}"
            return MOCK_RESPONSE_2


def test_wrapper_builder():
    """Test that build_load_inline_wrapper correctly parses C++ and builds Python."""
    print("Running wrapper builder test...")
    
    cuda_code = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void my_kernel(float* out, const float* in, int n) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < n) out[i] = fmaxf(0.f, in[i]);
}

torch::Tensor run_cuda(torch::Tensor input) {
    auto out = torch::empty_like(input);
    int n = input.numel();
    my_kernel<<<(n+255)/256, 256>>>(out.data_ptr<float>(), input.data_ptr<float>(), n);
    return out;
}
"""
    ref_code = """
class Model(nn.Module):
    def forward(self, x):
        return torch.relu(x)
"""
    
    wrapper = build_load_inline_wrapper(cuda_code, ref_code)
    assert wrapper is not None, "Wrapper should not be None"
    assert "load_inline" in wrapper, "Wrapper must contain load_inline"
    assert "ModelNew" in wrapper, "Wrapper must contain ModelNew"
    assert 'functions=["run_cuda"]' in wrapper or "functions=['run_cuda']" in wrapper, \
        f"Wrapper must list run_cuda as function, got: {wrapper}"
    assert "def forward(self, x)" in wrapper, f"Wrapper must have forward(self, x), got: {wrapper}"
    assert "ext.run_cuda(x)" in wrapper, f"Wrapper must call ext.run_cuda(x), got: {wrapper}"
    
    print("✅ PASS | Wrapper builder correctly parses C++ and builds Python wrapper")
    return True


def test_wrapper_multi_arg():
    """Test wrapper builder with multi-argument forward()."""
    print("Running multi-arg wrapper test...")
    
    cuda_code = """
#include <torch/extension.h>
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    return torch::zeros({A.size(0), B.size(1)}, A.options());
}
"""
    ref_code = """
class Model(nn.Module):
    def forward(self, A, B):
        return torch.matmul(A, B)
"""
    wrapper = build_load_inline_wrapper(cuda_code, ref_code)
    assert wrapper is not None
    assert "def forward(self, A, B)" in wrapper
    assert "ext.matmul_cuda(A, B)" in wrapper
    print("✅ PASS | Multi-argument forward() correctly parsed")
    return True


def test_react_loop():
    """Test the full ReAct loop with mock agent."""
    print("Running ReAct loop test...")
    agent = MockAgent()
    with patch("agent.evaluate") as mock_eval:
        with patch("agent.profile_kernel") as mock_prof:
            with patch("agent.calculate_reward") as mock_rew:
                mock_eval.side_effect = [
                    {"correct": False, "compiler_error": "error: expected ';' before '}'"},
                    {"correct": True, "runtime_ms": 0.5, "baseline_runtime_ms": 1.0}
                ]
                mock_prof.return_value = "Bottleneck: MEMORY-BOUND"
                mock_rew.return_value = 2.0
                
                best_code, reward = agent.run_react_loop(TARGET_CODE, max_steps=2)
                
                assert reward == 2.0, f"Expected reward 2.0, got {reward}"
                assert "load_inline" in best_code, f"Best code should contain load_inline wrapper"
                assert "ModelNew" in best_code, f"Best code should contain ModelNew class"
                assert agent.call_count == 2, f"Expected 2 generate calls, got {agent.call_count}"
                print("✅ PASS | ReAct loop: Syntax Error → Fix → Wrapper → Reward")
                return True


if __name__ == "__main__":
    results = []
    results.append(test_wrapper_builder())
    results.append(test_wrapper_multi_arg())
    results.append(test_react_loop())
    
    passed = sum(results)
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{len(results)} tests passed")
    print(f"{'='*50}")
    sys.exit(0 if all(results) else 1)
