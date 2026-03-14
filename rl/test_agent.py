"""
test_agent.py - Verifies the KernelForgeAgent ReAct loop state transitions.
"""
import sys
from unittest.mock import patch, MagicMock

# Mock transformers before agent is imported so it runs locally without heavy installs
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent import KernelForgeAgent

TARGET_CODE = "import torch\nimport torch.nn as nn\nclass Model(nn.Module):\n    def __init__(self): super().__init__()\n    def forward(self, x): return torch.relu(x)\ndef get_inputs(): return [torch.randn(1024, 1024, device='cuda')]\ndef get_init_inputs(): return []\n"

MOCK_RESPONSE_1 = "Here is my first attempt:\n```python\nimport torch\nfrom torch.utils.cpp_extension import load_inline\ncuda_source = \"\"\"\n__global__ void relu_k(float* out, const float* in, int n) {\n    int i = blockIdx.x*blockDim.x+threadIdx.x;\n    if (i < n) out[i] = fmaxf(0.f, in[i])\n}\n\"\"\"\n```"

MOCK_RESPONSE_2 = "Fixed version:\n```python\nimport torch\nimport torch.nn as nn\nfrom torch.utils.cpp_extension import load_inline\ncuda_source = \"__global__ void relu_k(float* out, const float* in, int n) { int i = blockIdx.x*blockDim.x+threadIdx.x; if (i < n) out[i] = fmaxf(0.f, in[i]); }\"\ntorch::Tensor run_cuda(torch::Tensor in) { auto out = torch::empty_like(in); int n = in.numel(); relu_k<<<(n+255)/256, 256>>>(out.data_ptr<float>(), in.data_ptr<float>(), n); return out; }\next = load_inline('test_relu', 'torch::Tensor run_cuda(torch::Tensor in);', cuda_source, ['run_cuda'])\nclass ModelNew(nn.Module):\n    def __init__(self): super().__init__()\n    def forward(self, x): return ext.run_cuda(x)\ndef get_inputs(): return [torch.randn(1024, 1024, device='cuda')]\ndef get_init_inputs(): return []\n```"

class MockAgent(KernelForgeAgent):
    def __init__(self):
        self.system_prompt = "MOCK SYSTEM PROMPT"
        self.call_count = 0
        
    def generate(self, messages):
        self.call_count += 1
        if self.call_count == 1:
            return MOCK_RESPONSE_1
        else:
            assert "Your code failed during evaluation" in messages[-1]["content"] or "Error Log" in messages[-1]["content"]
            return MOCK_RESPONSE_2

def test_react_loop():
    print("Running SP6 Mock ReAct Loop Test...")
    agent = MockAgent()
    with patch("agent.evaluate") as mock_eval:
        with patch("agent.profile_kernel") as mock_prof:
            with patch("agent.calculate_reward") as mock_rew:
                mock_eval.side_effect = [
                    {"correct": False, "compiler_error": "error: expected ';' before '}' token"},
                    {"correct": True, "runtime_ms": 0.5, "baseline_ms": 1.0}
                ]
                mock_prof.return_value = "Bottleneck: MEMORY-BOUND"
                mock_rew.return_value = 2.0
                
                best_code, reward = agent.run_react_loop(TARGET_CODE, max_steps=2)
                
                assert reward == 2.0
                assert "load_inline" in best_code
                assert agent.call_count == 2
                print("✅ PASS | ReAct State Machine processed Syntax Error -> Fix -> Reward")
                return True

if __name__ == "__main__":
    if test_react_loop():
        sys.exit(0)
    sys.exit(1)
