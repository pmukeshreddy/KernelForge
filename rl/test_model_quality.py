"""
test_model_quality.py - Tests the SFT model's ability to generate correct CUDA kernels
across different difficulty levels.

Runs the model on 4 test cases of increasing complexity:
  Level 1: Element-wise ReLU (trivial - 1D grid, no shared memory)
  Level 2: Vector Addition (trivial - two inputs)  
  Level 3: Row-wise Softmax (medium - reduction within rows)
  Level 4: Tiled Matrix Multiplication (hard - shared memory tiling)

Usage (run on GPU server):
  python3 test_model_quality.py [--model MODEL_NAME] [--steps N]
"""
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent import KernelForgeAgent

# ============================================================
# Test Cases: Increasing Difficulty
# ============================================================

TESTS = [
    {
        "name": "Level 1: Element-wise ReLU",
        "difficulty": "trivial",
        "pytorch_code": """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.relu(x)

def get_inputs():
    return [torch.randn(16, 16384, device='cuda')]

def get_init_inputs():
    return []
"""
    },
    {
        "name": "Level 2: Element-wise Addition",
        "difficulty": "trivial",
        "pytorch_code": """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, a, b):
        return a + b

def get_inputs():
    return [torch.randn(1024, 1024, device='cuda'), torch.randn(1024, 1024, device='cuda')]

def get_init_inputs():
    return []
"""
    },
    {
        "name": "Level 3: Row-wise Softmax",
        "difficulty": "medium",
        "pytorch_code": """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.softmax(x, dim=-1)

def get_inputs():
    return [torch.randn(128, 4096, device='cuda')]

def get_init_inputs():
    return []
"""
    },
    {
        "name": "Level 4: Matrix Multiplication",
        "difficulty": "hard",
        "pytorch_code": """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, A, B):
        return torch.matmul(A, B)

def get_inputs():
    return [torch.randn(512, 512, device='cuda'), torch.randn(512, 512, device='cuda')]

def get_init_inputs():
    return []
"""
    },
]


def run_quality_test(model_name: str, max_steps: int):
    """Run the model on each test case and report results."""
    print(f"=" * 70)
    print(f"  KernelForge Model Quality Test")
    print(f"  Model: {model_name}")
    print(f"  ReAct Steps: {max_steps}")
    print(f"=" * 70)

    agent = KernelForgeAgent(model_name=model_name)

    results = []
    for i, test in enumerate(TESTS):
        print(f"\n{'=' * 70}")
        print(f"  [{i+1}/{len(TESTS)}] {test['name']} (Difficulty: {test['difficulty']})")
        print(f"{'=' * 70}")

        best_code, reward = agent.run_react_loop(
            test["pytorch_code"],
            max_steps=max_steps
        )

        passed = reward >= 1.0  # At least 1x means correct output
        results.append({
            "name": test["name"],
            "difficulty": test["difficulty"],
            "reward": reward,
            "passed": passed,
        })

        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"\n  Result: {status} | Reward: {reward:.2f}x")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    total_pass = sum(1 for r in results if r["passed"])
    print(f"  Passed: {total_pass}/{len(results)}")
    print()
    for r in results:
        status = "✅" if r["passed"] else "❌"
        print(f"  {status} {r['name']:<40} Reward: {r['reward']:.2f}x")
    print(f"{'=' * 70}")

    # Save results to file
    out_path = "data/model_quality_results.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return total_pass == len(results)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test KernelForge model quality")
    parser.add_argument("--model", type=str, default="mukeshreddy/kernelforge-sft-qwen3-8b")
    parser.add_argument("--steps", type=int, default=5, help="Max ReAct steps per test")
    args = parser.parse_args()

    success = run_quality_test(args.model, args.steps)
    sys.exit(0 if success else 1)
