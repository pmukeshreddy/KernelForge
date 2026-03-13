"""
test_sandbox.py - SP1 Verification: 4 hand-written test cases.

| Test | Kernel                    | Expected                                  |
|------|---------------------------|--------------------------------------------|
| 1    | Valid correct ReLU kernel | compiles=True, correct=True, runtime > 0   |
| 2    | Syntax error kernel       | compiles=False, compiler_error != None     |
| 3    | Compiles but wrong output | compiles=True, correct=False               |
| 4    | Infinite loop kernel      | Timeout, compiles=True or timeout error    |

Run: python3 test_sandbox.py
"""
import sys
import json
from sandbox import evaluate

# ========== Reference model (same for all tests) ==========
REFERENCE = '''
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.relu(x)

def get_inputs():
    return [torch.randn(16, 16384)]

def get_init_inputs():
    return []
'''

# ========== Test 1: Valid correct kernel ==========
KERNEL_CORRECT = '''
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_kernel(float* output, const float* input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

torch::Tensor relu_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int n = input.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(
        output.data_ptr<float>(), input.data_ptr<float>(), n
    );
    return output;
}
"""

cpp_source = "torch::Tensor relu_cuda(torch::Tensor input);"

relu_ext = load_inline(
    name="relu_ext",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["relu_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return relu_ext.relu_cuda(x)
'''

# ========== Test 2: Syntax error kernel ==========
KERNEL_SYNTAX_ERROR = '''
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>

__global__ void relu_kernel(float* output, const float* input  // MISSING CLOSING PAREN
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    output[idx] = input[idx];
}

torch::Tensor relu_cuda(torch::Tensor input) {
    return input;  // won't get here
}
"""

cpp_source = "torch::Tensor relu_cuda(torch::Tensor input);"

relu_ext = load_inline(
    name="relu_broken",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["relu_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return relu_ext.relu_cuda(x)
'''

# ========== Test 3: Compiles but wrong output ==========
KERNEL_WRONG_OUTPUT = '''
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void wrong_kernel(float* output, const float* input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * 2.0f;  // WRONG: doubles instead of ReLU
    }
}

torch::Tensor wrong_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int n = input.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    wrong_kernel<<<blocks, threads>>>(
        output.data_ptr<float>(), input.data_ptr<float>(), n
    );
    return output;
}
"""

cpp_source = "torch::Tensor wrong_cuda(torch::Tensor input);"

wrong_ext = load_inline(
    name="wrong_ext",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["wrong_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return wrong_ext.wrong_cuda(x)
'''

# ========== Test 4: Infinite loop kernel ==========
KERNEL_INFINITE_LOOP = '''
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void loop_kernel(float* output, const float* input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Simulate infinite loop with very long busy-wait
        float val = input[idx];
        for (long long i = 0; i < 999999999999LL; i++) {
            val = val * 1.0001f;
        }
        output[idx] = val;
    }
}

torch::Tensor loop_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int n = input.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    loop_kernel<<<blocks, threads>>>(
        output.data_ptr<float>(), input.data_ptr<float>(), n
    );
    cudaDeviceSynchronize();
    return output;
}
"""

cpp_source = "torch::Tensor loop_cuda(torch::Tensor input);"

loop_ext = load_inline(
    name="loop_ext",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["loop_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return loop_ext.loop_cuda(x)
'''


def run_tests():
    passed = 0
    failed = 0
    
    tests = [
        ("Test 1: Valid correct kernel", KERNEL_CORRECT, 
         lambda r: r["compiles"] == True and r["correct"] == True and r["runtime_ms"] is not None and r["runtime_ms"] > 0),
        
        ("Test 2: Syntax error kernel", KERNEL_SYNTAX_ERROR,
         lambda r: r["compiles"] == False and r["compiler_error"] is not None),
        
        ("Test 3: Compiles but wrong output", KERNEL_WRONG_OUTPUT,
         lambda r: r["compiles"] == True and r["correct"] == False),
        
        ("Test 4: Infinite loop kernel (timeout=10s)", KERNEL_INFINITE_LOOP,
         lambda r: r["correct"] == False),  # Either timeout or very slow
    ]
    
    for name, kernel, check in tests:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}")
        
        timeout = 10 if "Infinite" in name else 30
        result = evaluate(kernel, REFERENCE, timeout=timeout, n_correctness=5, n_warmup=3, n_timed=10)
        
        print(json.dumps(result, indent=2))
        
        ok = check(result)
        if ok:
            print(f"✅ PASSED: {name}")
            passed += 1
        else:
            print(f"❌ FAILED: {name}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Results: {passed}/4 passed, {failed}/4 failed")
    print(f"{'='*60}")
    
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
