"""
test_profiler.py - Verification tests for SP4 Profiler Parser.
"""
import sys
from profiler import profile_kernel


def run_tests():
    passed = 0
    failed = 0
    
    # Needs a dummy reference for the profiler script to load
    reference_code = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): return x

def get_inputs():
    # 256MB per tensor (total 512MB for read+write) to blow past A100's 40MB L2 Cache 
    # and guarantee we hit global memory bandwidth limits.
    return [torch.randn(8192, 8192)]

def get_init_inputs():
    return []
"""

    # 1. Memory-Bound Kernel (Copy)
    # Reads and writes memory directly without math.
    code_memory = """
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = \"\"\"
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void copy_kernel(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx];
}

torch::Tensor run_cuda(torch::Tensor in) {
    auto out = torch::empty_like(in);
    int n = in.numel();
    copy_kernel<<<(n+255)/256, 256>>>(out.data_ptr<float>(), in.data_ptr<float>(), n);
    return out;
}
\"\"\"

ext = load_inline(name="ext_mem", cpp_sources="torch::Tensor run_cuda(torch::Tensor in);", cuda_sources=cuda_source, functions=["run_cuda"])

class ModelNew(nn.Module):
    def forward(self, x):
        return ext.run_cuda(x)
"""

    # 2. Compute-Bound Kernel (Math Loop)
    # Does an absurd amount of math registers on a single load.
    code_compute = """
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = \"\"\"
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void math_kernel(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = in[idx];
        #pragma unroll
        for(int i=0; i<5000; ++i) {
            val = sinf(val) * cosf(val);
        }
        out[idx] = val;
    }
}

torch::Tensor run_cuda(torch::Tensor in) {
    auto out = torch::empty_like(in);
    int n = in.numel();
    math_kernel<<<(n+255)/256, 256>>>(out.data_ptr<float>(), in.data_ptr<float>(), n);
    return out;
}
\"\"\"

ext = load_inline(name="ext_comp", cpp_sources="torch::Tensor run_cuda(torch::Tensor in);", cuda_sources=cuda_source, functions=["run_cuda"])

class ModelNew(nn.Module):
    def forward(self, x):
        return ext.run_cuda(x)
"""

    # 3. Low-Occupancy Kernel (Shared Memory Hog)
    # Requests 48KB of layout-breaking shared memory per block.
    code_occupancy = """
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = \"\"\"
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void occ_kernel(float* out, const float* in, int n) {
    // Hog 48KB of shared memory to kill occupancy
    __shared__ float hog[12288]; 
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (threadIdx.x == 0) {
        hog[0] = in[0];
    }
    __syncthreads();
    
    if (idx < n) {
        float val = in[idx] + hog[0];
        // artificial compute to sustain the kernel lifetime 
        // so NCU registers the terrible occupancy accurately
        #pragma unroll
        for(int i=0; i<500; ++i) {
            val = sinf(val) * cosf(val);
        }
        out[idx] = val;
    }
}

torch::Tensor run_cuda(torch::Tensor in) {
    auto out = torch::empty_like(in);
    int n = in.numel();
    occ_kernel<<<(n+255)/256, 256>>>(out.data_ptr<float>(), in.data_ptr<float>(), n);
    return out;
}
\"\"\"

ext = load_inline(name="ext_occ", cpp_sources="torch::Tensor run_cuda(torch::Tensor in);", cuda_sources=cuda_source, functions=["run_cuda"])

class ModelNew(nn.Module):
    def forward(self, x):
        return ext.run_cuda(x)
"""

    tests = [
        ("Memory-Bound Kernel", code_memory, "MEMORY-BOUND"),
        ("Compute-Bound Kernel", code_compute, "COMPUTE-BOUND"),
        ("Low-Occupancy Kernel", code_occupancy, "Warning: Occupancy is very low"),
    ]
    
    print("Running SP4 Profiler Parser Tests (Requires GPU)...\n")
    
    for name, code, expected_keyword in tests:
        print(f"Profiling {name}...")
        
        feedback = profile_kernel(code, reference_code)
        
        if expected_keyword in feedback:
            status = "✅ PASS"
            passed += 1
        else:
            status = "❌ FAIL"
            failed += 1
            
        print(f"{status} | {name}")
        print(f"Output:\n{feedback}\n")
            
    print(f"\nResults: {passed}/{len(tests)} passed, {failed}/{len(tests)} failed")
    
    return failed == 0


if __name__ == "__main__":
    # Test only runs successfully on a system with ncu installed
    import subprocess
    try:
        subprocess.run(["ncu", "--version"], capture_output=True, check=True)
        success = run_tests()
        sys.exit(0 if success else 1)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("Skipping tests: 'ncu' command not available locally.")
        sys.exit(0)
