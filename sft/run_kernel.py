import torch
from torch.utils.cpp_extension import load_inline

cuda_code = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* __restrict__ x, float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = x[idx] > 0 ? x[idx] : 0;
    }
}

torch::Tensor forward(torch::Tensor x) {
    auto out = torch::empty_like(x);
    int n = x.numel();
    relu_kernel<<<(n + 255) / 256, 256>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), n
    );
    return out;
}
"""

cpp_code = 'torch::Tensor forward(torch::Tensor x);'

ext = load_inline(name="relu_cuda", cuda_sources=cuda_code, cpp_sources=cpp_code,
                  functions=["forward"], verbose=False)

x = torch.randn(1024*1024, device="cuda")

# Correctness
out_cuda = ext.forward(x)
out_ref  = torch.relu(x)
print(f"Correct: {torch.allclose(out_cuda, out_ref)}")

# Speed
import time
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(1000): ext.forward(x)
torch.cuda.synchronize()
cuda_ms = (time.perf_counter() - t0)

torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(1000): torch.relu(x)
torch.cuda.synchronize()
torch_ms = (time.perf_counter() - t0)

print(f"Custom CUDA: {cuda_ms*1000:.2f}ms total")
print(f"torch.relu:  {torch_ms*1000:.2f}ms total")
print(f"Speedup: {torch_ms/cuda_ms:.2f}x")
