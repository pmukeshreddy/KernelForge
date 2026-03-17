---
name: CUDA Optimization
description: Expert guidelines for writing high-performance CUDA kernels for NVIDIA A100 GPUs within PyTorch Extension sandbox constraints.
---

# CUDA Optimization Playbook (A100 Architecture)

When writing or rewriting CUDA kernels inside the KernelForge framework, strictly follow these constraints and optimization heuristics based on the bottlenecks reported by Nsight Compute (`ncu`).

## 1. Environment and Constraints
- The target hardware is an NVIDIA A100 GPU (Compute Capability 8.0, 40MB L2 Cache, 108 SMs).
- Your code will be executed in a strict PyTorch `load_inline` sandbox.
- You MUST only write valid PyTorch C++ Extension code (`<torch/extension.h>`) and standard CUDA.
- You CANNOT use `torch::nn::functional` or pre-built PyTorch ATen kernels inside your C++ code. You must write the actual custom CUDA `__global__ void` kernel to achieve speedups.
- You CANNOT use external libraries like cuBLAS, cuDNN, or CUTLASS.
- You CANNOT use any form of `subprocess`, `os.system`, or file I/O.

## 2. Resolving Bottlenecks

### A. Memory-Bound
If Memory Throughput > ~70% of peak, the SMs are starving for data.
- **Coalescing**: Ensure threads in a warp access consecutive, aligned memory addresses (e.g., `data[blockIdx.x * blockDim.x + threadIdx.x]`). STRIDED ACCESS KILLS PERFORMANCE.
- **Vectorization**: Use `float4` or `double2` to load 128 bits of data per thread in a single instruction. This halves the number of memory transactions.
  ```cpp
  float4 val = reinterpret_cast<const float4*>(in)[idx];
  ```
- **Shared Memory Caching**: If multiple threads read the same data, load it into `__shared__` memory once per block, `__syncthreads()`, and have threads read from the fast shared memory.
- **L2 Cache Reuse**: Re-order loops (blocking/tiling) to keep working sets inside the 40MB L2 cache.

### B. Compute-Bound
If Compute Throughput > ~70% of peak, the FP32/FP64 mathematical units are saturated.
- **Fast Math**: Use intrinsic math functions (`__sinf`, `__expf`, `__fdividef`) instead of standard math functions.
- **Loop Unrolling**: Add `#pragma unroll` before fixed-size inner loops to remove branch overhead and expose Instruction-Level Parallelism (ILP).
- **Branch Divergence**: Avoid `if` statements that cause threads within the same warp (32 threads) to take different paths. If necessary, use `__shfl_sync` or compute both paths and blend if it's cheap.

### C. Latency-Bound (Poor Occupancy)
If Warp Occupancy is low (< 40%), the SM doesn't have enough active warps to hide memory/instruction latency.
- **Register Pressure**: If a kernel uses > 64 registers per thread, fewer blocks can fit on the SM. Simplify local variables, or force limits via `__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)`.
- **Shared Memory Limits**: The A100 has 164KB of shared memory per SM. If your block uses 48KB, only 3 blocks can fit (144KB). Reduce shared memory size, or increase the work done per block.
- **Block Sizing**: Always use block sizes that are multiples of 32 (preferably 128, 256, or 512). Very small blocks (e.g., 32 threads) waste SM resources due to block allocation overhead.
- **Excessive Synchronization**: Avoid `__syncthreads()` inside heavy loops unless absolutely necessary for data correctness.

## 3. Formatting
You must wrap your solution in a `ModelNew` class that strictly inherits from `torch.nn.Module`.
```python
# Compile
ext = load_inline(
    name="custom_ext",
    cpp_sources="torch::Tensor run_cuda(torch::Tensor input);",
    cuda_sources=cuda_source,
    functions=["run_cuda"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "-use_fast_math", "-Xcompiler", "-O3"] # Maximize math throughput
)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return ext.run_cuda(x)
```
