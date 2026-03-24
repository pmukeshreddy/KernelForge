# CUDA Kernel Best Practices

Reference: [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) | [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

---

## 1. Memory Access — Coalescing

NVIDIA GPUs access global memory in 32-byte or 128-byte cache line transactions. A warp of 32 threads fetches optimally when consecutive threads access consecutive addresses.

**Good — coalesced:**
```cpp
// Thread i reads element i: one transaction per 128-byte block
float val = input[blockIdx.x * blockDim.x + threadIdx.x];
```

**Bad — strided:**
```cpp
// Each thread jumps N elements: N separate transactions
float val = input[threadIdx.x * N];
```

Rules:
- Prefer Structure-of-Arrays (SoA) over Array-of-Structures (AoS).
- Align buffers to 128 bytes (PyTorch tensors are aligned by default).
- Use vector loads (`float4`, `int4`) to widen transactions: 128-bit loads = 4× fewer instructions.
- Use `__ldg(&ptr[i])` for read-only data — routes through texture cache, avoids L1 pollution.

```cpp
// Vectorized load: 4 floats at once (128-bit transaction)
float4 in4 = reinterpret_cast<const float4*>(input)[idx];
float4 out4;
out4.x = f(in4.x); out4.y = f(in4.y);
out4.z = f(in4.z); out4.w = f(in4.w);
reinterpret_cast<float4*>(output)[idx] = out4;
// Adjust grid: blocks = (n/4 + threads - 1) / threads
// Handle remainder if n % 4 != 0
```

---

## 2. Occupancy and Warp Management

Each NVIDIA SM schedules multiple warps (32 threads each). High occupancy hides memory latency by switching between warps while one waits for data.

**Thread/block sizing:**
```cpp
// Use __launch_bounds__ to guide the compiler
__global__ void __launch_bounds__(256, 4) myKernel(...) {
    // 256 threads/block, min 4 blocks per SM
}
```

**Key occupancy limits (A100/H100):**
| Resource per SM | A100 | H100 |
|----------------|------|------|
| Max threads     | 2048 | 2048 |
| Max warps       | 64   | 64   |
| Max blocks      | 32   | 32   |
| Registers/SM    | 65536| 65536|
| Shared mem/SM   | 164KB| 228KB|

Rules:
- Block size should be a multiple of 32 (warp size). Prefer 128 or 256.
- Fewer registers per thread = more warps can run concurrently.
- Use `cudaOccupancyMaxPotentialBlockSize()` to auto-tune block size.

---

## 3. Shared Memory

Shared memory provides ~100× faster bandwidth than global memory. Each SM has 48-228 KB (configurable vs L1 cache).

**Tiled GEMM example:**
```cpp
__global__ void tiled_gemm(const float* A, const float* B, float* C,
                           int M, int N, int K) {
    constexpr int TILE = 16;
    __shared__ float As[TILE][TILE + 1]; // +1 avoids bank conflicts
    __shared__ float Bs[TILE][TILE + 1];

    int tx = threadIdx.x, ty = threadIdx.y;
    float acc = 0.f;

    for (int t = 0; t < K / TILE; ++t) {
        As[ty][tx] = A[(blockIdx.y * TILE + ty) * K + t * TILE + tx];
        Bs[ty][tx] = B[(t * TILE + ty) * N + blockIdx.x * TILE + tx];
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    C[(blockIdx.y * TILE + ty) * N + blockIdx.x * TILE + tx] = acc;
}
```

**Bank conflicts:** Shared memory has 32 banks (4 bytes each). Threads in a warp accessing the same bank serialize. Pad arrays by 1 element to avoid:
```cpp
__shared__ float tile[TILE][TILE + 1]; // +1 avoids 32-way conflict
```

**Dynamic shared memory** (when size isn't known at compile time):
```cpp
extern __shared__ float smem[];
kernel<<<grid, block, shared_bytes>>>(...)
```

---

## 4. Parallel Reductions (sum, min, max, mean, argmin, argmax, prod)

Reductions (sum, max, min, mean, prod, argmin, argmax, norm) over arrays are the most common source of incorrect CUDA kernels. The pattern depends on the array size.

### Warp-level reduction (≤ 32 elements)
```cpp
// No shared memory needed — warp shuffle is fastest
float val = thread_value;
for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xffffffff, val, offset);
// Thread 0 of the warp now has the sum
```

### Block-level reduction (≤ 1024 elements)
```cpp
// Step 1: each warp reduces internally via shuffle
float val = thread_value;
for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xffffffff, val, offset);

// Step 2: warp leaders write to shared memory
__shared__ float warp_sums[32]; // max 32 warps per block
int lane = threadIdx.x % 32;
int warp_id = threadIdx.x / 32;
if (lane == 0) warp_sums[warp_id] = val;
__syncthreads();

// Step 3: first warp reduces the warp sums
if (warp_id == 0) {
    val = (lane < blockDim.x / 32) ? warp_sums[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
}
// Thread 0 of the block now has the block sum
```

### Grid-level reduction (> 1024 elements — MOST COMMON)
```cpp
// CRITICAL: when dim > blockDim.x, a single block CANNOT cover all elements.
// Use a grid-stride loop to accumulate locally, then reduce per-block.
__global__ void reduce_sum(const float* input, float* output, int n) {
    float local_sum = 0.0f;

    // Grid-stride loop: each thread accumulates multiple elements
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {
        local_sum += input[i];
    }

    // Block-level reduction (warp shuffle + shared memory)
    // ... (same as block-level pattern above)

    // Thread 0 of each block writes its partial sum
    if (threadIdx.x == 0)
        atomicAdd(output, block_sum);
}
```

**Common mistakes:**
- ❌ `__shared__ float arr[1024]` when dim=16384 — only processes first 1024 elements
- ❌ Each thread computes its own `row_max` independently — no cross-thread communication
- ❌ Multiple threads write to the same shared variable without atomics or synchronization
- ❌ Using `cudaMalloc`/`cudaMemcpy` for intermediate results — use `torch::zeros({1}, input.options())` to keep data on GPU
- ❌ `std::vector` or `std::` containers in `__global__` functions — not available in device code

### Reducing along a dimension (keepdim pattern)
When reducing along one dimension (e.g., `torch.min(x, dim=1)`), the output must **preserve all other dimensions**:
```cpp
// Reduce dim=1 of input [B, D, N] → output [B, N]
// Each thread handles one output element
__global__ void min_along_dim1(const float* input, float* output,
                                int B, int D, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * N) return;
    int b = idx / N;
    int n = idx % N;

    float min_val = INFINITY;
    for (int d = 0; d < D; ++d) {
        float val = input[b * D * N + d * N + n];
        if (val < min_val) min_val = val;
    }
    output[b * N + n] = min_val;
}
// Output shape = input shape with reduced dim removed.
// Use input.size(0), input.size(1), etc. to get dims — NOT .sizes().
// Reshape output: torch::empty({B, N}, input.options())
```

---

## 5. Softmax / LogSoftmax Pattern

Softmax requires two reductions (max, then sum of exp) over the same data. This is the canonical "hard reduction" pattern.

```cpp
// Per-row softmax: input [batch, dim], output [batch, dim]
__global__ void softmax_kernel(const float* input, float* output,
                               int batch, int dim) {
    int row = blockIdx.x;  // one block per row
    if (row >= batch) return;

    const float* row_in = input + row * dim;
    float* row_out = output + row * dim;

    // Pass 1: find row max (for numerical stability)
    float thread_max = -INFINITY;
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        thread_max = fmaxf(thread_max, row_in[i]);

    // Block-wide max reduction
    // ... warp shuffle + shared memory reduction for max ...

    // Pass 2: compute sum of exp(x - max)
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        thread_sum += expf(row_in[i] - row_max);

    // Block-wide sum reduction
    // ... warp shuffle + shared memory reduction for sum ...

    // Pass 3: write normalized output
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        row_out[i] = expf(row_in[i] - row_max) / row_sum;
}
// Launch: softmax_kernel<<<batch, 256>>>(...)
```

Key points:
- One block per row (batch element) — threads within the block cooperate on the reduction.
- Use grid-stride loop (`for i += blockDim.x`) so any dim works, regardless of block size.
- Two reductions: max (numerical stability), then sum (normalization).
- Three passes over the data — can be reduced to two with online softmax.

---

## 6. Normalization Patterns (LayerNorm, RMSNorm, L1/L2 Norm)

All normalization operations follow: reduce → normalize.

```cpp
// L2 Norm (Frobenius): ||x|| = sqrt(sum(x^2))
// normalize: y = x / ||x||
__global__ void l2_norm_kernel(const float* input, float* output, int n) {
    float thread_sum_sq = 0.0f;

    // Grid-stride accumulation
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += blockDim.x * gridDim.x) {
        float val = input[i];
        thread_sum_sq += val * val;
    }

    // Block-level reduction for sum of squares
    // ... warp shuffle + shared memory ...

    // Store block partial sum to global (atomicAdd)
    // Then second kernel: output[i] = input[i] / sqrt(total_sum_sq)
}
```

**Do NOT** use `cudaMalloc`/`cudaMemcpy` for the intermediate sum. Use:
```cpp
auto sum_tensor = torch::zeros({1}, input.options());  // stays on GPU
atomicAdd(sum_tensor.data_ptr<float>(), block_sum);
```

---

## 7. Elementwise Operations (Activation, Loss, Pointwise)

Covers: GELU, ReLU, SELU, Sigmoid, Tanh, Swish/SiLU, Mish, Softplus,
smooth_l1_loss, mse_loss, l1_loss, huber_loss, and any per-element function.

Simplest pattern — map one operation per element. Perfect for kernel fusion.

```cpp
// Smooth L1 / Huber loss: elementwise then reduce to scalar
__global__ void smooth_l1_kernel(const float* pred, const float* target,
                                  float* per_elem, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float diff = pred[idx] - target[idx];
        float abs_diff = fabsf(diff);
        per_elem[idx] = (abs_diff < 1.0f) ? 0.5f * diff * diff : abs_diff - 0.5f;
    }
}
// Host: auto per_elem = torch::empty_like(pred);
//       smooth_l1_kernel<<<blocks, 256>>>(pred_ptr, tgt_ptr, per_elem_ptr, n);
//       auto loss = per_elem.mean();  // reduce on PyTorch side (fastest)

// GELU: y = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
__global__ void gelu_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        output[idx] = x * cdf;
    }
}
```

**For loss functions that reduce to scalar**: compute per-element values in the kernel,
then call `torch::mean()` or `torch::sum()` on the output tensor. Do NOT try to reduce
inside the kernel with atomicAdd to a single float — use PyTorch's built-in reduction.

Optimizations:
- **Vectorize** with `float4`: process 4 elements per thread.
- **Grid-stride loop**: launch fewer blocks, each thread processes multiple elements.
- **Fuse** adjacent elementwise ops into one kernel to eliminate intermediate memory traffic.
- **`__ldg()`**: for read-only input, use texture cache path.

---

## 8. Convolution / Sliding Window

For conv2d, max_pool2d, and similar sliding-window operations:

```cpp
// MaxPool2D: each thread computes one output element
__global__ void maxpool2d_kernel(
    const float* input, float* output,
    int batch, int channels, int H, int W,
    int outH, int outW, int kH, int kW,
    int strideH, int strideW, int padH, int padW) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * outH * outW;
    if (idx >= total) return;

    // Decompose linear index to (b, c, oh, ow)
    int ow = idx % outW;
    int oh = (idx / outW) % outH;
    int c  = (idx / (outW * outH)) % channels;
    int b  = idx / (outW * outH * channels);

    float max_val = -INFINITY;
    for (int kh = 0; kh < kH; ++kh) {
        for (int kw = 0; kw < kW; ++kw) {
            int ih = oh * strideH - padH + kh;
            int iw = ow * strideW - padW + kw;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                max_val = fmaxf(max_val, input[((b * channels + c) * H + ih) * W + iw]);
            }
        }
    }
    output[((b * channels + c) * outH + oh) * outW + ow] = max_val;
}
```

**Output shape formula:**
```
outH = (H + 2*padH - kH) / strideH + 1
outW = (W + 2*padW - kW) / strideW + 1
```

Common mistakes:
- ❌ Wrong output shape: forgetting padding, dilation, or using ceil instead of floor division.
- ❌ Not handling padding boundaries (reading out-of-bounds input).

---

## 9. Cross-Entropy Loss

Combines log-softmax + NLL loss. The output is a **scalar** (mean over batch).

```cpp
// Step 1: per-sample loss via log-softmax + NLL (one block per sample)
__global__ void cross_entropy_kernel(
    const float* logits,   // [batch, num_classes]
    const long* targets,   // [batch]
    float* losses,         // [batch] per-sample losses
    int batch, int num_classes) {

    int b = blockIdx.x;
    if (b >= batch) return;

    const float* logit_row = logits + b * num_classes;

    // Compute log-softmax for the target class
    // (same pattern as softmax: max-reduce, sum-reduce)
    float row_max = -INFINITY;
    for (int i = threadIdx.x; i < num_classes; i += blockDim.x)
        row_max = fmaxf(row_max, logit_row[i]);
    // ... block reduction for max ...

    float sum_exp = 0.0f;
    for (int i = threadIdx.x; i < num_classes; i += blockDim.x)
        sum_exp += expf(logit_row[i] - row_max);
    // ... block reduction for sum ...

    if (threadIdx.x == 0) {
        long target = targets[b];
        float log_softmax = logit_row[target] - row_max - logf(sum_exp);
        losses[b] = -log_softmax;
    }
}

// Step 2: mean reduction over batch → scalar output
// output = torch::mean(per_sample_losses)
```

**CRITICAL**: `torch.nn.functional.cross_entropy` returns a **scalar** (shape `[]`), not per-sample losses. Your output shape must match.

---

## 10. PyTorch C++ Extension API

When writing CUDA kernels for PyTorch's `load_inline`:

```cpp
#include <torch/extension.h>
#include <cuda_runtime.h>

// Tensor allocation — ALWAYS on same device as input
auto output = torch::empty_like(input);                              // same shape/dtype/device
auto output = torch::zeros({M, N}, input.options());                 // custom shape, same device
auto output = torch::empty({M}, torch::TensorOptions()
    .dtype(torch::kFloat32).device(input.device()));                 // explicit

// Getting data pointers
float* ptr = input.data_ptr<float>();         // NOT .ptr<float>()
const int64_t* sizes = input.sizes().data();  // for passing to kernels

// Getting dimensions
int batch = input.size(0);
int dim   = input.size(1);
int n     = input.numel();

// Launching kernels
int threads = 256;
int blocks = (n + threads - 1) / threads;
my_kernel<<<blocks, threads>>>(ptr, out_ptr, n);

// DON'T use:
// - .size() without args → use .sizes() or .size(dim)
// - new float / malloc for kernel outputs → use torch::zeros({}, input.options())
// - cudaMalloc/cudaMemcpy for tensor operations (use torch:: allocators)
// - std::vector in __global__ functions (not available in device code)
// - torch::ScalarTensor (doesn't exist)
// - .type() (use .scalar_type())
// - __host__ or __device__ on binding functions
// - PYBIND11_MODULE with load_inline (it generates bindings automatically)
```

---

## 11. Common Pitfalls

| Mistake | Fix |
|---------|-----|
| `std::vector` in `__global__` | Use C arrays or pass as kernel args |
| `std::max/min/abs` in device code | Use `fmaxf/fminf/fabsf` |
| `std::numeric_limits<float>::infinity()` | Use `INFINITY` or `-INFINITY` |
| `__shfl_down()` without `_sync` | Use `__shfl_down_sync(0xffffffff, ...)` |
| `cudaMalloc` for intermediate tensors | Use `torch::zeros({}, input.options())` |
| `tensor.type()` | Use `tensor.scalar_type()` |
| `.ptr<T>()` | Use `.data_ptr<T>()` |
| `auto [a, b] = sizes()` (C++17 structured bindings) | Not supported by nvcc; index manually |
| Output tensor on CPU | Always use `input.options()` or `.device(torch::kCUDA)` |
| `tensor.size()` (no args) | Use `tensor.sizes()` — `.size()` needs a dim arg: `.size(0)` |
| `new float(0.0f)` passed to kernel | Host memory! Use `torch::zeros({1}, input.options())` |
| `atomicAdd` to single scalar | Compute per-element, reduce with `torch::mean()` after kernel |
| Hardcoded shared memory size | Use `extern __shared__` or grid-stride loop |
| `__host__ __device__` on binding functions | Binding functions must be plain host functions |

---

## 12. Quick Checklist

- [ ] Access pattern is coalesced (consecutive threads → consecutive addresses)
- [ ] Block size is a multiple of 32 (warp size), prefer 128 or 256
- [ ] Shared memory arrays padded by 1 to avoid bank conflicts
- [ ] Reductions use warp shuffle (`__shfl_down_sync`) + shared memory, NOT per-thread independent computation
- [ ] Grid-stride loop handles dim > blockDim.x
- [ ] Output tensors allocated on GPU (same device as input)
- [ ] No `std::` containers or functions in device code
- [ ] Output shape matches reference PyTorch operation exactly
- [ ] Boundary checks (`if idx < n`) prevent out-of-bounds access
- [ ] No `cudaMalloc`/`cudaMemcpy` — use PyTorch tensor API
- [ ] Loop unrolling with `#pragma unroll` on small fixed loops
- [ ] `__ldg()` for read-only inputs
