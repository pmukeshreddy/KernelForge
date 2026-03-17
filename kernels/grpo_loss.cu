/*
 * grpo_loss.cu - Fused GRPO + DAPO Clip-Higher Loss Kernel
 *
 * Replaces the multi-op PyTorch loss computation in train_grpo.py with a
 * single fused CUDA kernel. Reduces memory bandwidth by computing ratio,
 * asymmetric clipping, advantage multiplication, and token masking in one pass.
 *
 * PyTorch equivalent (what this replaces):
 *   log_ratio = new_logp - old_logp
 *   ratio = torch.exp(log_ratio)
 *   clipped = torch.clamp(ratio, 1 - clip_low, 1 + clip_high)  # asymmetric
 *   loss = -torch.min(ratio * adv, clipped * adv)
 *   loss = (loss * mask).sum() / mask.sum()
 *
 * Usage (via load_inline or torch.utils.cpp_extension.load):
 *   from torch.utils.cpp_extension import load
 *   grpo_ext = load(name="grpo_loss", sources=["kernels/grpo_loss.cu"])
 *   loss = grpo_ext.grpo_loss(new_logp, old_logp, advantages, mask, 0.2, 0.28)
 */

#include <torch/extension.h>
#include <cuda_runtime.h>


/*
 * Per-token fused loss kernel.
 *
 * For each token position (b, t):
 *   ratio       = exp(new_logp[b,t] - old_logp[b,t])
 *   clip_lo     = 1.0 - clip_low
 *   clip_hi     = 1.0 + clip_high  (if adv >= 0, DAPO allows higher upside)
 *                 1.0 + clip_low   (if adv <  0, symmetric on downside)
 *   clipped     = clamp(ratio, clip_lo, clip_hi)
 *   token_loss  = -min(ratio * adv, clipped * adv) * mask[b,t]
 */
__global__ void grpo_dapo_loss_kernel(
    const float* __restrict__ new_logp,    // (B, T)
    const float* __restrict__ old_logp,    // (B, T)
    const float* __restrict__ advantages,  // (B,)
    const float* __restrict__ mask,        // (B, T)
    float* __restrict__ loss_per_token,    // (B, T) output
    int B, int T,
    float clip_low,
    float clip_high
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * T) return;

    int b = idx / T;

    float m = mask[idx];
    if (m == 0.0f) {
        loss_per_token[idx] = 0.0f;
        return;
    }

    float adv        = advantages[b];
    float log_ratio  = new_logp[idx] - old_logp[idx];
    float ratio      = __expf(log_ratio);

    // DAPO Clip-Higher: asymmetric upper bound
    float clip_hi = (adv >= 0.0f) ? (1.0f + clip_high) : (1.0f + clip_low);
    float clip_lo = 1.0f - clip_low;

    float clipped = fmaxf(clip_lo, fminf(clip_hi, ratio));

    // Pessimistic PPO bound
    loss_per_token[idx] = -fminf(ratio * adv, clipped * adv);
}


/*
 * Parallel reduction: sum loss_per_token and mask over N elements.
 * Uses shared memory reduction — two arrays packed into one extern shared block.
 */
__global__ void masked_sum_kernel(
    const float* __restrict__ loss_per_token,  // (N,)
    const float* __restrict__ mask,            // (N,)
    float* __restrict__ out_loss,              // scalar accumulator
    float* __restrict__ out_count,             // scalar accumulator
    int N
) {
    extern __shared__ float sdata[];
    float* sloss  = sdata;
    float* scount = sdata + blockDim.x;

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    float local_loss  = 0.0f;
    float local_count = 0.0f;

    // Grid-stride loop: each thread accumulates multiple elements
    while (idx < N) {
        local_loss  += loss_per_token[idx];
        local_count += mask[idx];
        idx         += blockDim.x * gridDim.x;
    }

    sloss[tid]  = local_loss;
    scount[tid] = local_count;
    __syncthreads();

    // Tree reduction within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sloss[tid]  += sloss[tid + s];
            scount[tid] += scount[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out_loss,  sloss[0]);
        atomicAdd(out_count, scount[0]);
    }
}


/*
 * C++ binding: called from Python as grpo_ext.grpo_loss(...)
 *
 * Inputs:
 *   new_logp   (B, T) float32 — log probs from current policy
 *   old_logp   (B, T) float32 — log probs from rollout policy (frozen)
 *   advantages (B,)   float32 — group-relative normalized advantages
 *   mask       (B, T) float32 — 1.0 for real tokens, 0.0 for padding
 *   clip_low          float   — lower clip epsilon (DAPO default: 0.2)
 *   clip_high         float   — upper clip epsilon (DAPO Clip-Higher: 0.28)
 *
 * Returns:
 *   scalar float32 — mean token-level GRPO loss
 */
torch::Tensor grpo_loss(
    torch::Tensor new_logp,
    torch::Tensor old_logp,
    torch::Tensor advantages,
    torch::Tensor mask,
    float clip_low,
    float clip_high
) {
    TORCH_CHECK(new_logp.is_cuda(),  "new_logp must be on CUDA");
    TORCH_CHECK(old_logp.is_cuda(),  "old_logp must be on CUDA");
    TORCH_CHECK(advantages.is_cuda(),"advantages must be on CUDA");
    TORCH_CHECK(mask.is_cuda(),      "mask must be on CUDA");
    TORCH_CHECK(new_logp.dtype() == torch::kFloat32, "Expected float32");

    int B = new_logp.size(0);
    int T = new_logp.size(1);
    int N = B * T;

    auto loss_per_token = torch::empty({B, T}, new_logp.options());

    // Pass 1: fused per-token loss
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;

    grpo_dapo_loss_kernel<<<blocks, threads>>>(
        new_logp.data_ptr<float>(),
        old_logp.data_ptr<float>(),
        advantages.data_ptr<float>(),
        mask.data_ptr<float>(),
        loss_per_token.data_ptr<float>(),
        B, T, clip_low, clip_high
    );

    // Pass 2: masked mean via parallel reduction
    auto out_loss  = torch::zeros({1}, new_logp.options());
    auto out_count = torch::zeros({1}, new_logp.options());

    int red_blocks = min(blocks, 1024);
    size_t shared  = 2 * threads * sizeof(float);

    masked_sum_kernel<<<red_blocks, threads, shared>>>(
        loss_per_token.data_ptr<float>(),
        mask.data_ptr<float>(),
        out_loss.data_ptr<float>(),
        out_count.data_ptr<float>(),
        N
    );

    return (out_loss / (out_count + 1e-8f)).squeeze();
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("grpo_loss", &grpo_loss,
          "Fused GRPO + DAPO Clip-Higher token-level loss (CUDA)");
}
