"""
train_grpo.py - Multi-Turn Agentic RL via GRPO + DAPO fixes.

Replaces train_ppo.py. Based on:
  - Kevin (Cognition): GRPO with multi-turn ReAct, γ=0.4 discount, G=16 per task
  - Dr. Kernel: TRLOO unbiased advantage for multi-turn
  - DAPO (ByteDance): Token-level loss, Clip-Higher, no KL penalty

Key differences from PPO:
  - No value head (no critic) — saves ~16GB VRAM
  - No reference model / KL penalty — DAPO showed it's unnecessary
  - No GAE — advantages from group-relative reward normalization
  - Token-level loss — prevents vanishing gradients on long sequences
  - Asymmetric clipping (Clip-Higher) — prevents entropy collapse
"""

import os
import re
import random
from dataclasses import dataclass

import math
import time
import torch
from tqdm import tqdm
import torch.nn.functional as F
import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
_MP_SPAWN_CTX = multiprocessing.get_context("spawn")

# SGLang imports — optional, falls back to model.generate() if not installed
try:
    import sglang as sgl
    from sglang import RuntimeEndpoint
    SGLANG_AVAILABLE = True
except ImportError:
    SGLANG_AVAILABLE = False

from agent import _extract_cuda_code, _fix_cuda_api
from cuda_rag import CudaRAG
from llm_feedback import LLMFeedback, _format_llm_hint
from profiler import profile_kernel
from reward import calculate_reward, calculate_wrong_reward, calculate_opt_reward
from sandbox import evaluate
from sys_prompt import get_system_prompt

# Global RAG instance — indexes cuda_best_practices.md once at import time
_cuda_rag = CudaRAG()

# Global LLM feedback instance — initialized lazily when llm_feedback_model is set
_llm_feedback = None  # type: LLMFeedback | None


def _worker_run_eval(args):
    cand, prompt_text, timed, n_correctness = args
    if cand is None: return None
    return evaluate(cand, prompt_text, timed=timed, n_correctness=n_correctness)


def _extract_python_block(text: str) -> str:
    """Extract the first ```python ... ``` block from model output."""
    import re
    # Strip Qwen3 think blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    m = re.search(r'```python\s*(.*?)```', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: unclosed block (truncated by max_new_tokens)
    m = re.search(r'```python\s*(.*)', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GRPOConfig:
    model_id: str = "Qwen/Qwen3-14B"
    adapter_path: str = "checkpoints/kernelforge_redi"
    dataset_path: str = "data/rft_dataset.json"
    output_dir: str = "checkpoints/kernelforge_grpo"

    # Episode
    group_size: int = 16              # G trajectories per prompt

    # GRPO + DAPO hyperparameters
    grpo_epochs: int = 2              # gradient updates per batch
    batch_size: int = 4               # prompts per batch
    learning_rate: float = 5e-6       # higher LR for LoRA (fewer trainable params)
    warmup_pct: float = 0.1            # warmup as fraction of total steps (10%)
    cliprange_low: float = 0.2        # standard lower clip
    cliprange_high: float = 0.28      # DAPO Clip-Higher (asymmetric)
    max_grad_norm: float = 0.5        # standard GRPO clip (0.05 was too aggressive)

    # Multi-turn (Kevin's recipe): T refinement turns per trajectory
    # γ=0.4 discounts later turns so getting it right on turn 1 is worth more.
    # Kevin's ablation explicitly found 0.4 > 0.8 for CUDA kernel RL.
    num_turns: int = 4
    gamma: float = 0.4

    # Generation
    max_new_tokens: int = 6000        # total budget (thinking + code)
    think_budget: int = 2000          # phase-1 thinking cap; code gets the rest
    temperature: float = 0.9          # higher temp for diverse exploration (DAPO uses 1.0)
    mock_mode: bool = False

    # SGLang server-mode generation (faster than model.generate())
    # SGLang MUST be installed in a separate venv to avoid dependency conflicts.
    # Set SGLANG_PYTHON=/path/to/sglang_env/bin/python before running, or pass --sglang_python.
    use_sglang: bool = False
    sglang_python: str = ""      # path to SGLang venv python; falls back to $SGLANG_PYTHON
    sglang_port: int = 30000
    sglang_tp: int = 1           # tensor parallel degree (set to GPU count for multi-GPU)

    # Training
    num_train_epochs: int = 1
    save_steps: int = 50
    eval_steps: int = 50
    wandb_project: str = "kernelforge-rl"
    wandb_run_name: str = "grpo-qwen-14b"

    # Discrete milestone rewards (CUDA Agent style + graduated negatives)
    # -1 = no code/compile fail, -0.5 = wrong output, 1 = correct, 2 = beats eager, 3 = beats torch.compile
    reward_no_code: float = -1.0        # no ```python block found at all
    reward_compile_fail: float = -0.7   # code found but fails to compile
    reward_wrong_output: float = -0.5   # compiles but wrong output (stepping stone)

    # Entropy bonus — prevents entropy collapse (critical for coding tasks)
    # A small positive coefficient adds H(π) to the objective, keeping exploration alive.
    # Dr. Kernel and DAPO both recommend ~0.01-0.05 for code generation.
    entropy_coef: float = 0.02

    # Curriculum: weight sampling toward easier tasks (lower level_id) early in training
    # Set to True to use level_id field from dataset if available
    curriculum: bool = True

    # Dynamic Sampling: skip degenerate groups where all rewards are identical
    dynamic_sampling: bool = True
    max_resample_attempts: int = 3

    # SCoRe-style reward shaping: bonus for improvement, penalty for regression
    # Widens the GRPO advantage gap between "optimize successfully" vs "break working code"
    score_improvement_bonus: float = 0.5   # bonus when turn N improves over trajectory's best
    score_regression_penalty: float = 0.3  # extra penalty when correct→wrong regression

    # Dataset cap: limit to N prompts (0 = use all). Useful for time-boxed runs.
    max_prompts: int = 0

    # LLM feedback: path to GGUF model file for diagnosis/optimization hints
    # When set, replaces BM25 RAG with an LLM that diagnoses specific bugs
    # and identifies optimization bottlenecks. Runs on CPU (no GPU conflict).
    # Empty string = disabled (uses BM25 RAG as before).
    llm_feedback_model: str = ""


# ---------------------------------------------------------------------------
# Format helpers (must match agent.py)
# ---------------------------------------------------------------------------

PREFILL = ""

FORMAT_EXAMPLE = """\
Here is an example of the expected output format:

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = \"\"\"
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void add_kernel(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] + b[idx];
}

torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b) {
    auto out = torch::empty_like(a);
    int n = a.numel();
    add_kernel<<<(n+255)/256, 256>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}
\"\"\"

ext = load_inline(
    name="add_ext",
    cpp_sources="torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b);",
    cuda_sources=cuda_source,
    functions=["add_cuda"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, a, b):
        return ext.add_cuda(a, b)
```

Now write the complete model_new.py for the following operation. End your response with:
Reflection: <2-3 sentences covering: (1) your parallelization strategy and block/grid dimensions, (2) memory optimizations used (shared memory, vectorized loads, etc.), (3) what you would try next to improve performance>
"""

# ---------------------------------------------------------------------------
# Optimization Rules Catalog — universal CUDA techniques for turn 2+
# Each rule is a concrete before/after code pattern the model can apply.
# Different trajectories get assigned different rules for GRPO diversity.
# ---------------------------------------------------------------------------

OPTIMIZATION_RULES = [
    {
        "name": "Vectorized Memory Access (float4)",
        "bottleneck": "memory",
        "ops": ["elementwise", "reduction", "other"],  # NOT matmul/conv (complex indexing)
        "rule": (
            "Load/store 4 floats at once using float4 for 128-bit memory transactions (4x fewer transactions).\n"
            "Replace:\n"
            "  float val = input[idx];\n"
            "  output[idx] = f(val);\n"
            "With:\n"
            "  float4 in4 = reinterpret_cast<const float4*>(input)[idx];\n"
            "  float4 out4;\n"
            "  out4.x = f(in4.x); out4.y = f(in4.y); out4.z = f(in4.z); out4.w = f(in4.w);\n"
            "  reinterpret_cast<float4*>(output)[idx] = out4;\n"
            "Adjust grid size: blocks = (n/4 + threads - 1) / threads. Handle remainder elements separately if n % 4 != 0.\n"
            "IMPORTANT: only use float4 on the CONTIGUOUS dimension (last dim). The pointer must be 16-byte aligned.\n"
            "Check alignment: if (n % 4 != 0) fall back to scalar loads for remainder. "
            "Do NOT cast strided pointers (e.g. input + b*stride) to float4* — this causes misaligned address crashes."
        ),
    },
    {
        "name": "Grid-Stride Loop",
        "bottleneck": "both",
        "ops": ["elementwise", "reduction", "other"],  # universal for simple kernels
        "rule": (
            "Process multiple elements per thread with a grid-stride loop. Handles any size, improves occupancy.\n"
            "Replace:\n"
            "  int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
            "  if (idx < n) output[idx] = f(input[idx]);\n"
            "With:\n"
            "  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {\n"
            "      output[i] = f(input[i]);\n"
            "  }\n"
            "Launch fewer blocks (e.g. 128 blocks × 256 threads) and let each thread loop over its stride."
        ),
    },
    {
        "name": "Warp-Level Reduction",
        "bottleneck": "compute",
        "ops": ["reduction"],  # ONLY reductions — never elementwise/matmul
        "rule": (
            "For reductions (sum, max, min), use warp shuffle instead of shared memory — no sync needed.\n"
            "  float val = thread_value;\n"
            "  for (int offset = 16; offset > 0; offset >>= 1)\n"
            "      val += __shfl_down_sync(0xffffffff, val, offset);\n"
            "  // Now thread 0 of each warp has the warp's sum.\n"
            "For block-wide reduction: each warp reduces to lane 0, then lane 0s reduce via shared memory (only 32 elements)."
        ),
    },
    {
        "name": "Shared Memory Tiling",
        "bottleneck": "memory",
        "ops": ["matmul", "conv"],  # ONLY matmul/conv — never elementwise
        "rule": (
            "For operations that reuse data (matmul, conv, stencil), tile data into shared memory.\n"
            "IMPORTANT: __shared__ array sizes MUST be compile-time constants (not runtime variables).\n"
            "  #define TILE_SIZE 32\n"
            "  __shared__ float tile[TILE_SIZE][TILE_SIZE];\n"
            "  // Step 1: Load tile from global memory (coalesced access)\n"
            "  tile[ty][tx] = input[row * N + col];\n"
            "  __syncthreads();\n"
            "  // Step 2: Compute from shared memory (fast, no global memory latency)\n"
            "  for (int k = 0; k < TILE_SIZE; k++) sum += tile[ty][k] * ...;\n"
            "  __syncthreads();\n"
            "For dynamic sizes use: extern __shared__ float tile[]; and pass size as 3rd kernel launch arg.\n"
            "Typical tile sizes: 16x16 or 32x32. Shared memory is ~100x faster than global."
        ),
    },
    {
        "name": "Loop Unrolling",
        "bottleneck": "both",
        "ops": ["matmul", "conv", "reduction"],  # needs inner loops — NOT pure elementwise
        "rule": (
            "Unroll inner loops so the compiler can pipeline instructions and use registers efficiently.\n"
            "  #pragma unroll\n"
            "  for (int k = 0; k < TILE_K; k++) {\n"
            "      sum += a[k] * b[k];\n"
            "  }\n"
            "Or manually unroll by 4:\n"
            "  sum += a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3];\n"
            "Best for small, fixed-size inner loops. Combine with vectorized loads for maximum throughput."
        ),
    },
    {
        "name": "Kernel Fusion",
        "bottleneck": "memory",
        "ops": ["elementwise", "other"],  # fusing is for multi-op elementwise chains
        "rule": (
            "Combine multiple element-wise operations into a single kernel to eliminate intermediate memory traffic.\n"
            "Instead of separate kernels:\n"
            "  kernel1: temp[i] = relu(x[i])\n"
            "  kernel2: out[i] = temp[i] + bias[i]\n"
            "Fuse into one kernel:\n"
            "  out[i] = fmaxf(0.0f, x[i]) + bias[i];\n"
            "This halves memory traffic (no temp array) and eliminates kernel launch overhead."
        ),
    },
    {
        "name": "Read-Only Cache (__ldg)",
        "bottleneck": "memory",
        "ops": ["elementwise", "reduction", "matmul", "conv", "other"],  # universal
        "rule": (
            "For read-only input data, use __ldg() to use the texture/L2 cache path.\n"
            "Replace:\n"
            "  float val = input[idx];\n"
            "With:\n"
            "  float val = __ldg(&input[idx]);\n"
            "This hints the hardware to cache the read in L2 without polluting L1. "
            "Especially useful for scattered or read-many access patterns."
        ),
    },
    {
        "name": "Occupancy Tuning",
        "bottleneck": "both",
        "ops": ["elementwise", "reduction", "matmul", "conv", "other"],  # universal
        "rule": (
            "Increase occupancy (active warps per SM) to better hide memory latency.\n"
            "- Try 128 threads/block instead of 256 — more blocks fit per SM.\n"
            "- Reduce local variables to lower register pressure.\n"
            "- Reduce shared memory usage so more blocks can coexist.\n"
            "- Use __launch_bounds__(maxThreads, minBlocks) to guide the compiler:\n"
            "  __global__ void __launch_bounds__(128, 8) my_kernel(...) { ... }\n"
            "Check: blocks_per_SM = shared_mem_per_SM / shared_mem_per_block. Aim for >= 4."
        ),
    },
    {
        "name": "Coalesced Memory Access (Loop Reordering)",
        "bottleneck": "memory",
        "ops": ["matmul", "conv", "reduction", "other"],  # NOT pure elementwise (already coalesced)
        "rule": (
            "Ensure consecutive threads access consecutive memory addresses (coalesced access).\n"
            "Bad (strided access — threads jump across memory):\n"
            "  for (int c = 0; c < C; c++)\n"
            "    for (int h = 0; h < H; h++)\n"
            "      output[c * H + h] = ...;  // threads access c*H apart\n"
            "Good (consecutive access — threads access neighbors):\n"
            "  for (int h = 0; h < H; h++)\n"
            "    for (int c = 0; c < C; c++)\n"
            "      output[c * H + h] = ...;  // threads access 1 apart\n"
            "Rule: the innermost loop index should match the fastest-varying memory dimension. "
            "For row-major tensors, the last dimension is fastest."
        ),
    },
]


def _get_bottleneck_type(profiler_feedback: str) -> str:
    """Extract bottleneck type from profiler feedback text.
    Returns 'memory', 'compute', or 'unknown'."""
    if not profiler_feedback:
        return "unknown"
    text = profiler_feedback.upper()
    if "MEMORY-BOUND" in text or "MEMORY BOUND" in text:
        return "memory"
    if "COMPUTE-BOUND" in text or "COMPUTE BOUND" in text:
        return "compute"
    return "unknown"


def _classify_code_structure(cuda_code: str) -> dict:
    """Analyze generated CUDA code to detect what structures are present.
    Returns a dict of boolean flags for code features.

    This is the ground truth — looks at the ACTUAL code the model wrote,
    not keywords in the prompt. Works for any operation type.
    """
    if not cuda_code:
        return {}
    features = {}

    # Inner for-loops (candidates for unrolling / tiling)
    # Look for `for (` patterns inside __global__ functions
    for_loops = re.findall(r'for\s*\(', cuda_code)
    features["has_inner_loops"] = len(for_loops) >= 2  # at least 2 loops = nested

    # Shared memory usage
    features["has_shared_memory"] = "__shared__" in cuda_code

    # Reduction patterns (atomicAdd, __shfl, warp-level ops)
    features["has_reduction"] = any(p in cuda_code for p in [
        "atomicAdd", "atomicMax", "atomicMin",
        "__shfl_down_sync", "__shfl_xor_sync", "__shfl_sync",
        "__syncthreads",  # often signals data sharing / reduction
    ])

    # Data reuse patterns (same array indexed with different offsets in a loop)
    # Heuristic: multiple accesses to same array in a for-loop body
    features["has_data_reuse"] = bool(re.search(
        r'for\s*\(.*?\).*?\{[^}]*\w+\[[^]]+\].*?\w+\[[^]]+\].*?\}',
        cuda_code, re.DOTALL
    ))

    # Simple elementwise pattern: one idx, one read, one write, no loops
    features["is_simple_elementwise"] = (
        not features["has_inner_loops"]
        and not features["has_shared_memory"]
        and not features["has_reduction"]
        and bool(re.search(r'blockIdx\.x\s*\*\s*blockDim\.x\s*\+\s*threadIdx\.x', cuda_code))
    )

    # Multiple nested loops with accumulation (matmul-like)
    features["has_accumulation_loop"] = bool(re.search(
        r'for\s*\(.*?\).*?\{[^}]*\+=', cuda_code, re.DOTALL
    ))

    # Sliding window / stencil pattern (offsets like [i+1], [i-1], [row+k])
    features["has_stencil"] = bool(re.search(
        r'\w+\[\s*\w+\s*[+-]\s*\d+\s*\]', cuda_code
    ))

    return features


def _get_operation_type(prompt_text: str, cuda_code: str | None = None) -> str:
    """Classify the kernel operation type.
    Returns 'elementwise', 'reduction', 'matmul', 'conv', or 'other'.

    Primary signal: structural analysis of the generated CUDA code (if available).
    Fallback: keyword matching on the prompt text.

    Code-based classification generalizes to ANY operation because it looks at
    what the model actually wrote, not what the prompt asked for.
    """
    # ── Primary: analyze the actual generated code ──
    if cuda_code:
        feat = _classify_code_structure(cuda_code)

        # Simple elementwise: no loops, no shared mem, no reduction, just idx→read→write
        if feat.get("is_simple_elementwise"):
            return "elementwise"

        # Reduction: has warp shuffles, atomics, or syncthreads without nested loops
        if feat.get("has_reduction") and not feat.get("has_inner_loops"):
            return "reduction"

        # Matmul-like: nested loops with accumulation, possible data reuse
        if feat.get("has_inner_loops") and feat.get("has_accumulation_loop"):
            # Distinguish matmul from conv by checking for stencil patterns
            if feat.get("has_stencil"):
                return "conv"
            return "matmul"

        # Stencil/conv: sliding window with offsets
        if feat.get("has_stencil") and feat.get("has_inner_loops"):
            return "conv"

        # Reduction with loops (block-wide reduction, layernorm-style)
        if feat.get("has_reduction") and feat.get("has_inner_loops"):
            return "reduction"

        # Has loops but no accumulation or reduction — could be complex elementwise
        if feat.get("has_inner_loops") and not feat.get("has_accumulation_loop"):
            return "other"

        # Simple kernel with no interesting structure
        if not feat.get("has_inner_loops") and not feat.get("has_reduction"):
            return "elementwise"

    # ── Fallback: keyword matching on prompt text ──
    text = prompt_text.lower()

    matmul_kw = ["matmul", "matrix_multiply", "matrix multiply", "gemm",
                 "matrix multiplication", "torch.mm", "torch.bmm",
                 "torch.matmul", "einsum"]
    if any(kw in text for kw in matmul_kw):
        if "einsum" in text:
            if any(p in text for p in ["ij,jk", "bij,bjk", "bhij,bhjk", "ik,kj"]):
                return "matmul"
            return "other"
        return "matmul"

    conv_kw = ["conv1d", "conv2d", "conv3d", "convolution", "conv_transpose",
               "depthwise_conv", "deconv", "sliding window", "stencil",
               "torch.nn.conv", "f.conv"]
    if any(kw in text for kw in conv_kw):
        return "conv"

    reduce_kw = ["reduce", "sum(", ".sum(", "mean(", ".mean(", "norm(",
                 ".norm(", "softmax", "layernorm", "layer_norm",
                 "batchnorm", "batch_norm", "groupnorm", "group_norm",
                 "argmax", "argmin", ".max(", ".min(", "logsumexp",
                 "cross_entropy", "nll_loss"]
    if any(kw in text for kw in reduce_kw):
        return "reduction"

    elem_kw = ["relu", "gelu", "silu", "sigmoid", "tanh", "leaky_relu",
               "leakyrelu", "elu(", "selu", "mish", "swish", "hardswish",
               "hardsigmoid", "softplus", "clamp", "elementwise",
               "pointwise", "element-wise", "point-wise",
               "activation", "+ bias", "add(", "mul(", "dropout"]
    if any(kw in text for kw in elem_kw):
        return "elementwise"

    return "other"


def _filter_rules_by_bottleneck(bottleneck: str, op_type: str = "other") -> list:
    """Return rules matching the profiler bottleneck AND operation type.
    'both' rules always included for bottleneck. Falls back if too few match."""
    # First filter by bottleneck
    if bottleneck == "unknown":
        by_bottleneck = OPTIMIZATION_RULES
    else:
        by_bottleneck = [r for r in OPTIMIZATION_RULES
                         if r["bottleneck"] == bottleneck or r["bottleneck"] == "both"]
        if len(by_bottleneck) < 2:
            by_bottleneck = OPTIMIZATION_RULES

    # Then filter by operation type
    matching = [r for r in by_bottleneck if op_type in r.get("ops", ["other"])]
    if len(matching) < 2:
        # Fallback: at least return safe universal rules
        safe = [r for r in by_bottleneck
                if r["name"] in ("Occupancy Tuning", "Read-Only Cache (__ldg)", "Grid-Stride Loop")]
        return safe if len(safe) >= 2 else by_bottleneck
    return matching




# ---------------------------------------------------------------------------
# Multi-turn helpers (Kevin's recipe)
# ---------------------------------------------------------------------------

def _strip_thinking(text: str, truncate_code: bool = False) -> str:
    """
    Strip internal <think> blocks from a prior-turn response, keeping all
    code and the Reflection line intact.

    If truncate_code=True, also replace the full code block with a short
    placeholder to save context tokens for older turns (only the most recent
    turn keeps full code visible).
    """
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL)
    cleaned = cleaned.strip()
    if truncate_code:
        # Replace ```python ... ``` blocks with a short placeholder
        cleaned = re.sub(
            r'```python\s*.*?```',
            '```python\n# [Previous code omitted — see feedback below for what to fix]\n```',
            cleaned,
            flags=re.DOTALL,
        )
    return cleaned


def _classify_error(eval_res: dict | None) -> str:
    """Classify an eval result into a coarse error bucket for stuck-detection."""
    if eval_res is None:
        return "no_code"
    if not eval_res.get("compiles", False):
        return "compile"
    if not eval_res.get("correct", False):
        if eval_res.get("shape_ok") is False:
            return "shape"
        wf = eval_res.get("wrong_frac") or 1.0
        me = eval_res.get("max_abs_error") or 0.0
        if wf > 0.3 and me > 5.0:
            return "algorithm"
        if abs(eval_res.get("systematic_bias") or 0.0) > 0.1:
            return "systematic"
        return "boundary"
    return "correct"


def _build_turn_feedback(eval_res: dict | None, prev_eval: dict | None = None,
                         best_speedup: float | None = None,
                         best_turn: int | None = None,
                         profiler_feedback: str | None = None,
                         group_error_summary: str | None = None,
                         rag_hint: str | None = None,
                         opt_hint: str | None = None) -> str:
    """
    Feedback: raw execution results + high water mark for optimization context.

    - Raw error data (compiler errors, wrong values with numbers) is the primary signal.
    - When the same error class repeats, prepends a "try again" signal.
    - For correct kernels: includes speedup number so model can iterate on speed.
    - When correctness breaks after a previous correct turn: references the high water
      mark (best speedup and which turn) so the model knows what it had.
    - On final turn: optionally includes profiler hardware metrics.
    - When ALL trajectories failed: includes group error summary so model avoids dead ends.
    """
    # Group error suffix: when ALL trajectories failed, show what everyone tried
    group_suffix = ""
    if group_error_summary and (eval_res is None or not eval_res.get("correct", False)):
        group_suffix = f"\n\n{group_error_summary}"

    stuck = (
        prev_eval is not None
        and _classify_error(eval_res) == _classify_error(prev_eval)
        and _classify_error(eval_res) != "correct"
    )
    stuck_prefix = "Your previous attempt made the same type of error. Try a different approach.\n\n" if stuck else ""

    # High water mark suffix: when correctness breaks, tell the model what it had
    hwm_suffix = ""
    if best_speedup is not None and best_turn is not None:
        if eval_res is None or not eval_res.get("correct", False):
            hwm_suffix = (
                f"\n\nNote: your turn {best_turn} solution was correct"
                f" at {best_speedup:.2f}x speedup over PyTorch."
            )

    # RAG hint: inject relevant CUDA patterns from best practices doc
    rag_suffix = f"\n\n{rag_hint}" if rag_hint else ""

    if eval_res is None:
        return (
            stuck_prefix +
            "Your previous answer failed to be parsed due to not adhering to the desired formatting."
            + hwm_suffix + group_suffix + rag_suffix + "\n\n"
            "Try a completely different approach and generate new, complete code."
        )
    if not eval_res.get("compiles", False):
        err = (eval_res.get("compiler_error") or "Unknown compile error")
        return (
            stuck_prefix +
            f"Your previous answer failed to compile. Here is the error message:\n{err}"
            + hwm_suffix + group_suffix + rag_suffix + "\n\n"
            "Fix the compilation error and generate new, complete code."
        )
    if not eval_res.get("correct", False):
        err = (eval_res.get("compiler_error") or "Outputs do not match reference")
        return (
            stuck_prefix +
            f"Your previous answer was incorrect. Here is the error message:\n{err}"
            + hwm_suffix + group_suffix + rag_suffix + "\n\n"
            "Fix the correctness issue and generate new, complete code."
        )
    # Correct kernel — include timing milestones and optional profiler data
    rt = eval_res.get("runtime_ms")
    bt = eval_res.get("baseline_runtime_ms")
    ct = eval_res.get("compile_runtime_ms")
    timing_lines = []
    if rt and bt:
        timing_lines.append(f"Your kernel: {rt:.3f}ms")
        timing_lines.append(f"PyTorch eager: {bt:.3f}ms ({bt/rt:.2f}x)")
    if ct:
        timing_lines.append(f"torch.compile: {ct:.3f}ms" + (f" ({ct/rt:.2f}x)" if rt else ""))
    timing_str = "\n".join(timing_lines)
    profiler_str = f"\n\n{profiler_feedback}" if profiler_feedback else ""

    speedup_val = (bt / rt) if (rt and bt and rt > 0) else 0.0

    # LLM optimization hint (if available) replaces generic advice
    opt_suffix = f"\n\n{opt_hint}" if opt_hint else ""

    if speedup_val >= 1.0:
        # OPT TURN: already faster than PyTorch — safe micro-optimizations only
        if opt_hint:
            advice = (
                "IMPORTANT: Your kernel is correct AND already faster than PyTorch.\n"
                "Do NOT rewrite it from scratch. Make small, incremental changes only."
                + opt_suffix
            )
        else:
            advice = (
                "IMPORTANT: Your kernel is correct AND already faster than PyTorch.\n"
                "Do NOT rewrite it from scratch. Make small, incremental changes only.\n"
                "Safe optimizations: adjust block/grid sizes, add #pragma unroll, "
                "use float4/int4 vectorized loads, reduce redundant computation.\n"
                "Do NOT attempt shared memory tiling or major algorithmic rewrites — "
                "those often break correctness."
            )
        return (
            f"Your previous answer was correct.\n{timing_str}{profiler_str}\n\n"
            + advice + "\nGenerate the complete improved code."
        )
    else:
        # REWRITE TURN: slower than PyTorch — allow algorithmic restructuring
        if opt_hint:
            advice = (
                "Your kernel is correct but SLOWER than PyTorch. "
                "You may restructure the algorithm to improve performance."
                + opt_suffix
            )
        else:
            advice = (
                "Your kernel is correct but SLOWER than PyTorch. "
                "You may restructure the algorithm to improve performance.\n"
                "Focus on: reducing total work, improving memory access patterns, "
                "increasing parallelism, and ensuring coalesced memory access."
            )
        return (
            f"Your previous answer was correct.\n{timing_str}{profiler_str}\n\n"
            + advice + "\nKeep the same function signature and output shape. "
            "Generate the complete improved code."
        )


# ---------------------------------------------------------------------------
# SGLang server helpers
# ---------------------------------------------------------------------------

_sglang_server = None  # global handle so we can shut it down at exit
_lora_temp_dir = None  # temp dir for LoRA sync between trainer and SGLang
_LORA_NAME = "kf_grpo_adapter"  # name SGLang uses for the live LoRA adapter


def launch_sglang_server(model_path: str, adapter_path: str, port: int, tp: int,
                         sglang_python: str = None):
    """
    Launch SGLang as an inference server in a subprocess using the base model
    + LoRA loaded dynamically via /load_lora_adapter (no merge needed).

    After each training step: save LoRA → /release_memory_occupation →
    backward pass → /resume_memory_occupation → /load_lora_adapter (hot-reload).
    No NCCL required.

    Returns the server process handle.
    """
    import subprocess, sys, time, requests, os

    python_bin = (
        sglang_python
        or os.environ.get("SGLANG_PYTHON")
        or sys.executable
    )
    if python_bin == sys.executable:
        print("[SGLang] WARNING: using training venv Python for SGLang server. "
              "Set SGLANG_PYTHON=/path/to/sglang_env/bin/python to avoid "
              "dependency conflicts.")

    cmd = [
        python_bin, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--port", str(port),
        "--tp", str(tp),
        "--dtype", "bfloat16",
        "--trust-remote-code",
        "--enable-lora",
        "--lora-paths", f"{_LORA_NAME}={adapter_path}",
        "--max-loras-per-batch", "1",
        "--mem-fraction-static", "0.60",  # 0.45→0.60: doubles KV cache 16GB→30GB, fixes turn 3+ eviction
        "--context-length", "32768",
        "--log-level", "error",
        "--max-running-requests", "16",   # G=8 × 2 phases = 16 max concurrent requests
        "--schedule-policy", "lpm",       # longest-prefix-match: maximises KV reuse across turns
        "--enable-mixed-chunk",           # mix prefill+decode in same batch, reduces stalls
    ]

    import glob as _glob
    cuda_homes = sorted(_glob.glob("/usr/local/cuda-*"), reverse=True) + ["/usr/local/cuda"]
    cuda_home = next((p for p in cuda_homes if os.path.isfile(f"{p}/bin/nvcc")), None)
    base_env = dict(os.environ)
    if cuda_home:
        cuda_bin = f"{cuda_home}/bin"
        cuda_inc = f"{cuda_home}/include"
        path = base_env.get("PATH", "")
        if cuda_bin not in path:
            base_env["PATH"] = f"{cuda_bin}:{path}"
        cpath = base_env.get("CPATH", "")
        if cuda_inc not in cpath:
            base_env["CPATH"] = f"{cuda_inc}:{cpath}"
    env = {**base_env, "SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK": "1"}
    proc = subprocess.Popen(cmd, env=env)

    # Wait for server to be ready
    url = f"http://localhost:{port}/health"
    for _ in range(120):
        try:
            if requests.get(url, timeout=2).status_code == 200:
                print(f"[SGLang] Server ready on port {port}")
                break
        except Exception:
            pass
        time.sleep(2)
    else:
        proc.terminate()
        raise RuntimeError(f"SGLang server failed to start on port {port}")

    # Initial LoRA is loaded at server startup via --lora-paths
    print(f"[SGLang] Server ready with LoRA '{_LORA_NAME}' loaded from {adapter_path}")
    return proc


def release_sglang_memory(port: int):
    """Release SGLang's GPU allocation so the trainer can use full VRAM for backward pass."""
    import requests
    try:
        r = requests.post(f"http://localhost:{port}/release_memory_occupation",
                          json={}, timeout=60)
        print(f"[SGLang] Memory released (status={r.status_code})")
    except Exception as e:
        print(f"[SGLang] release_memory_occupation failed: {e}")


def resume_sglang_memory(port: int):
    """Restore SGLang's GPU allocation after the training step is done."""
    import requests
    try:
        r = requests.post(f"http://localhost:{port}/resume_memory_occupation",
                          json={}, timeout=120)
        print(f"[SGLang] Memory resumed (status={r.status_code})")
    except Exception as e:
        print(f"[SGLang] resume_memory_occupation failed: {e}")


def sync_lora_to_sglang(model, port: int):
    """
    Save the updated LoRA adapter to a temp dir and hot-reload it in SGLang.
    Called after each optimizer step so SGLang always generates with the latest policy.
    """
    import requests, tempfile, os

    global _lora_temp_dir
    if _lora_temp_dir is None:
        import tempfile as _tmp
        _lora_temp_dir = _tmp.mkdtemp(prefix="kf_lora_sync_")

    lora_path = os.path.join(_lora_temp_dir, "adapter")
    os.makedirs(lora_path, exist_ok=True)

    # Save only the LoRA adapter weights (small — ~100MB)
    model.save_pretrained(lora_path)

    # Unload the old adapter, then load the updated one
    try:
        requests.post(f"http://localhost:{port}/unload_lora_adapter",
                      json={"lora_name": _LORA_NAME}, timeout=30)
    except Exception:
        pass  # best-effort; load will fail if still loaded

    try:
        r = requests.post(
            f"http://localhost:{port}/load_lora_adapter",
            json={"lora_name": _LORA_NAME, "lora_path": lora_path},
            timeout=60,
        )
        if r.status_code == 200:
            print(f"[SGLang] LoRA hot-reloaded (step saved → {lora_path})")
        else:
            print(f"[SGLang] load_lora_adapter: status={r.status_code} {r.text[:200]}")
    except Exception as e:
        print(f"[SGLang] load_lora_adapter failed: {e}")

    # Flush stale KV cache (entries computed with old weights are invalid)
    try:
        requests.post(f"http://localhost:{port}/flush_cache", timeout=30)
    except Exception:
        pass




def _sglang_post(port: int, contexts: list[str], max_new_tokens: int,
                 temperature: float, stop: list[str]) -> list[str]:
    """
    Raw SGLang /generate call. Returns full text (context + completion).
    If a batch request fails (e.g. context too long), falls back to
    per-request calls so one long context doesn't crash the whole batch.
    """
    import requests

    def _single(ctx):
        payload = {
            "text": ctx,
            "lora_name": _LORA_NAME,
            "sampling_params": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": 1.0 if temperature == 0.0 else 0.95,
                "stop": stop,
            },
        }
        try:
            r = requests.post(f"http://localhost:{port}/generate", json=payload, timeout=600)
            r.raise_for_status()
            res = r.json()
            return res["text"] if isinstance(res, dict) else res[0]["text"]
        except Exception as e:
            print(f"  [SGLang] single request failed (context too long?): {e}")
            return ctx  # return context unchanged → empty completion

    # Try batch first (fast path)
    payload = {
        "text": contexts,
        "lora_name": _LORA_NAME,
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": 1.0 if temperature == 0.0 else 0.95,
            "stop": stop,
        },
    }
    try:
        resp = requests.post(f"http://localhost:{port}/generate", json=payload, timeout=600)
        resp.raise_for_status()
        result = resp.json()
        if isinstance(result, list):
            return [r["text"] for r in result]
        return [result["text"]]
    except Exception:
        # Batch failed — fall back to per-request concurrently so one bad context doesn't kill all 8
        print("  [SGLang] Batch request failed, falling back to concurrent per-request mode...")
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=len(contexts)) as ex:
            return list(ex.map(_single, contexts))


def _generate_with_sglang(context_texts: list[str], config: "GRPOConfig") -> list[str]:
    """
    Budget-forcing generation:
      Phase 1 — let the model think for up to think_budget tokens,
                 stop at </think> or ```python (whichever comes first).
      Phase 1b — if thinking was too short (< MIN_THINK_WORDS), retry once
                 with a nudge to encourage deeper reasoning.
      Phase 2 — inject </think>\n```python\n and generate the code.

    This prevents the model from spending all 6000 tokens in <think>
    and never writing any code, AND prevents it from skipping thinking entirely.
    """
    think_budget = config.think_budget
    code_budget   = config.max_new_tokens - think_budget
    MIN_THINK_WORDS = 30  # minimum words of reasoning before we accept

    # ── Phase 1: thinking ───────────────────────────────────────────────────
    # Stop at </think> so model can't skip thinking and jump straight to code.
    # SGLang strips stop tokens from output, so comp won't contain </think>.
    phase1_raw = _sglang_post(
        config.sglang_port, context_texts, think_budget, config.temperature,
        stop=["<|im_end|>", "```python", "</think>"],
    )

    # ── Phase 1b: retry if thinking too short ────────────────────────────────
    for i, (ctx, full) in enumerate(zip(context_texts, phase1_raw)):
        comp = full[len(ctx):] if full.startswith(ctx) else full
        think_words = len(comp.strip().split())
        if think_words < MIN_THINK_WORDS:
            # Model skipped thinking — retry with accumulated context + nudge
            retry_ctx = ctx + comp + "\nLet me think step by step about the CUDA kernel:\n1. "
            retry_raw = _sglang_post(
                config.sglang_port, [retry_ctx], think_budget, config.temperature,
                stop=["<|im_end|>", "```python", "</think>"],
            )
            phase1_raw[i] = retry_raw[0]
            context_texts[i] = retry_ctx  # update for phase 2
            print(f"  [THINK] traj={i}: thinking too short ({think_words} words), retried", flush=True)

    # ── Build phase-2 contexts ───────────────────────────────────────────────
    phase2_contexts = []
    phase1_completions = []

    for ctx, full in zip(context_texts, phase1_raw):
        comp = full[len(ctx):] if full.startswith(ctx) else full
        phase1_completions.append(comp)

        if "```python" in comp:
            # Model started writing code inside thinking — split at code boundary
            idx = comp.index("```python")
            phase2_contexts.append(ctx + comp[:idx] + "```python\n")
        else:
            # Normal case: thinking stopped at </think> or budget.
            # Inject transition to code.
            phase2_contexts.append(ctx + comp + "\n</think>\n```python\n")

    # ── Phase 2: code ────────────────────────────────────────────────────────
    phase2_raw = _sglang_post(
        config.sglang_port, phase2_contexts, code_budget, config.temperature,
        stop=["<|im_end|>"],
    )

    # ── Combine: return context + phase1 + phase2 ────────────────────────────
    results = []
    for ctx, p2ctx, p2full in zip(context_texts, phase2_contexts, phase2_raw):
        full_with_ctx = p2full if p2full.startswith(ctx) else (p2ctx + p2full)
        results.append(full_with_ctx)
    return results


# ---------------------------------------------------------------------------
# Single-turn episode (standard GRPO — no ReAct loop)
# ---------------------------------------------------------------------------

# Each turn: (context_ids, response_ids)
TurnData = tuple[torch.Tensor, torch.Tensor]


def _run_group_episodes(
    prompt_text: str,
    model,
    tokenizer,
    config: GRPOConfig,
    difficulty: int = 1,
    rag_prob: float = 1.0,
) -> tuple[list[list[TurnData]], list[list[float]]]:
    """
    Generate G trajectories × T turns (Kevin's multi-turn recipe).

    Each turn: generate → evaluate → build feedback → build next-turn context.
    Thinking is stripped from inter-turn context so the model only sees its code + feedback.

    Args:
        difficulty: 1/2/3 from level_id. Scales positive rewards so harder
                    problems are worth more (1.0x / 1.25x / 1.5x).

    Returns:
      group_turns:   list[G] of list[T] of (ctx_ids, resp_ids)
      group_rewards: list[G] of list[T] of scalar rewards
    """
    # Difficulty multiplier: harder problems are worth more
    diff_scale = {1: 1.0, 2: 1.25, 3: 1.5}.get(difficulty, 1.0)
    G = config.group_size
    T = config.num_turns

    sys_content = get_system_prompt().strip()
    user_msg = FORMAT_EXAMPLE + f"Reference Program:\n```python\n{prompt_text}\n```"
    base_messages = [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": user_msg},
    ]
    print(f"  [DEBUG] System prompt: {len(sys_content)} chars | User msg: {len(user_msg)} chars")

    group_turns:   list[list[TurnData]]   = [[] for _ in range(G)]
    group_rewards: list[list[float]]      = [[] for _ in range(G)]

    # Per-trajectory conversation state across turns
    traj_responses: list[list[str]]  = [[] for _ in range(G)]  # raw completions (keep <think>)
    traj_evals:     list[list[dict]] = [[] for _ in range(G)]  # eval results per turn

    # High water mark: best correct speedup per trajectory (for optimization feedback)
    traj_best_speedup: list[float | None] = [None] * G  # best speedup achieved so far
    traj_best_turn:    list[int | None]   = [None] * G  # which turn achieved it

    # SCoRe tracking: per-trajectory correctness history for regression/improvement detection
    traj_was_correct:  list[bool]  = [False] * G  # was this trajectory ever correct?
    traj_best_reward:  list[float] = [float('-inf')] * G  # best reward achieved so far

    # Track what feedback each trajectory received on turn 2+ (rule, profiler, RAG)
    # so we can correlate feedback → outcome
    traj_turn_feedback_source: list[dict] = [{} for _ in range(G)]  # reset each turn

    # Warm start: overall best correct kernel across ALL trajectories and ALL turns
    # Failed trajectories in turn 2+ get this kernel injected so they focus on optimization
    # Updated each turn — always tracks the fastest correct kernel seen so far
    ws_best_code: str | None = None
    ws_best_eval: dict | None = None
    ws_best_completion: str | None = None
    ws_best_speedup: float = 0.0
    ws_best_source: str = ""  # "turn X traj Y" for debug logging
    # Bug 1 fix: freeze the speedup when optimisation phase began so delta
    # comparisons use a stable baseline instead of the drifting ws_best_speedup.
    ws_speedup_at_opt_start: float | None = None

    # ── DEBUG: what operation is this prompt? ───────────────────────────────
    # Extract first torch call from prompt to label the operation
    op_match = re.search(r'torch\.\w+|nn\.\w+', prompt_text)
    op_label = op_match.group(0) if op_match else "unknown_op"
    prompt_lines = prompt_text.strip().count('\n') + 1
    print(f"  [DEBUG] Prompt: {op_label}, {prompt_lines} lines, {len(prompt_text)} chars")
    # ── END DEBUG ────────────────────────────────────────────────────────────

    group_error_summary = None  # set after each turn if all failed

    for turn_idx in range(T):
        turn_label = f"Turn {turn_idx+1}/{T}"

        # Build context texts: base prompt + all prior turns [code + feedback].
        # Kevin, Dr. Kernel, and MURPHY all keep full history — the model needs to
        # see all prior attempts to identify which approach was best and avoid repeating mistakes.
        context_texts = []

        # ── Pre-compute deduplicated LLM diagnoses for this turn ──────────
        # When multiple trajectories have the same error class on the
        # previous turn, diagnose once and reuse — avoids redundant
        # sequential LLM calls (saves ~35s per duplicate).
        _diag_cache: dict[str, str] = {}  # error_class → formatted diagnosis
        if (turn_idx > 0
                and _llm_feedback is not None and _llm_feedback.available):
            _prev_turn = turn_idx - 1
            # Group trajectories by error class
            _error_classes: dict[str, int] = {}  # class → first traj index
            for _gi in range(G):
                _prev_eval = traj_evals[_gi][_prev_turn] if _prev_turn < len(traj_evals[_gi]) else None
                _ec = _classify_error(_prev_eval)
                if _ec != "correct" and _ec not in _error_classes:
                    _error_classes[_ec] = _gi
            # Diagnose one representative per error class
            for _ec, _rep_i in _error_classes.items():
                _rep_eval = traj_evals[_rep_i][_prev_turn]
                _rep_code = ""
                if _prev_turn < len(traj_responses[_rep_i]) and traj_responses[_rep_i][_prev_turn]:
                    _rep_code = _extract_python_block(traj_responses[_rep_i][_prev_turn])
                _rep_error = ""
                if _rep_eval is None:
                    _rep_error = "No code was extracted from the response"
                elif not _rep_eval.get("compiles", False):
                    _rep_error = _rep_eval.get("compiler_error") or "Unknown compile error"
                else:
                    _rep_error = _rep_eval.get("compiler_error") or "Outputs do not match reference"
                    _wf = _rep_eval.get("wrong_frac")
                    _mae = _rep_eval.get("max_abs_error")
                    if _wf is not None:
                        _rep_error += f" | {_wf*100:.0f}% of elements wrong"
                    if _mae is not None:
                        _rep_error += f" | max_abs_error={_mae:.5f}"
                _hint = _llm_feedback.diagnose_error(
                    task=prompt_text,
                    code=_rep_code or "(no code extracted)",
                    error=_rep_error,
                )
                if _hint:
                    _diag_cache[_ec] = _format_llm_hint(_hint, "diagnosis")
            _n_unique = len(_error_classes)
            _n_failed = sum(1 for _gi in range(G)
                            if _classify_error(
                                traj_evals[_gi][_prev_turn] if _prev_turn < len(traj_evals[_gi]) else None
                            ) != "correct")
            if _n_unique > 0:
                print(f"  [LLM-DEDUP] {_n_failed} failed trajs → {_n_unique} unique error class(es), "
                      f"{_n_unique} diagnosis call(s) (saved {max(0, _n_failed - _n_unique)})",
                      flush=True)

        for i in range(G):
            msgs = list(base_messages)

            # ── Optimization-first turn strategy ─────────────────────────────
            # Turn 2+: if a correct kernel exists at reasonable speed, ALL
            # trajectories get it with specific optimization rules + profiler.
            # >= 0.5x: algorithm is sound, needs micro-optimization (float4, etc.)
            # < 0.5x:  algorithm is fundamentally wrong, needs rewrite not tuning
            #
            # Skip opt turn when profiler says "Already Faster" with low
            # utilization — the kernel is already near-optimal for a small
            # workload. Forcing optimization risks breaking correctness
            # for marginal gain.
            _ws_prof = ws_best_eval.get("profiler_feedback", "") if ws_best_eval else ""
            _already_fast_low_util = (
                "Already Faster Than PyTorch" in _ws_prof
                and ws_best_speedup >= 1.0
            )
            use_optimization_turn = (
                turn_idx > 0
                and ws_best_code is not None
                and ws_best_speedup >= 0.5
                and not _already_fast_low_util
            )
            if _already_fast_low_util and turn_idx > 0 and ws_best_code is not None:
                if i == 0:
                    print(f"  [OPT SKIP] Profiler says 'Already Faster' with low util "
                          f"({ws_best_speedup:.2f}x) — skipping optimization turn",
                          flush=True)
            use_opt_rules = use_optimization_turn  # always True when opt turn is active

            if use_optimization_turn:
                # Freeze baseline on first opt turn (Bug 1)
                if ws_speedup_at_opt_start is None:
                    ws_speedup_at_opt_start = ws_best_speedup

                # ALL trajectories get the best correct kernel
                stripped = _strip_thinking(ws_best_completion, truncate_code=False)
                msgs.append({"role": "assistant", "content": stripped})

                # Build timing string from the best kernel's eval
                rt = ws_best_eval.get("runtime_ms") if ws_best_eval else None
                bt = ws_best_eval.get("baseline_runtime_ms") if ws_best_eval else None
                ct = ws_best_eval.get("compile_runtime_ms") if ws_best_eval else None
                timing_lines = []
                if rt and bt:
                    timing_lines.append(f"Your kernel: {rt:.3f}ms")
                    timing_lines.append(f"PyTorch eager: {bt:.3f}ms ({bt/rt:.2f}x)")
                if ct:
                    timing_lines.append(f"torch.compile: {ct:.3f}ms" + (f" ({ct/rt:.2f}x)" if rt else ""))
                timing_str = "\n".join(timing_lines)

                # Bug 2 fix: use THIS trajectory's profiler data if available,
                # falling back to the base kernel's profile only if needed.
                traj_prof = None
                if turn_idx > 1 and len(traj_evals[i]) >= turn_idx:
                    prev_e = traj_evals[i][turn_idx - 1]
                    if prev_e and prev_e.get("correct") and prev_e.get("profiler_feedback"):
                        traj_prof = prev_e["profiler_feedback"]
                if traj_prof is None:
                    traj_prof = ws_best_eval.get("profiler_feedback") if ws_best_eval else None
                profiler_str = f"\n\n{traj_prof}" if traj_prof else ""

                # ── Delta feedback from previous optimization attempt ─────
                # Includes: rich error detail (Bug 5), regression warning
                # (Bug 6), stuck detection (Bug 3), actual error messages (Bug 4).
                delta_str = ""
                prev_opt_eval = traj_evals[i][turn_idx - 1] if turn_idx > 1 and len(traj_evals[i]) >= turn_idx else None
                # Filter OPT rules by profiler bottleneck AND operation type —
                # don't assign shared memory tiling to elementwise ops, etc.
                # Uses the actual generated code (ws_best_code) for classification,
                # not just prompt keywords — generalizes to any operation.
                _bottleneck = _get_bottleneck_type(traj_prof or "")
                _op_type = _get_operation_type(prompt_text, cuda_code=ws_best_code)
                _filtered_rules = _filter_rules_by_bottleneck(_bottleneck, _op_type)

                if prev_opt_eval is not None:
                    prev_rule = _filtered_rules[(i + turn_idx - 2) % len(_filtered_rules)]
                    technique_ref = f"'{prev_rule['name']}'"

                    # Bug 3: stuck detection — same error class 2 turns in a row
                    stuck_str = ""
                    if turn_idx > 2 and len(traj_evals[i]) >= turn_idx:
                        prev_prev = traj_evals[i][turn_idx - 2]
                        if (prev_prev is not None
                                and _classify_error(prev_opt_eval) == _classify_error(prev_prev)
                                and _classify_error(prev_opt_eval) != "correct"):
                            stuck_str = (
                                "You made the SAME TYPE of error two turns in a row. "
                                "Try a COMPLETELY DIFFERENT approach. "
                            )

                    if not prev_opt_eval.get("compiles", False):
                        # Bug 4+5: include actual compiler error
                        err_msg = prev_opt_eval.get("compiler_error") or "Unknown error"
                        # Truncate to avoid flooding context
                        if len(err_msg) > 500:
                            err_msg = err_msg[:500] + "..."
                        delta_str = (
                            f"\n\n⚠ REGRESSION: Last turn {technique_ref} — "
                            f"it BROKE COMPILATION of your working kernel.\n"
                            f"{stuck_str}"
                            f"Error: {err_msg}"
                        )
                    elif not prev_opt_eval.get("correct", False):
                        # Bug 4+5+6: include rich error detail from sandbox
                        err_msg = prev_opt_eval.get("compiler_error") or "Outputs do not match reference"
                        if len(err_msg) > 500:
                            err_msg = err_msg[:500] + "..."
                        wf = prev_opt_eval.get("wrong_frac")
                        mae = prev_opt_eval.get("max_abs_error")
                        bias = prev_opt_eval.get("systematic_bias")
                        detail_parts = []
                        if wf is not None:
                            detail_parts.append(f"{wf*100:.0f}% of elements wrong")
                        if mae is not None:
                            detail_parts.append(f"max_abs_error={mae:.5f}")
                        if bias is not None and abs(bias) > 0.01:
                            detail_parts.append(f"systematic_bias={bias:+.4f} ({'too high' if bias > 0 else 'too low'})")
                        detail_suffix = " | ".join(detail_parts)
                        delta_str = (
                            f"\n\n⚠ REGRESSION: Last turn {technique_ref} — "
                            f"it BROKE CORRECTNESS of your working kernel.\n"
                            f"{stuck_str}"
                            f"{err_msg}\n"
                            f"{detail_suffix}"
                        )
                    else:
                        # Correct — compare against frozen baseline (Bug 1)
                        prev_rt = prev_opt_eval.get("runtime_ms")
                        baseline_sp = ws_speedup_at_opt_start or ws_best_speedup
                        if prev_rt and bt:
                            prev_sp = bt / prev_rt
                            if prev_sp > baseline_sp * 1.05:
                                delta_str = (
                                    f"\n\n✓ Last turn {technique_ref} — "
                                    f"it IMPROVED speed to {prev_sp:.2f}x "
                                    f"(started at {baseline_sp:.2f}x). Good technique."
                                )
                            elif prev_sp < baseline_sp * 0.95:
                                delta_str = (
                                    f"\n\n✗ Last turn {technique_ref} — "
                                    f"it made the kernel SLOWER at {prev_sp:.2f}x "
                                    f"(started at {baseline_sp:.2f}x). Avoid that approach."
                                )
                            else:
                                delta_str = (
                                    f"\n\n→ Last turn {technique_ref} — "
                                    f"no significant speed change ({prev_sp:.2f}x vs "
                                    f"started at {baseline_sp:.2f}x). Try a different technique."
                                )

                # Assign a different optimization rule — rotate by BOTH
                # trajectory index AND turn so each traj tries a different
                # technique each turn instead of the same one every time.
                # Rules are filtered by profiler bottleneck type.
                rule = _filtered_rules[(i + turn_idx - 1) % len(_filtered_rules)]

                # LLM hints for even-indexed trajectories (with per-traj
                # technique + varied temperature for diversity).
                # Odd-indexed trajectories use hardcoded rule rotation
                # to maintain exploration diversity across the group.
                _LLM_OPT_TEMPS = [0.3, 0.5, 0.7, 0.9]
                _use_llm_for_traj = (i % 2 == 0) and _llm_feedback is not None and _llm_feedback.available

                _opt_llm_hint = None
                if _use_llm_for_traj:
                    _opt_speedup = 0.0
                    if rt and bt and rt > 0:
                        _opt_speedup = bt / rt
                    _opt_code = ws_best_code or ""
                    _traj_temp = _LLM_OPT_TEMPS[i % len(_LLM_OPT_TEMPS)]
                    _opt_llm_hint = _llm_feedback.suggest_optimization(
                        task=prompt_text,
                        code=_opt_code,
                        speedup=_opt_speedup,
                        profiler_info=traj_prof or "",
                        technique_hint=rule["name"],
                        temperature=_traj_temp,
                    )
                    if i == 0:
                        print(f"  [LLM-OPT] traj=0 turn {turn_idx+1}: "
                              f"{'got hint' if _opt_llm_hint else 'no response, using rules'}"
                              f" (technique={rule['name']}, temp={_traj_temp})",
                              flush=True)

                if _opt_llm_hint:
                    # Use LLM-generated optimization hint (targeted to rule)
                    rule_str = _format_llm_hint(_opt_llm_hint, "optimization")
                    traj_turn_feedback_source[i] = {
                        "type": "OPT-LLM",
                        "rule": rule["name"],
                        "bottleneck": _bottleneck,
                        "op_type": _op_type,
                        "profiler": bool(traj_prof),
                    }
                else:
                    # Hardcoded rules (odd trajectories, or LLM fallback)
                    traj_turn_feedback_source[i] = {
                        "type": "OPT",
                        "rule": rule["name"],
                        "bottleneck": _bottleneck,
                        "op_type": _op_type,
                        "profiler": bool(traj_prof),
                    }
                    rule_str = (
                        f"\n\n--- Optimization Technique to Apply ---\n"
                        f"**{rule['name']}**\n{rule['rule']}\n"
                        f"---\n"
                        f"Apply this technique to the kernel above. "
                        f"Keep the kernel correct while making it faster.\n"
                        f"IMPORTANT: If this technique does not naturally fit your kernel "
                        f"(e.g., no inner loops to unroll, no data reuse for shared memory), "
                        f"skip it and instead do minor tuning: adjust block/grid sizes, "
                        f"add __ldg() for read-only data, or try #pragma unroll on any existing loops."
                    )

                feedback = (
                    f"Your previous answer was correct.\n{timing_str}{profiler_str}"
                    f"{delta_str}"
                    f"{rule_str}\n\n"
                    "Apply the optimization above to improve speed. "
                    "Generate the complete improved code."
                )

                if i == 0:
                    _hint_source = f"LLM(rule={rule['name']})" if _opt_llm_hint else f"rule='{rule['name']}'"
                    print(f"  [OPT TURN] traj=0 turn {turn_idx+1}: "
                          f"best kernel from {ws_best_source} ({ws_best_speedup:.2f}x), "
                          f"{_hint_source}"
                          f"{', delta: ' + delta_str.strip()[:200] if delta_str else ''}")
                    print(f"  [DEBUG] Feedback traj=0 →{turn_idx+1}:\n{feedback[:800]}...")
                msgs.append({"role": "user", "content": feedback})
            else:
                # Normal path (turn 1, or turn 2+ with no correct kernel yet):
                # build context from this trajectory's own history
                for t in range(turn_idx):
                    # Only keep full code for the most recent prior turn;
                    # older turns get code truncated to save context tokens.
                    is_old_turn = (t < turn_idx - 1)
                    stripped = _strip_thinking(traj_responses[i][t], truncate_code=is_old_turn)
                    if i == 0:
                        has_ref = "Reflection:" in stripped
                        print(f"  [DEBUG] Turn {t+1}→{turn_idx+1} traj=0: "
                              f"reflection={'YES' if has_ref else 'NO'}, stripped_len={len(stripped)} chars"
                              f"{' (truncated)' if is_old_turn else ''}")
                    msgs.append({"role": "assistant", "content": stripped})
                    prev_eval = traj_evals[i][t - 1] if t > 0 else None
                    # Compute high water mark up to turn t for feedback context
                    hwm_speedup, hwm_turn = None, None
                    for pt in range(t):
                        pe = traj_evals[i][pt]
                        if pe and pe.get("correct") and pe.get("baseline_runtime_ms") and pe.get("runtime_ms"):
                            sp = pe["baseline_runtime_ms"] / pe["runtime_ms"]
                            if hwm_speedup is None or sp > hwm_speedup:
                                hwm_speedup, hwm_turn = sp, pt + 1
                    # Feedback hints: LLM diagnosis (if available) or BM25 RAG
                    # Dropout: rag_prob decays from 1.0→0.0 over training so model
                    # learns patterns early but doesn't depend on hints later.
                    _rag_hint = None
                    _rag_sections = []
                    _llm_hint = None
                    _eval_t = traj_evals[i][t]
                    _is_error = (_eval_t is None or not _eval_t.get("correct", False))
                    if _is_error:
                        # Extract code for diagnosis
                        _code = ""
                        if t < len(traj_responses[i]) and traj_responses[i][t]:
                            _code = _extract_python_block(traj_responses[i][t])

                        # LLM feedback ALWAYS fires for errors (not gated by rag_prob)
                        # Use pre-computed deduplicated cache for the most recent
                        # prior turn; fall back to direct call for older turns.
                        if _llm_feedback is not None and _llm_feedback.available:
                            _ec = _classify_error(_eval_t)
                            if t == turn_idx - 1 and _ec in _diag_cache:
                                # Reuse cached diagnosis (deduplicated)
                                _rag_hint = _diag_cache[_ec]
                                _llm_hint = _rag_hint  # for printing
                            else:
                                # Direct call for older turns or cache miss
                                _error_msg = ""
                                if _eval_t is None:
                                    _error_msg = "No code was extracted from the response"
                                elif not _eval_t.get("compiles", False):
                                    _error_msg = _eval_t.get("compiler_error") or "Unknown compile error"
                                else:
                                    _error_msg = _eval_t.get("compiler_error") or "Outputs do not match reference"
                                    _wf = _eval_t.get("wrong_frac")
                                    _mae = _eval_t.get("max_abs_error")
                                    if _wf is not None:
                                        _error_msg += f" | {_wf*100:.0f}% of elements wrong"
                                    if _mae is not None:
                                        _error_msg += f" | max_abs_error={_mae:.5f}"

                                _llm_hint = _llm_feedback.diagnose_error(
                                    task=prompt_text,
                                    code=_code or "(no code extracted)",
                                    error=_error_msg,
                                )
                                if _llm_hint:
                                    _rag_hint = _format_llm_hint(_llm_hint, "diagnosis")
                            if i == 0:
                                _cache_hit = (t == turn_idx - 1 and _ec in _diag_cache)
                                print(f"  [LLM] traj=0 turn {t+1}→{turn_idx+1}: "
                                      f"{'got diagnosis' if _llm_hint else 'no response, falling back to RAG'}"
                                      f"{' (cached)' if _cache_hit else ''}",
                                      flush=True)

                        # Fallback to BM25 RAG if LLM unavailable or returned nothing
                        # RAG is gated by rag_prob dropout (decays over training)
                        if not _rag_hint and random.random() < rag_prob:
                            _rag_query = prompt_text
                            if _eval_t and _eval_t.get("compiler_error"):
                                _rag_query += "\n" + _eval_t["compiler_error"]
                            if _code:
                                _rag_query += "\n" + _code[:1000]
                            _rag_sections = _cuda_rag.retrieve(_rag_query, top_k=2)
                            _rag_hint = _cuda_rag.retrieve_text(_rag_query, top_k=2, max_chars=2000)
                            if i == 0 and _rag_sections:
                                print(f"  [RAG] traj=0 turn {t+1}→{turn_idx+1}: retrieved {len(_rag_sections)} sections:", flush=True)
                                for _rs in _rag_sections:
                                    print(f"    → '{_rs.title}' ({len(_rs.content)} chars)", flush=True)
                            elif i == 0:
                                print(f"  [RAG] traj=0 turn {t+1}→{turn_idx+1}: skipped (correct={_eval_t.get('correct', False) if _eval_t else None}, rag_prob={rag_prob:.2f})", flush=True)

                    # LLM optimization hint for correct-but-improvable kernels
                    _opt_hint = None
                    if (_eval_t is not None and _eval_t.get("correct", False)
                            and _llm_feedback is not None and _llm_feedback.available):
                        _speedup = 0.0
                        _rt = _eval_t.get("runtime_ms")
                        _bt = _eval_t.get("baseline_runtime_ms")
                        if _rt and _bt and _rt > 0:
                            _speedup = _bt / _rt
                        _opt_code = ""
                        if t < len(traj_responses[i]) and traj_responses[i][t]:
                            _opt_code = _extract_python_block(traj_responses[i][t])
                        _opt_hint = _llm_feedback.suggest_optimization(
                            task=prompt_text,
                            code=_opt_code or "(no code extracted)",
                            speedup=_speedup,
                            profiler_info=_eval_t.get("profiler_feedback") or "",
                        )
                        if _opt_hint:
                            _opt_hint = _format_llm_hint(_opt_hint, "optimization")
                        if i == 0:
                            print(f"  [LLM-OPT] traj=0 turn {t+1}→{turn_idx+1}: "
                                  f"{'got optimization hint' if _opt_hint else 'no response'}",
                                  flush=True)

                    _has_profiler = bool(_eval_t and _eval_t.get("profiler_feedback"))
                    _rag_titles = [_rs.title for _rs in _rag_sections] if _rag_sections else []
                    traj_turn_feedback_source[i] = {
                        "type": "LLM" if (_llm_hint or _opt_hint) else "NORMAL",
                        "profiler": _has_profiler,
                        "rag": _rag_titles,
                        "llm_diagnosis": bool(_llm_hint),
                        "llm_optimization": bool(_opt_hint),
                        "correct_prev": bool(_eval_t and _eval_t.get("correct")),
                    }
                    feedback = _build_turn_feedback(
                        _eval_t, prev_eval=prev_eval,
                        best_speedup=hwm_speedup, best_turn=hwm_turn,
                        profiler_feedback=_eval_t.get("profiler_feedback") if _eval_t else None,
                        group_error_summary=group_error_summary,
                        rag_hint=_rag_hint,
                        opt_hint=_opt_hint,
                    )
                    if i == 0:
                        _fb_has_llm = "Bug Diagnosis" in feedback or "Optimization Hint" in feedback
                        _fb_has_rag = "--- Relevant CUDA Pattern ---" in feedback
                        _fb_has_group = "ALL" in feedback and "FAILED" in feedback
                        _fb_has_stuck = "same type of error" in feedback
                        print(f"  [DEBUG] Feedback traj=0 turn {t+1}→{turn_idx+1} "
                              f"({len(feedback)} chars, LLM={'YES' if _fb_has_llm else 'NO'}, "
                              f"RAG={'YES' if _fb_has_rag else 'NO'}, "
                              f"group_fail={'YES' if _fb_has_group else 'NO'}, "
                              f"stuck={'YES' if _fb_has_stuck else 'NO'}):\n{feedback}", flush=True)
                    elif (traj_evals[i][t] is not None
                          and not traj_evals[i][t].get("correct", False)
                          and t == turn_idx - 1):
                        # Print feedback for first wrong traj so we see what error msg it received
                        is_first_wrong = not any(
                            traj_evals[j][t] is not None
                            and not traj_evals[j][t].get("correct", False)
                            and t == turn_idx - 1
                            for j in range(1, i)
                        )
                        if is_first_wrong:
                            print(f"  [DEBUG] Feedback traj={i} turn {t+1}→{turn_idx+1}:\n{feedback}")
                    msgs.append({"role": "user", "content": feedback})

            ctx_str = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            # Qwen3 thinking mode: append <think> so model enters reasoning before code
            ctx_str += "<think>\n"
            if i == 0:
                ctx_tokens = len(tokenizer(ctx_str).input_ids)
                print(f"  [DEBUG] Context traj=0 turn {turn_idx+1}: {ctx_tokens} tokens"
                      f"{' (opt turn)' if use_optimization_turn else ''}")
            context_texts.append(ctx_str)

        # Optimization turn summary
        if use_optimization_turn:
            # Determine bottleneck and op type from best kernel's profiler for the summary
            _sum_prof = ws_best_eval.get("profiler_feedback", "") if ws_best_eval else ""
            _sum_bottleneck = _get_bottleneck_type(_sum_prof)
            _sum_op_type = _get_operation_type(prompt_text)
            _sum_filtered = _filter_rules_by_bottleneck(_sum_bottleneck, _sum_op_type)
            print(f"  [OPT TURN] Turn {turn_idx+1}: ALL {G} trajectories optimizing "
                  f"best kernel from {ws_best_source} ({ws_best_speedup:.2f}x) "
                  f"[bottleneck={_sum_bottleneck}, op_type={_sum_op_type}, "
                  f"{len(_sum_filtered)}/{len(OPTIMIZATION_RULES)} rules]")
            rules_used = [_sum_filtered[(i + turn_idx - 1) % len(_sum_filtered)]["name"] for i in range(G)]
            _llm_avail = _llm_feedback is not None and _llm_feedback.available
            _llm_trajs = [j for j in range(G) if j % 2 == 0 and _llm_avail]
            _rule_trajs = [j for j in range(G) if j not in _llm_trajs]
            print(f"  [OPT TURN] Rules assigned: {rules_used}")
            print(f"  [OPT TURN] LLM trajs: {_llm_trajs}, Rule trajs: {_rule_trajs}")
        elif turn_idx > 0:
            print(f"  [TURN {turn_idx+1}] No correct kernel yet — normal error-feedback path")

        # Generate G completions for this turn
        if not config.mock_mode:
            if config.use_sglang and (SGLANG_AVAILABLE or config.sglang_python):
                t_gen = time.time()
                print(f"  [{turn_label}] Generating {G} responses...", end=" ", flush=True)
                raw_completions = _generate_with_sglang(context_texts, config)
                print(f"done ({time.time()-t_gen:.1f}s)")
                completions = [
                    full[len(ctx):] if full.startswith(ctx) else full
                    for ctx, full in zip(context_texts, raw_completions)
                ]
            else:
                tokenizer.padding_side = "left"
                inputs = tokenizer(context_texts, return_tensors="pt", padding=True)
                input_ids_tensor = inputs.input_ids.to(next(model.parameters()).device)
                attention_mask = inputs.attention_mask.to(next(model.parameters()).device)
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=input_ids_tensor,
                        attention_mask=attention_mask,
                        max_new_tokens=config.max_new_tokens,
                        temperature=config.temperature,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                completions = [
                    tokenizer.decode(outputs[i][input_ids_tensor[i].shape[0]:], skip_special_tokens=True)
                    for i in range(G)
                ]
        else:
            completions = [
                "```python\nimport torch\nimport torch.nn as nn\nclass ModelNew(nn.Module):\n    def forward(self, a): return a\n```"
            ] * G

        # Tokenize context + response for GRPO loss
        for i, (ctx_text, gen_text) in enumerate(zip(context_texts, completions)):
            if config.mock_mode:
                ctx_ids  = torch.tensor([1, 2, 3])
                resp_ids = torch.tensor([4, 5, 6])
            else:
                ctx_ids  = tokenizer(ctx_text, return_tensors="pt").input_ids[0]
                resp_ids = tokenizer(gen_text,  return_tensors="pt").input_ids[0]
            group_turns[i].append((ctx_ids.cpu(), resp_ids.cpu()))

        # ── DEBUG: dump model's thinking for traj 0 ──────────────────────────
        if completions:
            _raw0 = completions[0]
            # Extract think block — <think> may be in context prefix, so also
            # check for content before </think> without opening tag
            _think_match = re.search(r'<think>(.*?)</think>', _raw0, re.DOTALL)
            if not _think_match:
                _think_match = re.search(r'<think>(.*)', _raw0, re.DOTALL)
            if not _think_match:
                # <think> was in context prefix; completion starts with reasoning before </think>
                _think_match = re.search(r'^(.*?)</think>', _raw0, re.DOTALL)
            _think_text = _think_match.group(1).strip() if _think_match else "(no think block)"
            _think_tokens = len(_think_text.split())
            _total_tokens = len(_raw0.split())
            _code_text = _extract_python_block(_raw0)
            _code_tokens = len(_code_text.split()) if _code_text else 0
            print(f"  [THINK] traj=0 turn {turn_idx+1}: {_think_tokens} words thinking, "
                  f"{_code_tokens} words code, {_total_tokens} words total", flush=True)
            # Show first 500 chars of thinking so we can see the model's reasoning
            print(f"  [THINK CONTENT] traj=0:\n    {_think_text[:500]}", flush=True)
            if len(_think_text) > 500:
                print(f"    ... ({len(_think_text) - 500} more chars)", flush=True)
        # ── END DEBUG ─────────────────────────────────────────────────────────

        # Extract code and evaluate
        candidates = []
        for i, gen_text in enumerate(completions):
            model_new_py = _extract_python_block(gen_text)
            if model_new_py:
                # Fix common CUDA API mismatches before evaluation
                model_new_py = _fix_cuda_api(model_new_py)
            if i == 0:  # always print traj=0 every turn so we can trace the full repair loop
                print(f"  [CODE DUMP turn={turn_idx} traj=0]:\n{(model_new_py or gen_text)}")
            candidates.append(model_new_py if model_new_py else None)

        # Time ALL turns: rough timing (timed=True) on every turn so the model
        # gets speedup feedback throughout. Final turn uses more trials for accuracy.
        is_final_turn = (turn_idx == T - 1)
        n_valid = sum(1 for c in candidates if c is not None)
        t_eval = time.time()
        print(f"  [{turn_label}] Evaluating {n_valid}/{G} valid kernels...", end=" ", flush=True)
        with ProcessPoolExecutor(max_workers=min(G, 16), mp_context=_MP_SPAWN_CTX) as pool:
            eval_results = list(pool.map(
                _worker_run_eval,
                [(c, prompt_text, True, 10 if is_final_turn else 2) for c in candidates],
            ))
        n_compiled = sum(1 for r in eval_results if r is not None and r.get("compiles", False))
        n_correct  = sum(1 for r in eval_results if r is not None and r.get("correct",  False))
        print(f"done ({time.time()-t_eval:.1f}s) | compiled={n_compiled}/{n_valid} correct={n_correct}/{G}")

        # ── DEBUG: when all fail on turn 2+, show what each traj was thinking ──
        if turn_idx > 0 and n_correct == 0:
            print(f"  [ALL FAIL DEBUG] Turn {turn_idx+1}: 0/{G} correct. Think summaries:", flush=True)
            for i, gen_text in enumerate(completions):
                _tm = re.search(r'<think>(.*?)</think>', gen_text, re.DOTALL)
                if not _tm:
                    _tm = re.search(r'<think>(.*)', gen_text, re.DOTALL)
                if not _tm:
                    # <think> was in context prefix; completion starts inside think block
                    _tm = re.search(r'^(.*?)</think>', gen_text, re.DOTALL)
                _tt = _tm.group(1).strip()[:200] if _tm else "(no think)"
                _err = "compile" if (eval_results[i] and not eval_results[i].get("compiles")) else \
                       "wrong" if (eval_results[i] and not eval_results[i].get("correct")) else \
                       "no_code" if not candidates[i] else "?"
                print(f"    traj {i} [{_err}]: {_tt}", flush=True)

        # ── DEBUG: detect when model repeats itself across turns ──
        if turn_idx > 0:
            n_repeated = 0
            for i in range(G):
                if candidates[i] and len(traj_evals[i]) > 0:
                    prev_code = None
                    for t in range(turn_idx - 1, -1, -1):
                        if t < len(traj_responses[i]):
                            prev_code = _extract_python_block(traj_responses[i][t])
                            if prev_code:
                                break
                    if prev_code and candidates[i]:
                        # Normalize whitespace for comparison
                        _prev_norm = re.sub(r'\s+', ' ', prev_code.strip())
                        _curr_norm = re.sub(r'\s+', ' ', candidates[i].strip())
                        if _prev_norm == _curr_norm:
                            n_repeated += 1
            if n_repeated > 0:
                print(f"  [REPEAT] Turn {turn_idx+1}: {n_repeated}/{G} trajectories produced IDENTICAL code to their previous turn", flush=True)

        # OOM detection: if ALL evals on turn 1 fail (often means ref model itself OOMs),
        # skip this prompt entirely — no useful learning signal.
        if turn_idx == 0 and n_compiled == 0:
            oom_count = sum(
                1 for r in eval_results
                if r is not None and "OutOfMemory" in (r.get("compiler_error") or "")
            )
            if oom_count > 0:
                print(f"  [SKIP] {oom_count}/{G} trajectories OOM on turn 1 — prompt too heavy, skipping.")
                for i in range(G):
                    group_rewards[i].extend([config.reward_compile_fail] * T)
                    for _ in range(T):
                        dummy_ids = torch.zeros(1, dtype=torch.long)
                        group_turns[i].append((dummy_ids, dummy_ids))
                return group_turns, group_rewards

        # Compile errors (all turns)
        for i, res in enumerate(eval_results):
            if res and not res.get("compiles", False) and res.get("compiler_error"):
                print(f"  [COMPILE ERROR traj={i}]: {res['compiler_error']}")

        # First wrong (compiled-but-incorrect) trajectory: print error + code so we can see
        # exactly what the kernel computed and what the sandbox told it.
        wrong_traj = next(
            (i for i in range(G)
             if eval_results[i] is not None
             and eval_results[i].get("compiles", False)
             and not eval_results[i].get("correct", False)),
            None,
        )
        if wrong_traj is not None:
            err = eval_results[wrong_traj].get("compiler_error") or "no error detail"
            print(f"  [WRONG traj={wrong_traj} turn={turn_idx+1}] {err}")
            if candidates[wrong_traj]:
                print(f"  [CODE DUMP turn={turn_idx} traj={wrong_traj}]:\n{candidates[wrong_traj]}")

        # Compute per-turn rewards
        # Penalty scale: easier problems penalized more, harder problems forgiven more
        # Level 1: 1.0x penalty, Level 2: 0.75x, Level 3: 0.5x
        penalty_scale = {1: 1.0, 2: 0.75, 3: 0.5}.get(difficulty, 1.0)
        for i, (eval_res, gen_text) in enumerate(zip(eval_results, completions)):
            if candidates[i] is None:
                r = config.reward_no_code * penalty_scale
            elif eval_res is None or not eval_res.get("compiles", False):
                r = config.reward_compile_fail * penalty_scale
            elif not eval_res["correct"]:
                r = calculate_wrong_reward(eval_res) * penalty_scale
            else:
                # Use opt reward during optimization turns for speed gradient
                if use_optimization_turn and ws_speedup_at_opt_start and ws_speedup_at_opt_start > 0:
                    r = calculate_opt_reward(eval_res, ws_speedup_at_opt_start) * diff_scale
                else:
                    r = calculate_reward(eval_res) * diff_scale  # scale by difficulty
                # Update high water mark for this trajectory
                rt = eval_res.get("runtime_ms")
                bt = eval_res.get("baseline_runtime_ms")
                if rt and bt:
                    sp = bt / rt
                    if traj_best_speedup[i] is None or sp > traj_best_speedup[i]:
                        traj_best_speedup[i] = sp
                        traj_best_turn[i] = turn_idx + 1
                rt_str = f"{eval_res['runtime_ms']:.3f}ms" if eval_res.get('runtime_ms') is not None else "no timing"
                sp_str = f" ({bt/rt:.2f}x)" if (rt and bt) else ""
                opt_tag = " [OPT]" if use_optimization_turn else ""
                print(f"    ✅ Turn {turn_idx+1} Traj {i}: {rt_str}{sp_str} reward={r:.2f}{opt_tag}")

            # ── SCoRe reward shaping: bonus/penalty for improvement/regression ──
            is_correct = (eval_res is not None
                          and eval_res.get("compiles", False)
                          and eval_res.get("correct", False))
            score_tag = ""
            if turn_idx > 0:
                if traj_was_correct[i] and not is_correct:
                    # Regression: was correct before, now broken → extra penalty
                    r -= config.score_regression_penalty
                    score_tag = f" [REG -{config.score_regression_penalty}]"
                elif is_correct and r > traj_best_reward[i]:
                    # Improvement: correct AND better than previous best → bonus
                    r += config.score_improvement_bonus
                    score_tag = f" [IMP +{config.score_improvement_bonus}]"
            if score_tag:
                print(f"      SCoRe traj {i}: {r:.2f}{score_tag}")

            # Update SCoRe tracking
            if is_correct:
                traj_was_correct[i] = True
            if r > traj_best_reward[i]:
                traj_best_reward[i] = r

            group_rewards[i].append(r)

        # ── DEBUG: reward breakdown this turn ───────────────────────────────
        turn_rewards = [group_rewards[i][turn_idx] for i in range(G)]
        n_correct = sum(1 for r in turn_rewards if r > 0)
        n_wrong = sum(1 for r in turn_rewards if r <= 0)
        print(f"  [DEBUG] Turn {turn_idx+1} rewards: correct={n_correct} wrong={n_wrong} "
              f"| values=[{', '.join(f'{r:.2f}' for r in turn_rewards)}] "
              f"std={torch.tensor(turn_rewards).std().item():.3f}")

        # ── FEEDBACK→OUTCOME: track what advice each traj got and what happened ──
        if turn_idx > 0:
            print(f"  [FEEDBACK→OUTCOME] Turn {turn_idx+1}:", flush=True)
            for i in range(G):
                fb = traj_turn_feedback_source[i]
                er = eval_results[i]
                # Determine outcome
                if er is None or not er.get("compiles", False):
                    outcome = "COMPILE_FAIL"
                elif not er.get("correct", False):
                    outcome = "WRONG"
                else:
                    rt = er.get("runtime_ms")
                    bt = er.get("baseline_runtime_ms")
                    if rt and bt:
                        sp = bt / rt
                        prev_sp = ws_speedup_at_opt_start or ws_best_speedup or 0
                        if prev_sp > 0 and sp > prev_sp * 1.02:
                            outcome = f"IMPROVED ({sp:.2f}x vs {prev_sp:.2f}x)"
                        elif prev_sp > 0 and sp < prev_sp * 0.98:
                            outcome = f"SLOWER ({sp:.2f}x vs {prev_sp:.2f}x)"
                        else:
                            outcome = f"SAME ({sp:.2f}x)"
                    else:
                        outcome = "CORRECT (no timing)"

                if fb.get("type") == "OPT":
                    print(f"    traj {i}: rule='{fb['rule']}' bottleneck={fb['bottleneck']} op_type={fb.get('op_type','?')} → {outcome}", flush=True)
                elif fb.get("type") == "NORMAL":
                    rag_str = f" RAG={fb['rag']}" if fb.get('rag') else ""
                    prof_str = " +profiler" if fb.get('profiler') else ""
                    print(f"    traj {i}: NORMAL (prev_correct={fb.get('correct_prev')}){prof_str}{rag_str} → {outcome}", flush=True)
                else:
                    print(f"    traj {i}: (no feedback tracked) → {outcome}", flush=True)
        # ── END FEEDBACK→OUTCOME ────────────────────────────────────────────
        # ── END DEBUG ────────────────────────────────────────────────────────

        # Run profiler on correct kernels when there's a next turn to use the feedback.
        # Profile the BEST correct kernel (highest speedup) so warm start gets profiler data.
        # Only on non-final turns since profiler feedback is used for next turn's context.
        if not is_final_turn and not config.mock_mode:
            # Find the best correct kernel this turn
            profiled_idx, profiled_sp = None, 0.0
            for i in range(G):
                er = eval_results[i]
                if er and er.get("correct", False) and candidates[i]:
                    rt = er.get("runtime_ms")
                    bt = er.get("baseline_runtime_ms")
                    sp = bt / rt if rt and bt and rt > 0 else 0.0
                    if sp > profiled_sp:
                        profiled_sp = sp
                        profiled_idx = i
            if profiled_idx is not None:
                try:
                    t_prof = time.time()
                    prof_fb = profile_kernel(candidates[profiled_idx], prompt_text, speedup=profiled_sp)
                    # Store profiler feedback in eval result so warm start and _build_turn_feedback can use it
                    eval_results[profiled_idx]["profiler_feedback"] = prof_fb
                    print(f"  [PROFILER traj={profiled_idx} ({profiled_sp:.2f}x)] ({time.time()-t_prof:.1f}s):\n{prof_fb}")
                except Exception as e:
                    print(f"  [PROFILER] Failed: {e}")

        # ── Warm start: update overall best if this turn produced a faster kernel ──
        for i in range(G):
            er = eval_results[i]
            if er and er.get("correct", False) and candidates[i]:
                rt = er.get("runtime_ms")
                bt = er.get("baseline_runtime_ms")
                sp = bt / rt if rt and bt and rt > 0 else 0.0
                if sp > ws_best_speedup:
                    ws_best_speedup = sp
                    ws_best_code = candidates[i]
                    ws_best_eval = eval_results[i]
                    ws_best_completion = completions[i]
                    ws_best_source = f"turn {turn_idx+1} traj {i}"
        if ws_best_code is not None:
            n_failed = sum(1 for er in eval_results if er is None or not er.get("correct", False))
            if n_failed > 0:
                print(f"  [WARM START] Overall best: {ws_best_source} "
                      f"({ws_best_speedup:.2f}x) — {n_failed} failed trajectories will receive it next turn")
        # ── END warm start ────────────────────────────────────────────────────

        # Store for next-turn context building
        for i in range(G):
            traj_responses[i].append(completions[i])
            traj_evals[i].append(eval_results[i])

        # ── Compute group error summary when ALL trajectories failed ────────
        # This lets each trajectory learn from ALL failures, not just its own.
        all_failed = all(
            er is None or not er.get("correct", False)
            for er in eval_results
        )
        if all_failed and turn_idx < T - 1:  # no point on last turn
            error_classes = {}
            for idx_e, er in enumerate(eval_results):
                cls = _classify_error(er)
                if cls not in error_classes:
                    error_classes[cls] = []
                if er is None:
                    error_classes[cls].append(f"traj {idx_e}: format/parse error")
                elif not er.get("compiles", False):
                    raw = er.get("compiler_error", "unknown")
                    first_line = raw.split('\n')[0][:120]
                    error_classes[cls].append(f"traj {idx_e}: {first_line}")
                elif not er.get("correct", False):
                    wf = er.get("wrong_frac")
                    mae = er.get("max_abs_error")
                    bias = er.get("systematic_bias")
                    parts = []
                    if wf is not None: parts.append(f"{wf*100:.0f}% wrong")
                    if mae is not None: parts.append(f"max_err={mae:.4f}")
                    if bias and abs(bias) > 0.01: parts.append(f"bias={bias:+.3f}")
                    error_classes[cls].append(f"traj {idx_e}: {', '.join(parts) or 'incorrect'}")
            summary_lines = [f"--- ALL {G} trajectories FAILED this turn ---"]
            summary_lines.append("Common failure patterns (avoid ALL of these):")
            for cls, examples in error_classes.items():
                summary_lines.append(f"  [{cls.upper()}] ({len(examples)} trajs): {examples[0]}")
                if len(examples) > 1:
                    summary_lines.append(f"    + {len(examples)-1} more with same error type")
            group_error_summary = "\n".join(summary_lines)
            print(f"  [GROUP FAIL] {group_error_summary}")
        else:
            group_error_summary = None

    # ── DEBUG: final reward matrix ───────────────────────────────────────────
    print(f"  [DEBUG] Reward matrix [G={G} x T={T}]:")
    for i in range(G):
        print(f"    traj {i}: {[f'{r:.2f}' for r in group_rewards[i]]}")
    # ── END DEBUG ────────────────────────────────────────────────────────────

    return group_turns, group_rewards


# ---------------------------------------------------------------------------
# GRPO core
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 8192


def _get_token_log_probs(
    model,
    context_ids: torch.Tensor,
    response_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Forward pass for one turn. Returns per-token log probs [R].
    Truncates response from the right and context from the left so total
    sequence never exceeds MAX_SEQ_LEN (prevents OOM on long think+code responses).
    With reflection-only inter-turn context (Kevin's approach), contexts stay
    small (~1000-1300 tokens across all turns) so truncation rarely triggers.
    """
    max_resp = MAX_SEQ_LEN - 64
    if len(response_ids) > max_resp:
        response_ids = response_ids[:max_resp]
    R = len(response_ids)
    max_ctx = max(MAX_SEQ_LEN - R, 64)
    if len(context_ids) > max_ctx:
        context_ids = context_ids[-max_ctx:]

    device = next(model.parameters()).device
    input_ids = torch.cat([context_ids, response_ids]).unsqueeze(0).to(device)
    Q = len(context_ids)
    r_ids = response_ids.to(device)

    with torch.no_grad() if not torch.is_grad_enabled() else torch.enable_grad():
        outputs = model(input_ids=input_ids)

    resp_logits = outputs.logits[0, Q - 1 : Q + R - 1, :]  # [R, V]
    log_probs = F.log_softmax(resp_logits, dim=-1)           # [R, V]
    token_log_probs = log_probs[range(R), r_ids]              # [R]
    # Real per-token entropy H = -sum_v p(v) log p(v), mean over response tokens
    entropy = -(log_probs.exp() * log_probs).sum(-1).mean()  # scalar

    return token_log_probs, entropy


def _compute_grpo_loss_and_backward(
    model,
    group_turns: list[list[TurnData]],
    group_rewards: list[list[float]],
    old_log_probs: list[list[torch.Tensor]],
    config: GRPOConfig,
) -> float:
    """
    GRPO loss with TRLOO (Dr. Kernel) leave-one-out advantages.
    Calls backward() one sequence at a time to avoid OOM from holding
    G×T=32 computation graphs simultaneously.

    1. Discounted return:   R_i_t = Σ_{k=t}^{T-1} γ^{k-t} * r_i_k
    2. TRLOO baseline:      b_i_t = mean_{j≠i}(R_j_t)  (unbiased leave-one-out)
    3. Advantage:            A_i_t = (R_i_t - b_i_t) / (std_{j≠i} + ε)
    4. Token-level DAPO Clip-Higher
    5. Dr. GRPO per-sequence mean
    6. Entropy bonus

    Returns scalar loss value for logging (gradients already applied).
    """
    G = len(group_rewards)
    T = len(group_rewards[0])

    # Discounted returns [G, T]
    rewards_t = torch.tensor(group_rewards, dtype=torch.float32)
    disc_returns = torch.zeros(G, T, dtype=torch.float32)
    for t in range(T):
        for k in range(t, T):
            disc_returns[:, t] += (config.gamma ** (k - t)) * rewards_t[:, k]

    # TRLOO (Dr. Kernel): leave-one-out baseline per trajectory per turn
    # For trajectory i, baseline = mean of all OTHER trajectories' returns at turn t
    # This is unbiased unlike group-mean which includes the trajectory itself
    advantages = torch.zeros(G, T, dtype=torch.float32)
    for t in range(T):
        group_sum = disc_returns[:, t].sum()
        group_sq_sum = (disc_returns[:, t] ** 2).sum()
        for i in range(G):
            loo_mean = (group_sum - disc_returns[i, t]) / max(G - 1, 1)
            loo_var = (group_sq_sum - disc_returns[i, t] ** 2) / max(G - 1, 1) - loo_mean ** 2
            # Floor std to prevent exploding advantages. Typical correct rewards
            # are 1.0-2.0 (speedup), so 0.5 floor keeps advantages meaningful
            # while preventing division-by-near-zero when all rewards are similar.
            loo_std = max(loo_var.clamp(min=0).sqrt().item(), 0.5)
            advantages[i, t] = (disc_returns[i, t] - loo_mean) / loo_std

    # ── DEBUG: discounted returns and advantages ─────────────────────────────
    print(f"  [DEBUG] disc_returns per turn (mean): {disc_returns.mean(dim=0).tolist()}", flush=True)
    print(f"  [DEBUG] disc_returns std per turn:    {disc_returns.std(dim=0).tolist()}", flush=True)
    print(f"  [DEBUG] advantages range: min={advantages.min().item():.3f} max={advantages.max().item():.3f}", flush=True)
    for t in range(T):
        print(f"  [DEBUG]   turn {t+1}: rewards={rewards_t[:, t].tolist()} → disc_ret={disc_returns[:, t].tolist()} → adv={advantages[:, t].tolist()}", flush=True)
    # ── END DEBUG ────────────────────────────────────────────────────────────

    # Count valid sequences upfront for loss normalization
    n_seqs = sum(
        1 for i in range(G)
        for (_, resp) in group_turns[i]
        if len(resp) > 0
    )
    if n_seqs == 0:
        return 0.0

    total_loss = 0.0

    for i in range(G):
        for turn_idx, (ctx, resp) in enumerate(group_turns[i]):
            if len(resp) == 0:
                continue

            A_i_t = advantages[i, turn_idx].item()

            new_lp, entropy = _get_token_log_probs(model, ctx, resp)
            old_lp = old_log_probs[i][turn_idx].to(new_lp.device)

            ratio   = torch.exp(new_lp - old_lp.detach())
            clipped = torch.clamp(ratio, 1.0 - config.cliprange_low, 1.0 + config.cliprange_high)

            seq_loss = -torch.min(ratio * float(A_i_t), clipped * float(A_i_t))
            loss_term = seq_loss.mean()

            if config.entropy_coef > 0:
                # Real H(π) = -Σ_v p(v) log p(v), computed in _get_token_log_probs
                loss_term = loss_term - config.entropy_coef * entropy

            # Divide by n_seqs so the sum of all backward() calls = proper mean loss.
            # Call backward immediately — frees this sequence's computation graph,
            # keeping peak activation memory to 1 sequence at a time.
            (loss_term / n_seqs).backward()
            total_loss += loss_term.item()

    return total_loss / n_seqs


def _run_evaluation(model, tokenizer, config: GRPOConfig, val_prompts: list[tuple]) -> dict:
    """Run an evaluation pass on the validation set."""
    print(f"\n--- Running Evaluation ({len(val_prompts)} prompts) ---")
    if config.mock_mode:
        return {"eval/pass_rate": 0.5, "eval/valid_rate": 0.8, "eval/avg_speedup": 1.1}

    model.eval()
    candidates_to_eval = []

    with torch.no_grad():
        for prompt_text, _level in val_prompts:
            user_msg = FORMAT_EXAMPLE + f"Reference Program:\n```python\n{prompt_text}\n```"
            sys_content = get_system_prompt().strip()
            messages = [
                {"role": "system", "content": sys_content},
                {"role": "user", "content": user_msg},
            ]
            prompt_str = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_str += "<think>\n"
            input_ids = tokenizer(prompt_str, return_tensors="pt").input_ids.to(model.device)
            
            # Generate single best guess (greedy, T=0.0)
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=config.max_new_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            resp_ids = output_ids[0][len(input_ids[0]):]
            response_text = tokenizer.decode(resp_ids, skip_special_tokens=True)

            model_new_py = _extract_python_block(PREFILL + response_text)
            candidates_to_eval.append((model_new_py or None, prompt_text))

    # Parallel eval over generated candidates
    success = 0
    valid = 0
    total_speedup = 0.0

    with ProcessPoolExecutor(max_workers=min(16, len(val_prompts)), mp_context=_MP_SPAWN_CTX) as pool:
        eval_results = list(pool.map(_worker_run_eval,
                                     [(c, p, True, 10) for c, p in candidates_to_eval]))
        
    for res in eval_results:
        if res is not None:
            valid += 1
            if res["correct"]:
                success += 1
                total_speedup += calculate_reward(res)
                
    total = len(val_prompts)
    results = {
        "eval/pass_rate": success / total,
        "eval/valid_rate": valid / total,
        "eval/avg_speedup": total_speedup / max(1, success)
    }
    
    print(f"Eval Results: pass_rate={results['eval/pass_rate']:.2%}, "
          f"valid_rate={results['eval/valid_rate']:.2%}, "
          f"avg_speedup={results['eval/avg_speedup']:.2f}x")
    model.train()
    return results


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config: GRPOConfig = None):
    if config is None:
        config = GRPOConfig()

    print(f"Loading dataset from {config.dataset_path}...")
    try:
        raw = load_dataset("json", data_files=config.dataset_path, split="train")
        raw = raw.filter(lambda x: bool(x.get("pytorch_code", "").strip()))
    except Exception as e:
        print(f"Failed to load dataset.\n{e}")
        return
    rows = list(raw)

    # ── DEBUG: dataset fields and level distribution ─────────────────────────
    print(f"\n[DEBUG] Dataset fields: {list(rows[0].keys())}")
    has_level = "level_id" in rows[0]
    if has_level:
        from collections import Counter
        level_counts = Counter(r.get("level_id", "?") for r in rows)
        print(f"[DEBUG] level_id distribution: {dict(sorted(level_counts.items()))}")
    else:
        print(f"[DEBUG] No level_id field — curriculum disabled, random shuffle")
    # ── END DEBUG ─────────────────────────────────────────────────────────────

    # Curriculum: sort by level_id (easy first) if available and enabled
    if config.curriculum and has_level:
        rows.sort(key=lambda r: (r.get("level_id", 99), random.random()))
        print(f"Curriculum enabled: sorted {len(rows)} tasks by level_id (easy-first).")
    else:
        random.shuffle(rows)

    # Filter broken reference prompts using a fast AST check (no subprocess/torch needed).
    # Catches the most common failure: get_init_inputs() references a module-level name
    # (e.g. `bias`) that was never defined, causing every generated kernel to fail with
    # a misleading NameError that has nothing to do with the kernel code itself.
    import ast, hashlib, json as _json

    _cache_path = os.path.join(os.path.dirname(config.dataset_path), ".ref_valid_cache.json")
    try:
        with open(_cache_path) as _f:
            _valid_cache = _json.load(_f)
    except Exception:
        _valid_cache = {}

    def _ref_is_runnable(pytorch_code: str) -> bool:
        h = hashlib.md5(pytorch_code.encode()).hexdigest()
        if h in _valid_cache:
            return _valid_cache[h]
        try:
            tree = ast.parse(pytorch_code)
        except SyntaxError:
            _valid_cache[h] = False
            return False
        # Collect all names defined at module level (assignments, function defs, class defs, imports)
        defined = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defined.add(node.name)
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        defined.add(t.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    defined.add(alias.asname or alias.name.split(".")[0])
        # Check that get_inputs and get_init_inputs are defined
        if "get_inputs" not in defined or "get_init_inputs" not in defined:
            _valid_cache[h] = False
            return False
        # Check for NameErrors inside get_init_inputs by finding all Name loads and
        # checking they are defined at module level or are builtins
        builtins = set(dir(__builtins__)) if isinstance(__builtins__, dict) else set(dir(__builtins__))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef,)) and node.name == "get_init_inputs":
                for child in ast.walk(node):
                    if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                        if child.id not in defined and child.id not in builtins:
                            _valid_cache[h] = False
                            return False
        _valid_cache[h] = True
        return True

    valid_rows = [r for r in rows if _ref_is_runnable(r["pytorch_code"])]
    n_broken = len(rows) - len(valid_rows)
    if n_broken:
        print(f"[Dataset] Filtered {n_broken} broken reference prompts. {len(valid_rows)} remain.")
    try:
        with open(_cache_path, "w") as _f:
            _json.dump(_valid_cache, _f)
    except Exception:
        pass
    rows = valid_rows

    # Cap dataset size for time-boxed runs
    if config.max_prompts > 0 and len(rows) > config.max_prompts:
        rows = rows[:config.max_prompts]
        print(f"[Dataset] Capped to {config.max_prompts} prompts (time-boxed run).")

    # Carry (prompt_text, level_id) so reward can scale by difficulty
    LEVEL_TO_INT = {"level_1": 1, "level_2": 2, "level_3": 3}
    prompts = [(row["pytorch_code"], LEVEL_TO_INT.get(row.get("level_id", ""), 1)) for row in rows]
    # Split to train/val (10% val, max 20)
    val_size = min(int(len(prompts) * 0.1), 20)
    if val_size == 0 and len(prompts) > 1:
        val_size = 1

    val_prompts = prompts[:val_size]
    train_prompts = prompts[val_size:]

    if len(train_prompts) == 0:
        # Fallback for dummy datasets
        train_prompts = prompts
        val_prompts = prompts

    print(f"Loaded {len(train_prompts)} train prompts, {len(val_prompts)} val prompts.")

    if not config.mock_mode:
        tokenizer = AutoTokenizer.from_pretrained(config.model_id, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
    
        # Load base model + SFT LoRA adapter
        print(f"Loading base model: {config.model_id}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="kernels-community/flash-attn2",
        )
        print(f"Loading SFT adapter: {config.adapter_path}...")
        # Fix accelerate bug: _no_split_modules can contain sets which are
        # unhashable, causing get_balanced_memory() to crash. Flatten to
        # a plain list of strings so set() works in accelerate internals.
        _nsm = getattr(base_model, '_no_split_modules', None)
        if _nsm is not None:
            flat = []
            for item in (_nsm if not isinstance(_nsm, str) else [_nsm]):
                if isinstance(item, (set, frozenset, list, tuple)):
                    flat.extend(str(x) for x in item)
                else:
                    flat.append(str(item))
            base_model._no_split_modules = flat
        model = PeftModel.from_pretrained(base_model, config.adapter_path, is_trainable=True)
        
        # Enable gradient checkpointing to save VRAM
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"Trainable: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.2f}%)")
    
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate)
    else:
        print("MOCK MODE: Skipping model and tokenizer loading.")
        class MockTokenizer:
            eos_token = "<|endoftext|>"
            eos_token_id = 0
            pad_token = "<|endoftext|>"
            pad_token_id = 0
            def save_pretrained(self, path): pass
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                return "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        tokenizer = MockTokenizer()
        
        # Simple mock model for parameter device tracking
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Parameter(torch.randn(10, 10))
            def forward(self, input_ids, **kwargs):
                B, S = input_ids.shape
                # return dummy logits of shape [1, S, Vocabulary]
                logits = torch.randn(B, S, 50256, device=self.dummy.device) * self.dummy.sum() * 0.0
                class Output:
                    pass
                out = Output()
                out.logits = logits
                return out
            def generate(self, *args, **kwargs): pass
            def save_pretrained(self, *args, **kwargs): pass
        model = MockModel().to(torch.device("cpu"))
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        trainable_params = list(model.parameters())

    os.makedirs(config.output_dir, exist_ok=True)

    # Cosine LR schedule with linear warmup — stabilizes early steps when advantages are noisy
    total_steps = max(1, (len(train_prompts) // config.batch_size) * config.num_train_epochs)
    warmup_steps = max(2, int(total_steps * config.warmup_pct))
    print(f"[LR] total_steps={total_steps}, warmup_steps={warmup_steps} ({config.warmup_pct*100:.0f}%)")

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))  # floor at 10% of peak lr

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(
        f"\n🚀 GRPO+DAPO Training\n"
        f"   group_size={config.group_size}, batch_size={config.batch_size}, grpo_epochs={config.grpo_epochs}\n"
        f"   clip=[{config.cliprange_low}, {config.cliprange_high}] (DAPO Clip-Higher)\n"
    )

    global_step = 0

    # Launch SGLang server if enabled
    if not config.mock_mode and config.use_sglang:
        if not SGLANG_AVAILABLE and not config.sglang_python:
            raise RuntimeError("use_sglang=True but sglang is not installed. Run: pip install sglang")
        import atexit
        global _sglang_server
        _sglang_server = launch_sglang_server(
            config.model_id, config.adapter_path, config.sglang_port, config.sglang_tp,
            sglang_python=config.sglang_python or None,
        )
        atexit.register(lambda: _sglang_server.terminate() if _sglang_server else None)
        print(f"[SGLang] Server running on port {config.sglang_port} (TP={config.sglang_tp})")

    if not config.mock_mode:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=config.__dict__,
        )

    total_steps = config.num_train_epochs * (len(train_prompts) // config.batch_size)
    step_bar = tqdm(total=total_steps, desc="GRPO", unit="step", dynamic_ncols=True)

    for epoch in range(config.num_train_epochs):
        epoch_prompts = train_prompts.copy()
        random.shuffle(epoch_prompts)

        for batch_start in range(0, len(epoch_prompts), config.batch_size):
            batch = epoch_prompts[batch_start : batch_start + config.batch_size]

            # ── Rollout: G trajectories per prompt ──────────────────────
            all_group_turns:   list[list[list[TurnData]]] = []       # [B][G][T]
            all_group_rewards: list[list[list[float]]] = []           # [B][G][T]

            n_degenerate = 0
            t0 = time.time()
            for p_idx, (prompt_text, level) in enumerate(batch):
                print(f"\n[Prompt {p_idx+1}/{len(batch)} level={level}] Generating {config.group_size} trajectories (batched/parallel)...")
                # RAG dropout: cosine decay from 100% → 0% over training
                # Cosine decays slowly early (model still needs hints) then drops fast at end
                _rag_progress = min(1.0, global_step / max(total_steps, 1))
                _rag_prob = max(0.0, 0.5 * (1.0 + math.cos(math.pi * _rag_progress)))
                group_turns, group_rewards = _run_group_episodes(prompt_text, model, tokenizer, config, difficulty=level, rag_prob=_rag_prob)

                # Dynamic Sampling: skip degenerate groups (DAPO).
                # Use mean-reward-per-trajectory as the degeneracy signal.
                # Threshold 0.05: groups where all trajectories get nearly identical
                # rewards (e.g., all fail the same way) produce near-zero advantages
                # and waste a gradient step.  Old threshold 1e-4 was far too low.
                if config.dynamic_sampling:
                    traj_means = [sum(rews) / len(rews) for rews in group_rewards]
                    reward_std = torch.tensor(traj_means).std().item()
                    if reward_std < 0.05:
                        n_degenerate += 1
                        print(f"  [Dynamic Sampling] Degenerate group (std={reward_std:.4f}), skipping prompt.")
                        continue

                all_group_turns.append(group_turns)
                all_group_rewards.append(group_rewards)

                traj_means = [sum(rews) / len(rews) for rews in group_rewards]
                mean_r = sum(traj_means) / len(traj_means)
                best_r = max(traj_means)
                reward_std = torch.tensor(traj_means).std().item()
                print(f"  Group: mean={mean_r:.2f}, best={best_r:.2f}, std={reward_std:.3f}")

            elapsed = time.time() - t0
            step_bar.update(1)

            if not all_group_rewards:
                print(f"\n[Epoch {epoch+1} | Step {global_step+1}/{total_steps}] all prompts degenerate, skipping step.")
                global_step += 1
                continue

            batch_mean = sum(
                sum(sum(rews) / len(rews) for rews in group) / len(group)
                for group in all_group_rewards
            ) / len(all_group_rewards)
            step_bar.set_postfix(reward=f"{batch_mean:.3f}", step_time=f"{elapsed:.0f}s")
            print(f"\n[Epoch {epoch+1} | Step {global_step+1}/{total_steps}] reward={batch_mean:.3f}  step_time={elapsed:.0f}s")

            torch.cuda.empty_cache()

            # ── Sleep SGLang to free VRAM for the backward pass ─────────
            if not config.mock_mode and config.use_sglang:
                release_sglang_memory(config.sglang_port)

            # ── Collect old log probs (before gradient updates) ─────────
            model.eval()
            all_old_lps: list[list[list[torch.Tensor]]] = []  # [B][G][turns]

            for group_turns in all_group_turns:
                group_old = []
                for turns in group_turns:
                    turn_lps = []
                    for ctx, resp in turns:
                        if len(resp) == 0:
                            turn_lps.append(torch.zeros(0))
                        else:
                            with torch.no_grad():
                                lp, _ = _get_token_log_probs(model, ctx, resp)
                            turn_lps.append(lp.detach().cpu())
                    group_old.append(turn_lps)
                all_old_lps.append(group_old)
            model.train()

            # ── GRPO epochs ─────────────────────────────────────────────
            for grpo_ep in range(config.grpo_epochs):
                optimizer.zero_grad()
                total_loss_val = 0.0

                for group_turns, group_rewards, group_old_lps in zip(
                    all_group_turns, all_group_rewards, all_old_lps
                ):
                    # backward() called per-sequence inside to avoid OOM from
                    # holding G×T computation graphs simultaneously
                    loss_val = _compute_grpo_loss_and_backward(
                        model, group_turns, group_rewards, group_old_lps, config
                    )
                    total_loss_val += loss_val

                # ── DEBUG: gradient norm before clipping ────────────────────
                total_norm = torch.nn.utils.clip_grad_norm_(trainable_params, config.max_grad_norm)
                print(f"  [DEBUG] grad_norm={total_norm:.4f} (clip={config.max_grad_norm}) loss={total_loss_val:.4f}")
                # ── END DEBUG ────────────────────────────────────────────────
                optimizer.step()

            # One scheduler step per batch step (not per grpo_epoch) so LR
            # schedule completes in the expected number of training steps.
            scheduler.step()

            # ── Wake SGLang and hot-reload updated LoRA ──────────────────
            if not config.mock_mode and config.use_sglang:
                resume_sglang_memory(config.sglang_port)
                sync_lora_to_sglang(model, config.sglang_port)

            # Log metrics and print loss (always, not just when using SGLang)
            if not config.mock_mode:
                # Compute mean policy entropy from old log probs as collapse indicator
                all_lp_vals = [
                    lp for group in all_old_lps
                    for turn_lps in group
                    for lp in turn_lps
                    if len(lp) > 0
                ]
                mean_entropy = float(-torch.cat(all_lp_vals).mean()) if all_lp_vals else 0.0

                # Not-Okay Ratio: fraction of trajectories where final-turn reward < 0
                # High = model still mostly failing; should drop as training progresses.
                all_final_rewards = [
                    group[g][-1]
                    for group in all_group_rewards
                    for g in range(len(group))
                ]
                not_okay_ratio = sum(1 for r in all_final_rewards if r < 0) / max(1, len(all_final_rewards))

                wandb.log({
                    "train/loss": total_loss_val,
                    "train/batch_mean_reward": batch_mean,
                    "train/mean_entropy": mean_entropy,
                    "train/degenerate_groups": n_degenerate,
                    "train/not_okay_ratio": not_okay_ratio,
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "epoch": epoch + (global_step / max(1, len(train_prompts) // config.batch_size))
                }, step=global_step)

            print(f"  [Step {global_step+1}] loss={total_loss_val:.4f}")

            global_step += 1
            
            if global_step % config.eval_steps == 0:
                eval_metrics = _run_evaluation(model, tokenizer, config, val_prompts)
                if not config.mock_mode:
                    wandb.log(eval_metrics, step=global_step)

            if not config.mock_mode and global_step % config.save_steps == 0:
                ckpt = f"{config.output_dir}/step_{global_step}"
                model.save_pretrained(ckpt)
                tokenizer.save_pretrained(ckpt)
                print(f"  Checkpoint → {ckpt}")

    if not config.mock_mode:
        model.save_pretrained(config.output_dir)
        tokenizer.save_pretrained(config.output_dir)
        wandb.finish()
    print(f"\n✅ Training complete. Model saved to {config.output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="KernelForge GRPO+DAPO")
    parser.add_argument("--dataset", type=str, default="../sft/rl_prompts.jsonl")
    parser.add_argument("--output_dir", type=str, default="checkpoints/kernelforge_grpo")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--adapter", type=str, default="../sft/sft_qwen3_14b_lora")
    parser.add_argument("--group_size", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--grpo_epochs", type=int, default=2, help="Gradient updates per rollout (2 reduces staleness vs original 4)")
    parser.add_argument("--think_budget", type=int, default=2000, help="Max thinking tokens before forcing code output")
    parser.add_argument("--num_turns", type=int, default=4, help="Multi-turn refinement turns per trajectory (Kevin's recipe)")
    parser.add_argument("--gamma", type=float, default=0.4, help="Discount factor for multi-turn returns")
    parser.add_argument("--wandb_project", type=str, default="kernelforge-rl")
    parser.add_argument("--wandb_name", type=str, default="grpo-qwen-14b")
    parser.add_argument("--resume", action="store_true", help="Resume from output_dir if it exists")
    parser.add_argument("--mock_mode", action="store_true")
    parser.add_argument("--no_dynamic_sampling", action="store_true", help="Disable DAPO dynamic sampling")
    parser.add_argument("--use_sglang", action="store_true", help="Use SGLang server for generation (faster)")
    parser.add_argument("--sglang_python", type=str, default="", help="Path to SGLang venv python (e.g. /root/sglang_env/bin/python). Can also set SGLANG_PYTHON env var.")
    parser.add_argument("--sglang_port", type=int, default=30000)
    parser.add_argument("--sglang_tp", type=int, default=1, help="SGLang tensor parallel degree")
    parser.add_argument("--max_prompts", type=int, default=0, help="Cap dataset to N prompts (0=all). Use ~100 for a 2-3h run.")
    parser.add_argument("--no_curriculum", action="store_true", help="Disable curriculum (random prompt order instead of easy-first)")
    parser.add_argument("--llm_feedback_model", type=str, default="",
                        help="Path to GGUF model for LLM-based feedback (replaces BM25 RAG). "
                             "Runs on CPU, no GPU conflict. Empty = use RAG only.")
    args = parser.parse_args()

    cfg = GRPOConfig(
        model_id=args.model,
        adapter_path=args.adapter,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        group_size=args.group_size,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        grpo_epochs=args.grpo_epochs,
        mock_mode=args.mock_mode,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_name,
        dynamic_sampling=not args.no_dynamic_sampling,
        think_budget=args.think_budget,
        num_turns=args.num_turns,
        gamma=args.gamma,
        use_sglang=args.use_sglang,
        sglang_python=args.sglang_python,
        sglang_port=args.sglang_port,
        sglang_tp=args.sglang_tp,
        max_prompts=args.max_prompts,
        curriculum=not args.no_curriculum,
        llm_feedback_model=args.llm_feedback_model,
    )

    # Initialize LLM feedback if model path is provided
    if cfg.llm_feedback_model:
        _llm_feedback = LLMFeedback(cfg.llm_feedback_model)
    
    # Simple resume logic: swap base model for output_dir if resuming
    if args.resume and os.path.exists(args.output_dir):
        # find the highest step checkpoint, or output_dir itself
        import glob
        ckpts = glob.glob(f"{args.output_dir}/step_*")
        if ckpts:
            latest = sorted(ckpts, key=lambda x: int(x.split("_")[-1]))[-1]
            print(f"Resuming from checkpoint: {latest}")
            cfg.model_id = latest
            cfg.adapter_path = "" # weights are already merged in checkpoint
        else:
            print(f"Resuming from output dir: {args.output_dir}")
            cfg.model_id = args.output_dir
            cfg.adapter_path = ""

    train(cfg)
