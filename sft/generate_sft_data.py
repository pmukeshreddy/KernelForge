"""
generate_sft_data.py - Download and prepare CUDA kernel SFT training pairs.

Sources:
1. SakanaAI/AI-CUDA-Engineer-Archive - verified CUDA kernel pairs (Correct=True)
2. KernelBench (ScalingIntelligence/KernelBench) - 250 PyTorch reference problems
   → Saved separately to rl_prompts.jsonl (prompt-only, for GRPO stage)

Verification gates applied to every SFT example before inclusion:
  Gate 1: SakanaAI Correct=True flag
  Gate 2: GRPO pipeline eval — same _worker_eval_pair used during RL training:
            build_load_inline_wrapper(cuda_code, pytorch_code)
            → evaluate(wrapper, pytorch_code)
          If it passes here, it will pass in GRPO. No excuses.

Format matches GRPO rollout exactly — zero distribution shift at RL stage.
"""
import argparse
import json
import os
import sys
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from datasets import load_dataset

# ── rl/ imports — must come before any rl module usage ───────────────────
_rl_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rl"))
if _rl_dir not in sys.path:
    sys.path.insert(0, _rl_dir)


def _normalize_code(code: str) -> str:
    return unicodedata.normalize("NFKC", code).encode("utf-8").decode("utf-8")


# ─── System Prompt — must match rl/sys_prompt.py exactly ──────────────────
SYSTEM = """<|im_start|>system
You are an expert NVIDIA CUDA Systems Engineer.
Your objective is to write optimized CUDA C++ kernels to replace PyTorch operations.

# Constraints
- Write valid CUDA C++ with `#include <torch/extension.h>` and `#include <cuda_runtime.h>`.
- Write a `__global__ void` kernel with proper thread indexing.
- Write a C++ binding function returning `torch::Tensor` using PyTorch C++ API.
- Input tensors are `float32` by default. Use `float*` pointers and `data_ptr<float>()`.
- Do NOT use cuBLAS, cuDNN, or CUTLASS.

# Output Format
Output EXACTLY ONE ```cpp code block containing your kernel and binding function.

# Common Bugs to Avoid
- Use `fmaxf`/`fminf` in device code, NOT `std::max`/`std::min`.
- Max 1024 threads per block. For 2D blocks: blockDim.x * blockDim.y <= 1024.
- Declare `__shared__` arrays INSIDE the kernel function body.
- Use `torch::empty_like(input)` to preserve tensor shape and dtype.
<|im_end|>
"""

# ─── Format example — must match rl/train_grpo.py FORMAT_EXAMPLE exactly ─
FORMAT_EXAMPLE = """\
Here is an example of the expected output format:

```cpp
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void add_kernel(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] + b[idx];
}

torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b) {
    auto out = torch::empty_like(a);
    int n = a.numel();
    add_kernel<<<(n + 255) / 256, 256>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}
```

Now write the kernel for the following operation:
"""


def make_training_text(pytorch_code: str, cuda_code: str) -> str:
    """
    Training text format — must match GRPO rollout prompt in rl/train_grpo.py exactly:
      user_msg = FORMAT_EXAMPLE + "Reference Program:\\n```python\\n{code}\\n```"
    """
    user_msg = FORMAT_EXAMPLE + f"Reference Program:\n```python\n{pytorch_code}\n```"
    return (
        SYSTEM
        + f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        + f"<|im_start|>assistant\n```cpp\n{cuda_code}\n```<|im_end|>\n"
    )


# ─── Gate 2: GRPO pipeline eval (exact same code path as RL training) ─────

def _grpo_eval_worker(args):
    """
    Runs in a subprocess via ProcessPoolExecutor to isolate CUDA context.

    This is EXACTLY the same logic as _worker_eval_pair in rl/train_grpo.py:
        candidate = build_load_inline_wrapper(cuda_code, pytorch_code)
        return evaluate(candidate, pytorch_code)

    If a kernel passes here, it will pass during GRPO training. Period.
    """
    import sys, os
    rl_dir = args[2]
    if rl_dir not in sys.path:
        sys.path.insert(0, rl_dir)

    cuda_code, pytorch_code, _ = args
    try:
        from agent import build_load_inline_wrapper
        from sandbox import evaluate
        wrapper = build_load_inline_wrapper(cuda_code, pytorch_code)
        if not wrapper:
            return False, "no torch::Tensor binding found"
        result = evaluate(wrapper, pytorch_code)
        if result is None:
            return False, "evaluate returned None"
        if result.get("correct", False):
            return True, ""
        err = result.get("compiler_error") or "wrong output"
        return False, err[:300]
    except Exception as e:
        return False, str(e)[:300]


def grpo_pipeline_passes(cuda_code: str, pytorch_code: str, timeout: int = 120) -> tuple[bool, str]:
    """Gate 2: run the exact GRPO eval pipeline on this (cuda_code, pytorch_code) pair."""
    with ProcessPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_grpo_eval_worker, (cuda_code, pytorch_code, _rl_dir))
        try:
            return future.result(timeout=timeout)
        except Exception as e:
            return False, f"timeout/crash: {e}"


# ─── main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_level", type=int, default=1667,
                        help="Max examples per SakanaAI level (~5000 total)")
    parser.add_argument("--no_pipeline_check", action="store_true",
                        help="Skip Gate 2 (GRPO pipeline check). Faster but lower quality.")
    parser.add_argument("--output", default="./sft_training_pairs.jsonl")
    parser.add_argument("--rl_prompts_output", default="./rl_prompts.jsonl")
    args = parser.parse_args()

    run_pipeline_check = not args.no_pipeline_check

    print("=" * 60)
    print("SFT data generation — verification gates:")
    print("  Gate 1: SakanaAI Correct=True flag")
    if run_pipeline_check:
        print("  Gate 2: GRPO pipeline (build_load_inline_wrapper + evaluate)")
        print("          Same code as _worker_eval_pair in train_grpo.py")
    else:
        print("  Gate 2: SKIPPED (--no_pipeline_check)")
    print("=" * 60)

    pairs = []
    g1_pass = g1_fail = g2_pass = g2_fail = 0

    # === SakanaAI/AI-CUDA-Engineer-Archive ===================================
    print("\nLoading SakanaAI/AI-CUDA-Engineer-Archive...")
    for level in ["level_1", "level_2", "level_3"]:
        level_pairs = []
        try:
            ds = load_dataset("SakanaAI/AI-CUDA-Engineer-Archive", split=level)
        except Exception as e:
            print(f"  Warning: could not load {level}: {e}")
            continue

        checked = 0
        for row in ds:
            if len(level_pairs) >= args.per_level:
                break

            # Gate 1
            if not row.get("Correct", False):
                g1_fail += 1
                continue
            g1_pass += 1

            pytorch_code = (
                row.get("PyTorch_Code_Module", "")
                or row.get("PyTorch_Code_Functional", "")
            ).strip()
            cuda_code = row.get("CUDA_Code", "").strip()

            if not pytorch_code or not cuda_code:
                continue
            if "__global__" not in cuda_code or "torch::Tensor" not in cuda_code:
                continue

            # Gate 2: GRPO pipeline
            if run_pipeline_check:
                checked += 1
                ok, err = grpo_pipeline_passes(cuda_code, pytorch_code)
                if not ok:
                    g2_fail += 1
                    continue
                g2_pass += 1
            else:
                g2_pass += 1

            text = make_training_text(pytorch_code, cuda_code)
            level_pairs.append({
                "source": f"sakana-{level}",
                "task_id": str(row.get("task_id", "")),
                "level_id": level,
                "pytorch_code": pytorch_code,
                "cuda_kernel": cuda_code,
                "text": text,
            })

            if len(level_pairs) % 50 == 0:
                print(f"  {level}: {len(level_pairs)} verified so far "
                      f"(gate2: {g2_pass} pass / {g2_fail} fail)...")

        pairs.extend(level_pairs)
        print(f"  {level}: {len(level_pairs)} verified pairs")

    print(f"\nVerification summary:")
    print(f"  Gate 1 (Correct=True):     {g1_pass} pass, {g1_fail} fail")
    print(f"  Gate 2 (GRPO pipeline):    {g2_pass} pass, {g2_fail} fail")
    print(f"  Gate 2 rejection rate:     {g2_fail / max(1, g2_pass + g2_fail) * 100:.1f}%")

    # Deduplicate
    seen = set()
    unique_pairs = []
    for p in pairs:
        key = hash(p["pytorch_code"] + p["cuda_kernel"])
        if key not in seen:
            seen.add(key)
            unique_pairs.append(p)

    if not unique_pairs:
        print("\nERROR: No pairs survived verification. Check data source and GPU environment.")
        return

    with open(args.output, "w") as f:
        for p in unique_pairs:
            f.write(json.dumps(p) + "\n")
    print(f"\nSFT training pairs: {len(unique_pairs)} → {args.output}")

    # === KernelBench RL prompts (prompt-only) ================================
    print("\nLoading KernelBench RL prompts...")
    rl_prompts = []
    existing_codes = {p["pytorch_code"] for p in unique_pairs}
    for level in ["level_1", "level_2", "level_3"]:
        try:
            ds = load_dataset("ScalingIntelligence/KernelBench", split=level)
            for row in ds:
                code = (
                    row.get("code", "")
                    or row.get("pytorch_code", "")
                    or row.get("ref_code", "")
                ).strip()
                if not code or code in existing_codes:
                    continue
                code = _normalize_code(code)
                rl_prompts.append({
                    "source": f"kernelbench-{level}",
                    "task_id": row.get("task_id", row.get("name", "")),
                    "level_id": level,
                    "pytorch_code": code,
                })
                existing_codes.add(code)
        except Exception as e:
            print(f"  Warning: could not load KernelBench {level}: {e}")

    if rl_prompts:
        with open(args.rl_prompts_output, "w") as f:
            for p in rl_prompts:
                f.write(json.dumps(p) + "\n")
        print(f"RL prompts: {len(rl_prompts)} → {args.rl_prompts_output}")

    print("\n=== Done ===")
    print(f"SFT pairs: {len(unique_pairs)}")
    print(f"RL prompts: {len(rl_prompts)}")


if __name__ == "__main__":
    main()
