"""
generate_sft_data.py - Download and prepare CUDA kernel SFT training pairs.

Sources:
1. SakanaAI/AI-CUDA-Engineer-Archive - verified CUDA kernel pairs (Correct=True)
2. KernelBench (ScalingIntelligence/KernelBench) - 250 PyTorch reference problems
   → Saved separately to rl_prompts.jsonl (prompt-only, for GRPO stage)

Verification gates applied to every SFT example before inclusion:
  Gate 1: SakanaAI Correct=True flag (fast, no compilation)
  Gate 2: GRPO pipeline eval — same build_load_inline_wrapper + evaluate
          used during RL training. Runs in parallel across --workers processes.
          If it passes here, it passes in GRPO.

Format matches GRPO rollout exactly — zero distribution shift at RL stage.
"""
import argparse
import json
import os
import sys
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from datasets import load_dataset
from tqdm import tqdm

# ── rl/ imports ───────────────────────────────────────────────────────────
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
    Training text — must match GRPO rollout prompt exactly:
      FORMAT_EXAMPLE + "Reference Program:\\n```python\\n{code}\\n```"
    """
    user_msg = FORMAT_EXAMPLE + f"Reference Program:\n```python\n{pytorch_code}\n```"
    return (
        SYSTEM
        + f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        + f"<|im_start|>assistant\n```cpp\n{cuda_code}\n```<|im_end|>\n"
    )


# ─── Gate 2 worker — exact GRPO eval pipeline ─────────────────────────────
# Identical to _worker_eval_pair in rl/train_grpo.py.

def _grpo_eval_worker(pair):
    """(cuda_code, pytorch_code) → (ok: bool, err: str)"""
    import os, sys
    if _rl_dir not in sys.path:
        sys.path.insert(0, _rl_dir)
    # Suppress [WRAPPER DEBUG] prints
    devnull = open(os.devnull, 'w')
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = devnull
    try:
        from agent import build_load_inline_wrapper
        from sandbox import evaluate
        cuda_code, pytorch_code = pair
        wrapper = build_load_inline_wrapper(cuda_code, pytorch_code)
        sys.stdout, sys.stderr = old_stdout, old_stderr
        devnull.close()
        if not wrapper:
            return False, "no binding"
        result = evaluate(wrapper, pytorch_code)
        if result and result.get("correct", False):
            return True, ""
        err = (result or {}).get("compiler_error") or "wrong output"
        return False, err[:200]
    except Exception as e:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        try: devnull.close()
        except: pass
        return False, str(e)[:200]


# ─── main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_level", type=int, default=1667,
                        help="Max candidates per SakanaAI level to consider (~5000 total)")
    parser.add_argument("--no_pipeline_check", action="store_true",
                        help="Skip Gate 2. Faster but includes unverified kernels.")
    parser.add_argument("--workers", type=int, default=16,
                        help="Parallel workers for Gate 2 evaluation (default 16)")
    parser.add_argument("--output", default="./sft_training_pairs.jsonl")
    parser.add_argument("--rl_prompts_output", default="./rl_prompts.jsonl")
    args = parser.parse_args()

    run_pipeline_check = not args.no_pipeline_check

    print("=" * 60)
    print("SFT data generation")
    print("  Gate 1: SakanaAI Correct=True")
    print(f"  Gate 2: GRPO pipeline eval ({args.workers} parallel workers)"
          if run_pipeline_check else "  Gate 2: SKIPPED (--no_pipeline_check)")
    print("=" * 60)

    # ── Step 1: Collect all Gate 1 candidates (fast — no compilation) ────
    print("\nStep 1/2: Collecting Gate 1 candidates from SakanaAI...")
    candidates = []   # list of (cuda_code, pytorch_code, meta)
    g1_fail = 0

    for level in ["level_1", "level_2", "level_3"]:
        level_count = 0
        try:
            ds = load_dataset("SakanaAI/AI-CUDA-Engineer-Archive", split=level)
        except Exception as e:
            print(f"  Warning: could not load {level}: {e}")
            continue

        for row in ds:
            if level_count >= args.per_level:
                break
            if not row.get("Correct", False):
                g1_fail += 1
                continue

            pytorch_code = (
                row.get("PyTorch_Code_Module", "")
                or row.get("PyTorch_Code_Functional", "")
            ).strip()
            cuda_code = row.get("CUDA_Code", "").strip()

            if not pytorch_code or not cuda_code:
                continue
            if "__global__" not in cuda_code or "torch::Tensor" not in cuda_code:
                continue

            candidates.append((cuda_code, pytorch_code, {
                "source": f"sakana-{level}",
                "task_id": str(row.get("task_id", "")),
                "level_id": level,
            }))
            level_count += 1

        print(f"  {level}: {level_count} Gate 1 candidates")

    print(f"\nGate 1: {len(candidates)} pass, {g1_fail} fail (Correct=False)")

    # ── Step 2: Batch evaluate in parallel ───────────────────────────────
    pairs = []
    g2_pass = g2_fail = 0

    if run_pipeline_check:
        print(f"\nStep 2/2: Gate 2 — GRPO pipeline eval on {len(candidates)} candidates "
              f"({args.workers} workers)...")

        t0 = time.time()
        pairs_input = [(c[0], c[1]) for c in candidates]

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(tqdm(
                pool.map(_grpo_eval_worker, pairs_input, chunksize=1),
                total=len(pairs_input),
                unit="kernel",
                desc="Gate 2",
            ))

        for (ok, err), (cuda_code, pytorch_code, meta) in zip(results, candidates):
            if ok:
                g2_pass += 1
                pairs.append({
                    **meta,
                    "pytorch_code": pytorch_code,
                    "cuda_kernel": cuda_code,
                    "text": make_training_text(pytorch_code, cuda_code),
                })
            else:
                g2_fail += 1

        elapsed = time.time() - t0
        print(f"Gate 2: {g2_pass} pass, {g2_fail} fail "
              f"(rejection rate {g2_fail / max(1, len(candidates)) * 100:.1f}%) "
              f"in {elapsed:.0f}s")
    else:
        print("\nStep 2/2: Skipping Gate 2 — using all Gate 1 candidates.")
        for cuda_code, pytorch_code, meta in candidates:
            pairs.append({
                **meta,
                "pytorch_code": pytorch_code,
                "cuda_kernel": cuda_code,
                "text": make_training_text(pytorch_code, cuda_code),
            })
        g2_pass = len(pairs)

    # ── Deduplicate and save ──────────────────────────────────────────────
    seen = set()
    unique_pairs = []
    for p in pairs:
        key = hash(p["pytorch_code"] + p["cuda_kernel"])
        if key not in seen:
            seen.add(key)
            unique_pairs.append(p)

    if not unique_pairs:
        print("\nERROR: No pairs survived verification.")
        return

    with open(args.output, "w") as f:
        for p in unique_pairs:
            f.write(json.dumps(p) + "\n")
    print(f"SFT training pairs: {len(unique_pairs)} → {args.output}")

    # ── KernelBench RL prompts ────────────────────────────────────────────
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

    print(f"\n=== Done ===")
    print(f"SFT pairs: {len(unique_pairs)}")
    print(f"RL prompts: {len(rl_prompts)}")


if __name__ == "__main__":
    main()
