"""
generate_sft_data.py - Download and prepare CUDA kernel SFT training pairs.

Sources:
1. SakanaAI/AI-CUDA-Engineer-Archive - verified CUDA kernel pairs (Correct=True)
2. KernelBench (ScalingIntelligence/KernelBench) - 250 PyTorch reference problems
   → Saved separately to rl_prompts.jsonl (prompt-only, for GRPO stage)

Verification gates applied to every SFT example before inclusion:
  Gate 1: SakanaAI Correct=True flag
  Gate 2: nvcc -c compilation (catches API mismatches, syntax errors)
  Gate 3: Functional correctness — output matches PyTorch reference (optional, --no_functional_check to skip)

Format matches GRPO rollout exactly so there is zero distribution shift at RL stage.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from datasets import load_dataset

# ── import rl/sandbox for functional correctness check ────────────────────
_rl_dir = os.path.join(os.path.dirname(__file__), "..", "rl")
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
    Create full training prompt+target.
    Format must match GRPO rollout format in rl/train_grpo.py exactly:
      FORMAT_EXAMPLE + "Reference Program:\\n```python\\n{code}\\n```"
    """
    user_msg = FORMAT_EXAMPLE + f"Reference Program:\n```python\n{pytorch_code}\n```"
    return (
        SYSTEM
        + f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        + f"<|im_start|>assistant\n```cpp\n{cuda_code}\n```<|im_end|>\n"
    )


# ─── Gate 2: nvcc compilation ─────────────────────────────────────────────

def nvcc_compiles(cuda_code: str) -> tuple[bool, str]:
    """
    Gate 2: Compile CUDA C++ with nvcc -c.
    Returns (True, "") on success, (False, error_msg) on failure.
    """
    try:
        from torch.utils.cpp_extension import include_paths
        import sysconfig
        includes = []
        for p in include_paths():
            includes.extend(["-I", p])
        includes.extend(["-I", sysconfig.get_path("include")])
    except Exception:
        includes = []

    tmpdir = tempfile.mkdtemp(prefix="kf_sft_nvcc_")
    cu_path = os.path.join(tmpdir, "kernel.cu")
    obj_path = os.path.join(tmpdir, "kernel.o")
    try:
        with open(cu_path, "w") as f:
            f.write(cuda_code)
        cmd = (
            ["nvcc", "-c", cu_path, "-o", obj_path,
             "--std=c++17", "-w",
             "--expt-relaxed-constexpr", "--expt-extended-lambda"]
            + includes
        )
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            return True, ""
        return False, (result.stderr + result.stdout).strip()
    except subprocess.TimeoutExpired:
        return False, "nvcc timeout"
    except Exception as e:
        return False, str(e)
    finally:
        for f in [cu_path, obj_path]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass
        try:
            os.rmdir(tmpdir)
        except Exception:
            pass


# ─── Gate 3: functional correctness ───────────────────────────────────────

def _functional_check_worker(args):
    """Run in subprocess via ProcessPoolExecutor to isolate CUDA context."""
    cuda_code, pytorch_code = args
    try:
        from agent import build_load_inline_wrapper
        from sandbox import evaluate
        wrapper = build_load_inline_wrapper(cuda_code, pytorch_code)
        if not wrapper:
            return False, "no_binding"
        result = evaluate(wrapper, pytorch_code)
        if result is None:
            return False, "compile_fail"
        if result.get("correct", False):
            return True, ""
        return False, result.get("compiler_error", "wrong_output")[:200]
    except Exception as e:
        return False, str(e)[:200]


def functional_correct(cuda_code: str, pytorch_code: str, timeout: int = 60) -> tuple[bool, str]:
    """
    Gate 3: Run the CUDA kernel and verify output matches PyTorch reference.
    Uses ProcessPoolExecutor to isolate CUDA context per check.
    """
    with ProcessPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_functional_check_worker, (cuda_code, pytorch_code))
        try:
            return future.result(timeout=timeout)
        except Exception as e:
            return False, f"timeout/error: {e}"


# ─── main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_level", type=int, default=1667,
                        help="Max examples per SakanaAI level (default 1667 → ~5000 total)")
    parser.add_argument("--no_functional_check", action="store_true",
                        help="Skip Gate 3 (functional correctness). Faster but lower quality.")
    parser.add_argument("--output", default="./sft_training_pairs.jsonl")
    parser.add_argument("--rl_prompts_output", default="./rl_prompts.jsonl")
    args = parser.parse_args()

    run_functional = not args.no_functional_check

    # Check nvcc available
    try:
        subprocess.run(["nvcc", "--version"], stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, check=True)
        nvcc_available = True
        print("nvcc: available ✓")
    except Exception:
        nvcc_available = False
        print("WARNING: nvcc not found — Gate 2 will be skipped. Run on GPU machine.")

    if run_functional:
        print("Gate 3 (functional correctness): ENABLED")
    else:
        print("Gate 3 (functional correctness): DISABLED (--no_functional_check)")

    pairs = []
    gate1_pass = gate2_pass = gate3_pass = 0
    gate1_fail = gate2_fail = gate3_fail = 0

    # === Source: SakanaAI/AI-CUDA-Engineer-Archive ========================
    print("\nLoading SakanaAI/AI-CUDA-Engineer-Archive...")
    for level in ["level_1", "level_2", "level_3"]:
        level_pairs = []
        try:
            ds = load_dataset("SakanaAI/AI-CUDA-Engineer-Archive", split=level)
        except Exception as e:
            print(f"  Warning: could not load {level}: {e}")
            continue

        for row in ds:
            if len(level_pairs) >= args.per_level:
                break

            # Gate 1: SakanaAI Correct flag
            if not row.get("Correct", False):
                gate1_fail += 1
                continue
            gate1_pass += 1

            pytorch_code = (
                row.get("PyTorch_Code_Module", "")
                or row.get("PyTorch_Code_Functional", "")
            ).strip()
            cuda_code = row.get("CUDA_Code", "").strip()

            if not pytorch_code or not cuda_code:
                continue
            if "__global__" not in cuda_code or "torch::Tensor" not in cuda_code:
                continue

            # Gate 2: nvcc compilation
            if nvcc_available:
                ok, err = nvcc_compiles(cuda_code)
                if not ok:
                    gate2_fail += 1
                    continue
                gate2_pass += 1
            else:
                gate2_pass += 1  # skip gate if no GPU machine

            # Gate 3: functional correctness
            if run_functional:
                ok, err = functional_correct(cuda_code, pytorch_code)
                if not ok:
                    gate3_fail += 1
                    continue
                gate3_pass += 1
            else:
                gate3_pass += 1

            text = make_training_text(pytorch_code, cuda_code)
            level_pairs.append({
                "source": f"sakana-{level}",
                "task_id": str(row.get("task_id", "")),
                "level_id": level,
                "pytorch_code": pytorch_code,
                "cuda_kernel": cuda_code,
                "text": text,
            })

        pairs.extend(level_pairs)
        print(f"  {level}: {len(level_pairs)} verified pairs")

    print(f"\nVerification summary:")
    print(f"  Gate 1 (Correct=True): {gate1_pass} pass, {gate1_fail} fail")
    print(f"  Gate 2 (nvcc compile): {gate2_pass} pass, {gate2_fail} fail")
    if run_functional:
        print(f"  Gate 3 (functional):   {gate3_pass} pass, {gate3_fail} fail")

    # Deduplicate
    seen = set()
    unique_pairs = []
    for p in pairs:
        key = hash(p["pytorch_code"] + p["cuda_kernel"])
        if key not in seen:
            seen.add(key)
            unique_pairs.append(p)

    if len(unique_pairs) == 0:
        print("\nERROR: No training pairs survived verification. Check data sources.")
        return

    with open(args.output, "w") as f:
        for p in unique_pairs:
            f.write(json.dumps(p) + "\n")
    print(f"\nSFT training pairs: {len(unique_pairs)} → {args.output}")

    # === RL prompts: KernelBench (prompt-only, no solutions) ==============
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
