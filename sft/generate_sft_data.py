"""
generate_sft_data.py - Build verified (pytorch_ref → model_new.py) SFT pairs.

Pipeline:
  Gate 1 : SakanaAI Correct=True  (no compilation)
  Stage 1: build_load_inline_wrapper + sandbox eval  → ~1500 pass
  Stage 2: Claude API repairs failures               → ~2800 more pass
  Output : (pytorch_ref → complete model_new.py) training pairs

Training format: model outputs a complete Python file (model_new.py) with
load_inline + ModelNew class.  GRPO evaluates by exec()-ing the output directly.
Zero wrapper inference in the RL hot path.
"""
import argparse
import json
import os
import sys
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from datasets import load_dataset
from tqdm import tqdm

_rl_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rl"))
if _rl_dir not in sys.path:
    sys.path.insert(0, _rl_dir)


def _normalize_code(code: str) -> str:
    return unicodedata.normalize("NFKC", code).encode("utf-8").decode("utf-8")


# ── System prompt (must match rl/sys_prompt.py) ────────────────────────────
SYSTEM = """\
<|im_start|>system
You are an expert NVIDIA CUDA Systems Engineer.
Your objective is to write an optimized CUDA kernel to replace a PyTorch operation,
delivered as a complete, self-contained Python file.

# Output Format
Output EXACTLY ONE ```python code block containing a complete model_new.py file that:
1. Embeds the CUDA C++ source as a string.
2. Compiles it with `torch.utils.cpp_extension.load_inline`.
3. Defines a `ModelNew(torch.nn.Module)` class whose `forward()` calls the kernel.

# Constraints
- CUDA kernel: `#include <torch/extension.h>`, `#include <cuda_runtime.h>`.
- Binding function must return `torch::Tensor` and include `PYBIND11_MODULE`.
- Input tensors are `float32`. Use `float*` and `.data_ptr<float>()`.
- Do NOT use cuBLAS, cuDNN, or CUTLASS.

# Common Bugs to Avoid
- Use `fmaxf`/`fminf` in device code, NOT `std::max`/`std::min`.
- Max 1024 threads per block.
- Declare `__shared__` arrays INSIDE the kernel body.
<|im_end|>
"""

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &add_cuda, "add_cuda");
}
\"\"\"

ext = load_inline(
    name="add_ext",
    cpp_sources="torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b);",
    cuda_sources=cuda_source,
    functions=["add_cuda"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, a, b):
        return ext.add_cuda(a, b)
```

Now write the complete model_new.py for the following operation:
"""


def make_training_text(pytorch_code: str, model_new_py: str) -> str:
    """Training text: prompt → complete model_new.py Python file."""
    user_msg = FORMAT_EXAMPLE + f"Reference Program:\n```python\n{pytorch_code}\n```"
    return (
        SYSTEM
        + f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        + f"<|im_start|>assistant\n```python\n{model_new_py}\n```<|im_end|>\n"
    )


# ── Stage 1 worker: build wrapper + sandbox eval ───────────────────────────

def _stage1_worker(item):
    """(cuda_code, pytorch_code, meta) → (model_new_py | None, err_str)"""
    import os, sys, io
    if _rl_dir not in sys.path:
        sys.path.insert(0, _rl_dir)

    cuda_code, pytorch_code, meta = item
    buf = io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = buf

    try:
        from agent import build_load_inline_wrapper
        from sandbox import evaluate

        wrapper = build_load_inline_wrapper(cuda_code, pytorch_code)
        sys.stdout, sys.stderr = old_out, old_err

        if not wrapper:
            return None, "no_binding"

        result = evaluate(wrapper, pytorch_code)
        if result and result.get("correct", False):
            return wrapper, ""

        err = (result or {}).get("compiler_error") or "wrong output"
        return None, err[:300]

    except Exception as e:
        sys.stdout, sys.stderr = old_out, old_err
        return None, str(e)[:300]


# ── Stage 2: Claude API repair ─────────────────────────────────────────────

REPAIR_SYSTEM = """\
You are an expert NVIDIA CUDA engineer. Given a PyTorch reference implementation
and its CUDA C++ kernel, generate a complete, working model_new.py file.

Output EXACTLY ONE ```python code block with the complete model_new.py.
The file must compile with load_inline and pass correctness checks against the
reference PyTorch implementation.
"""

REPAIR_PROMPT = """\
Reference PyTorch implementation:
```python
{pytorch_code}
```

CUDA C++ kernel (raw, needs wrapping):
```cpp
{cuda_code}
```

Previous auto-wrapper error:
{error}

Generate a complete, correct model_new.py that compiles and produces correct output.
"""


def _extract_python_block(text: str) -> str:
    """Extract the first ```python ... ``` block from text."""
    import re
    m = re.search(r'```python\s*(.*?)```', text, re.DOTALL)
    return m.group(1).strip() if m else ""


def _repair_with_claude(failures: list, api_key: str, workers: int) -> list:
    """
    Send failures to Claude API.
    failures: list of (cuda_code, pytorch_code, meta, err)
    Returns: list of (model_new_py, pytorch_code, meta) for those Claude fixes.
    """
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    repaired = []
    print(f"\nStage 2: Sending {len(failures)} failures to Claude API...")

    for cuda_code, pytorch_code, meta, err in tqdm(failures, unit="kernel", desc="Claude repair"):
        prompt = REPAIR_PROMPT.format(
            pytorch_code=pytorch_code,
            cuda_code=cuda_code,
            error=err,
        )
        try:
            msg = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=4096,
                system=REPAIR_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            model_new_py = _extract_python_block(msg.content[0].text)
            if model_new_py:
                repaired.append((model_new_py, pytorch_code, meta))
        except Exception as e:
            print(f"  Claude API error for {meta.get('task_id','?')}: {e}")
            continue

    return repaired


def _verify_worker(item):
    """(model_new_py, pytorch_code, meta) → (ok, model_new_py, pytorch_code, meta)"""
    import os, sys, io
    if _rl_dir not in sys.path:
        sys.path.insert(0, _rl_dir)

    model_new_py, pytorch_code, meta = item
    buf = io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = buf

    try:
        from sandbox import evaluate
        sys.stdout, sys.stderr = old_out, old_err
        result = evaluate(model_new_py, pytorch_code)
        ok = bool(result and result.get("correct", False))
        return ok, model_new_py, pytorch_code, meta
    except Exception as e:
        sys.stdout, sys.stderr = old_out, old_err
        return False, model_new_py, pytorch_code, meta


# ── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_pairs", type=int, default=1000,
                        help="Stop once this many verified pairs are collected (default 1000)")
    parser.add_argument("--per_level", type=int, default=1000,
                        help="Max SakanaAI candidates per level (default 1000, ~3000 total)")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--claude_api_key", default=os.environ.get("ANTHROPIC_API_KEY", ""),
                        help="Anthropic API key for Stage 2 repair (or set ANTHROPIC_API_KEY)")
    parser.add_argument("--skip_claude", action="store_true",
                        help="Skip Stage 2 Claude repair — use Stage 1 results only")
    parser.add_argument("--output", default="./sft_training_pairs.jsonl")
    parser.add_argument("--rl_prompts_output", default="./rl_prompts.jsonl")
    args = parser.parse_args()

    print("=" * 60)
    print("SFT data generation — hybrid pipeline")
    print("  Gate 1 : SakanaAI Correct=True")
    print("  Stage 1: build_load_inline_wrapper + sandbox")
    if args.skip_claude:
        print("  Stage 2: SKIPPED (--skip_claude)")
    else:
        print("  Stage 2: Claude API repair of Stage 1 failures")
    print("  Format : pytorch_ref → complete model_new.py")
    print("=" * 60)

    # ── Gate 1: collect SakanaAI Correct=True ────────────────────────────
    print("\nGate 1: collecting candidates...")
    candidates = []
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
        print(f"  {level}: {level_count} candidates")

    print(f"Gate 1: {len(candidates)} pass, {g1_fail} fail")

    # ── Stage 1: auto-wrapper + sandbox eval ─────────────────────────────
    print(f"\nStage 1: wrapping + sandbox eval ({args.workers} workers)...")
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        stage1_results = list(tqdm(
            pool.map(_stage1_worker, candidates, chunksize=1),
            total=len(candidates), unit="kernel", desc="Stage 1",
        ))

    verified_pairs = []   # (model_new_py, pytorch_code, meta)
    failures       = []   # (cuda_code, pytorch_code, meta, err)

    for (wrapper, err), (cuda_code, pytorch_code, meta) in zip(stage1_results, candidates):
        if wrapper:
            verified_pairs.append((wrapper, pytorch_code, meta))
        else:
            failures.append((cuda_code, pytorch_code, meta, err))

    elapsed = time.time() - t0
    print(f"Stage 1: {len(verified_pairs)} pass, {len(failures)} fail in {elapsed:.0f}s")

    # ── Stage 2: Claude API repair (only for as many as we still need) ────
    still_need = args.target_pairs - len(verified_pairs)
    if not args.skip_claude and failures and still_need > 0:
        if not args.claude_api_key:
            print("WARNING: --claude_api_key not set, skipping Stage 2")
        else:
            # Only send as many failures as needed to reach target
            failures_to_send = failures[:still_need * 2]  # 2x buffer for ~50% repair rate
            repaired_raw = _repair_with_claude(failures_to_send, args.claude_api_key, args.workers)

            if repaired_raw:
                print(f"\nVerifying {len(repaired_raw)} Claude-repaired kernels...")
                with ProcessPoolExecutor(max_workers=args.workers) as pool:
                    verify_results = list(tqdm(
                        pool.map(_verify_worker, repaired_raw, chunksize=1),
                        total=len(repaired_raw), unit="kernel", desc="Verify",
                    ))

                s2_pass = 0
                for ok, model_new_py, pytorch_code, meta in verify_results:
                    if ok:
                        verified_pairs.append((model_new_py, pytorch_code, meta))
                        s2_pass += 1
                        if len(verified_pairs) >= args.target_pairs:
                            break

                print(f"Stage 2: {s2_pass}/{len(repaired_raw)} pass verification")
    elif still_need <= 0:
        print(f"Stage 1 already reached target ({args.target_pairs}), skipping Stage 2")

    # Cap at target
    verified_pairs = verified_pairs[:args.target_pairs]
    print(f"\nTotal verified pairs: {len(verified_pairs)} (target: {args.target_pairs})")

    # ── Deduplicate and build training text ───────────────────────────────
    seen = set()
    training_pairs = []
    for model_new_py, pytorch_code, meta in verified_pairs:
        key = hash(pytorch_code + model_new_py)
        if key in seen:
            continue
        seen.add(key)
        training_pairs.append({
            **meta,
            "pytorch_code": pytorch_code,
            "model_new_py": model_new_py,
            "text": make_training_text(pytorch_code, model_new_py),
        })

    if not training_pairs:
        print("ERROR: No pairs survived. Exiting.")
        return

    with open(args.output, "w") as f:
        for p in training_pairs:
            f.write(json.dumps(p) + "\n")
    print(f"SFT training pairs: {len(training_pairs)} → {args.output}")

    # ── KernelBench RL prompts ────────────────────────────────────────────
    print("\nLoading KernelBench RL prompts...")
    rl_prompts = []
    existing_codes = {p["pytorch_code"] for p in training_pairs}

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
    print(f"SFT pairs : {len(training_pairs)}")
    print(f"RL prompts: {len(rl_prompts)}")


if __name__ == "__main__":
    main()
