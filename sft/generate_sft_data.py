"""
generate_sft_data.py

1. Sample N kernels from SakanaAI (Correct=True)
2. Claude API generates a complete model_new.py for each
3. Sandbox compiles + verifies each one
4. Save passing pairs as SFT training data

Training format: pytorch_ref → complete model_new.py
GRPO evaluates by exec()-ing model output directly. No wrapper inference.
"""
import argparse
import json
import os
import random
import sys
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    user_msg = FORMAT_EXAMPLE + f"Reference Program:\n```python\n{pytorch_code}\n```"
    return (
        SYSTEM
        + f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        + f"<|im_start|>assistant\n```python\n{model_new_py}\n```<|im_end|>\n"
    )


# ── Claude API: batch generate model_new.py with caching ──────────────────

CLAUDE_SYSTEM = """\
You are an expert NVIDIA CUDA engineer.
Given a PyTorch reference implementation and its verified CUDA C++ kernel,
generate a complete, working model_new.py file.

Output EXACTLY ONE ```python code block with the complete model_new.py.
The file must:
- Embed the CUDA source as a string
- Use load_inline to compile it
- Define ModelNew(nn.Module) with correct forward() that calls the kernel
- Match the reference PyTorch model's interface exactly (same __init__ args, same forward args)

CRITICAL: If the CUDA source contains a PYBIND11_MODULE block, you MUST remove it entirely
from cuda_source. load_inline generates its own bindings via functions=[]. Having both causes
a duplicate symbol linker error and the kernel will fail to compile.
"""

CLAUDE_PROMPT = """\
Reference PyTorch implementation:
```python
{pytorch_code}
```

Verified CUDA C++ kernel (Correct=True, already tested by SakanaAI):
```cpp
{cuda_code}
```

Generate the complete model_new.py:
"""

CACHE_FILE = os.path.join(os.path.dirname(__file__), ".claude_cache.json")


def _cache_key(cuda_code: str, pytorch_code: str) -> str:
    import hashlib
    return hashlib.md5((cuda_code + pytorch_code).encode()).hexdigest()


def _load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)


def _extract_python_block(text: str) -> str:
    import re
    m = re.search(r'```python\s*(.*?)```', text, re.DOTALL)
    return m.group(1).strip() if m else ""


def _generate_batch(sample: list, api_key: str) -> list:
    """
    Submit all requests as a single Claude Batch API call.
    Checks cache first — only sends uncached items.
    Returns list of (model_new_py | None, pytorch_code, meta) in same order as sample.
    """
    import anthropic
    import time

    client = anthropic.Anthropic(api_key=api_key)
    cache  = _load_cache()

    # Split into cached and uncached
    results   = [None] * len(sample)
    to_submit = []   # (original_index, cuda_code, pytorch_code, meta)

    for i, (cuda_code, pytorch_code, meta) in enumerate(sample):
        key = _cache_key(cuda_code, pytorch_code)
        if key in cache:
            results[i] = (cache[key], pytorch_code, meta)
        else:
            to_submit.append((i, cuda_code, pytorch_code, meta))

    print(f"  Cache hits: {len(sample) - len(to_submit)}/{len(sample)}")

    if not to_submit:
        return results

    print(f"  Submitting {len(to_submit)} requests to Claude Batch API...")

    # Build batch requests
    requests = []
    for idx, (i, cuda_code, pytorch_code, meta) in enumerate(to_submit):
        requests.append({
            "custom_id": str(idx),
            "params": {
                "model": "claude-opus-4-6",
                "max_tokens": 4096,
                "system": CLAUDE_SYSTEM,
                "messages": [{"role": "user", "content": CLAUDE_PROMPT.format(
                    pytorch_code=pytorch_code,
                    cuda_code=cuda_code,
                )}],
            },
        })

    # Submit batch
    batch = client.beta.messages.batches.create(requests=requests)
    print(f"  Batch ID: {batch.id} — polling...")

    # Poll until complete
    while batch.processing_status != "ended":
        time.sleep(10)
        batch = client.beta.messages.batches.retrieve(batch.id)
        counts = batch.request_counts
        print(f"    processing={counts.processing} succeeded={counts.succeeded} "
              f"errored={counts.errored}", end="\r")
    print()

    # Collect results — track succeeded vs api_error separately
    # so we never cache API-level failures (billing outage would poison the cache)
    batch_results = {}   # idx -> model_new_py | None  (succeeded, possibly empty extraction)
    api_errors    = set()  # idx -> errored at API level, do NOT cache
    first_error   = None
    for result in client.beta.messages.batches.results(batch.id):
        idx = int(result.custom_id)
        if result.result.type == "succeeded":
            text = result.result.message.content[0].text
            batch_results[idx] = _extract_python_block(text) or None
        else:
            if first_error is None:
                first_error = result.result
                print(f"\n  [ERROR sample] type={result.result.type} error={result.result}")
            api_errors.add(idx)
            batch_results[idx] = None

    # Merge into results + update cache (skip caching API errors)
    for idx, (i, cuda_code, pytorch_code, meta) in enumerate(to_submit):
        model_new_py = batch_results.get(idx)
        results[i]   = (model_new_py, pytorch_code, meta)
        if idx not in api_errors:
            key = _cache_key(cuda_code, pytorch_code)
            cache[key] = model_new_py

    _save_cache(cache)
    return results


# ── Sandbox verification worker ────────────────────────────────────────────

def _verify_worker(item):
    import os, sys, io
    if _rl_dir not in sys.path:
        sys.path.insert(0, _rl_dir)

    model_new_py, pytorch_code, meta = item
    if not model_new_py:
        return False, None, pytorch_code, meta

    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = io.StringIO()
    try:
        from sandbox import evaluate
        sys.stdout, sys.stderr = old_out, old_err
        result = evaluate(model_new_py, pytorch_code)
        ok = bool(result and result.get("correct", False))
        return ok, model_new_py, pytorch_code, meta
    except Exception:
        sys.stdout, sys.stderr = old_out, old_err
        return False, None, pytorch_code, meta


# ── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500,
                        help="Number of SakanaAI kernels to sample (default 500)")
    parser.add_argument("--workers", type=int, default=32,
                        help="Parallel workers for sandbox verification")
    parser.add_argument("--claude_api_key", default=os.environ.get("ANTHROPIC_API_KEY", ""),
                        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    parser.add_argument("--output", default="./sft_training_pairs.jsonl")
    parser.add_argument("--rl_prompts_output", default="./rl_prompts.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.claude_api_key:
        print("ERROR: set ANTHROPIC_API_KEY or pass --claude_api_key")
        sys.exit(1)

    random.seed(args.seed)

    print("=" * 60)
    print(f"SFT data generation")
    print(f"  Sample : {args.n} random SakanaAI kernels (Correct=True)")
    print(f"  Step 1 : Claude generates model_new.py for each")
    print(f"  Step 2 : Sandbox compiles + verifies")
    print(f"  Format : pytorch_ref → complete model_new.py")
    print("=" * 60)

    # ── Collect all Correct=True candidates ──────────────────────────────
    print("\nLoading SakanaAI dataset...")
    all_candidates = []
    for level in ["level_1", "level_2", "level_3"]:
        try:
            ds = load_dataset("SakanaAI/AI-CUDA-Engineer-Archive", split=level)
        except Exception as e:
            print(f"  Warning: could not load {level}: {e}")
            continue
        for row in ds:
            if not row.get("Correct", False):
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
            all_candidates.append((cuda_code, pytorch_code, {
                "source": f"sakana-{level}",
                "task_id": str(row.get("task_id", "")),
                "level_id": level,
            }))
        print(f"  {level}: loaded")

    print(f"Total Correct=True candidates: {len(all_candidates)}")

    # ── Random sample ─────────────────────────────────────────────────────
    sample = random.sample(all_candidates, min(args.n, len(all_candidates)))
    print(f"Sampled: {len(sample)}")

    # ── Step 1: Claude Batch API generates model_new.py ──────────────────
    print(f"\nStep 1: Generating model_new.py via Claude Batch API ({len(sample)} kernels)...")
    t0 = time.time()
    generated = _generate_batch(sample, args.claude_api_key)
    n_generated = sum(1 for m, _, _ in generated if m)
    print(f"Claude generated: {n_generated}/{len(sample)} in {time.time()-t0:.0f}s")

    # ── Step 2: Sandbox verification ─────────────────────────────────────
    # Filter out None entries before the pool to avoid wasting worker slots
    to_verify = [(m, p, meta) for m, p, meta in generated if m]
    skipped = len(generated) - len(to_verify)
    print(f"\nStep 2: Compiling + verifying {len(to_verify)} generated wrappers "
          f"({args.workers} workers, {skipped} skipped as None)...")
    t0 = time.time()

    verify_results = []
    n_pass = n_fail = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_verify_worker, item): item for item in to_verify}
        with tqdm(total=len(to_verify), unit="kernel", desc="Verify") as bar:
            for fut in as_completed(futures):
                ok, m, p, meta = fut.result()
                verify_results.append((ok, m, p, meta))
                if ok:
                    n_pass += 1
                else:
                    n_fail += 1
                bar.set_postfix(passed=n_pass, failed=n_fail, rate=f"{n_pass}/{n_pass+n_fail}")
                bar.update(1)

    passed = [(m, p, meta) for ok, m, p, meta in verify_results if ok]
    print(f"Verified: {len(passed)}/{len(to_verify)} pass ({n_fail} failed) in {time.time()-t0:.0f}s")

    if not passed:
        print("ERROR: No pairs passed verification.")
        return

    # ── Save training pairs ───────────────────────────────────────────────
    training_pairs = []
    seen = set()
    for model_new_py, pytorch_code, meta in passed:
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

    with open(args.output, "w") as f:
        for p in training_pairs:
            f.write(json.dumps(p) + "\n")
    print(f"\nSFT training pairs: {len(training_pairs)} → {args.output}")

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
