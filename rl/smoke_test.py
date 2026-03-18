"""
smoke_test.py — Quick model + pipeline sanity check

Pulls 2 problems from each KernelBench level (6 total), generates one
CUDA kernel per problem using the same SGLang path as GRPO, then compiles
and evaluates using the same sandbox. Tells you clearly:
  - Did the model generate parseable code?
  - Did nvcc compile it?
  - Did the output match?

Usage:
    cd /root/KernelForge/rl
    python smoke_test.py --adapter ../sft/sft_qwen3_14b_lora \
                         --sglang_python /root/sglang_env/bin/python
"""

import argparse
import os
import sys
import json
import subprocess
import time
import requests
import signal

# ── make sure rl/ imports resolve ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from datasets import load_dataset
from agent import build_load_inline_wrapper, _extract_cuda_code
from sandbox import evaluate
from sys_prompt import get_system_prompt

PREFILL = "```cpp\n#include <torch/extension.h>\n"
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

PROBLEMS_PER_LEVEL = 2
LEVELS = ["level_1", "level_2", "level_3"]


# ── SGLang helpers ──────────────────────────────────────────────────────────

_server_proc = None

def _launch_sglang(model_path: str, port: int, tp: int, sglang_python: str):
    global _server_proc
    print(f"[SGLang] Launching server on port {port}...")
    python_bin = sglang_python or os.environ.get("SGLANG_PYTHON", sys.executable)
    cmd = [
        python_bin, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--port", str(port),
        "--tp", str(tp),
        "--dtype", "bfloat16",
        "--trust-remote-code",
        "--mem-fraction-static", "0.7",
        "--log-level", "error",
    ]
    _server_proc = subprocess.Popen(cmd)

    url = f"http://localhost:{port}/health"
    for _ in range(120):
        try:
            if requests.get(url, timeout=2).status_code == 200:
                print("[SGLang] Server ready.")
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("SGLang server failed to start.")


def _shutdown_sglang():
    global _server_proc
    if _server_proc:
        _server_proc.terminate()
        _server_proc = None


def _generate(prompt: str, port: int, temperature: float = 0.0) -> str:
    resp = requests.post(
        f"http://localhost:{port}/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": 3000,
                "temperature": temperature,
                "top_p": 1.0,
                "skip_special_tokens": True,
            },
        },
        timeout=120,
    )
    resp.raise_for_status()
    result = resp.json()
    if isinstance(result, list):
        return result[0].get("text", "")
    return result.get("text", "")


# ── prompt builder ──────────────────────────────────────────────────────────

def _build_prompt(pytorch_code: str) -> str:
    user_msg = f"{FORMAT_EXAMPLE}Reference Program:\n```python\n{pytorch_code}\n```"
    return (
        get_system_prompt()
        + f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        + f"<|im_start|>assistant\n{PREFILL}"
    )


# ── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter directory")
    parser.add_argument("--base_model", default="Qwen/Qwen3-14B")
    parser.add_argument("--port", type=int, default=31000)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--sglang_python", default="")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0.0=greedy (deterministic), 0.3=slight diversity")
    args = parser.parse_args()

    # ── 1. Merge LoRA → tmp dir for SGLang ─────────────────────────────────
    merged_path = os.path.join(os.path.dirname(args.adapter), "_smoke_merged")
    if not os.path.exists(merged_path):
        print("Merging LoRA weights for SGLang...")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype="auto", device_map="cpu", trust_remote_code=True)
        model = PeftModel.from_pretrained(base, args.adapter)
        model = model.merge_and_unload()
        model.save_pretrained(merged_path)
        tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        tok.save_pretrained(merged_path)
        print(f"Merged saved to {merged_path}")

    # Fix tokenizer_config if needed (Qwen3 quirk)
    tok_cfg = os.path.join(merged_path, "tokenizer_config.json")
    if os.path.exists(tok_cfg):
        import json as _json
        cfg = _json.load(open(tok_cfg))
        if "extra_special_tokens" in cfg and isinstance(cfg["extra_special_tokens"], list):
            cfg["extra_special_tokens"] = {}
            _json.dump(cfg, open(tok_cfg, "w"), indent=2)

    # ── 2. Launch SGLang ────────────────────────────────────────────────────
    _launch_sglang(merged_path, args.port, args.tp, args.sglang_python)
    signal.signal(signal.SIGINT, lambda *_: (_shutdown_sglang(), sys.exit(0)))

    # ── 3. Load problems ────────────────────────────────────────────────────
    problems = []
    for level in LEVELS:
        try:
            ds = load_dataset("ScalingIntelligence/KernelBench", split=level)
            for row in ds.select(range(min(PROBLEMS_PER_LEVEL, len(ds)))):
                problems.append({"level": level, "code": row["code"]})
            print(f"Loaded {PROBLEMS_PER_LEVEL} problems from {level}")
        except Exception as e:
            print(f"Warning: could not load {level}: {e}")

    if not problems:
        print("No problems loaded. Check HF dataset access.")
        _shutdown_sglang()
        return

    # ── 4. Generate + evaluate ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Running {len(problems)} problems (temp={args.temperature})")
    print(f"{'='*60}\n")

    results = []
    for i, prob in enumerate(problems):
        level = prob["level"]
        pytorch_code = prob["code"]
        print(f"[{i+1}/{len(problems)}] {level}")

        # Generate
        prompt = _build_prompt(pytorch_code)
        try:
            t0 = time.time()
            full_text = _generate(prompt, args.port, args.temperature)
            gen_time = time.time() - t0
            # Restore PREFILL (SGLang returns completion only)
            completion = full_text[len(prompt):] if full_text.startswith(prompt) else full_text
            response = PREFILL + completion
        except Exception as e:
            print(f"  ✗ Generation failed: {e}")
            results.append({"level": level, "status": "gen_error"})
            continue

        # Extract CUDA code
        cuda_code = _extract_cuda_code(response)
        if not cuda_code:
            print(f"  ✗ No ```cpp block found (gen={gen_time:.1f}s)")
            print(f"    Response preview: {response[:200]}")
            results.append({"level": level, "status": "no_code"})
            continue

        # Build wrapper
        wrapper = build_load_inline_wrapper(cuda_code, pytorch_code)
        if not wrapper:
            print(f"  ✗ No torch::Tensor binding found (gen={gen_time:.1f}s)")
            results.append({"level": level, "status": "no_binding"})
            continue

        # Compile + evaluate
        try:
            eval_res = evaluate(wrapper, pytorch_code)
        except Exception as e:
            print(f"  ✗ Evaluate exception: {e}")
            results.append({"level": level, "status": "eval_error"})
            continue

        if eval_res is None:
            print(f"  ✗ Compile failed (gen={gen_time:.1f}s)")
            results.append({"level": level, "status": "compile_fail"})
        elif eval_res.get("correct", False):
            speedup = eval_res.get("speedup", 1.0)
            print(f"  ✓ PASS  speedup={speedup:.2f}x  (gen={gen_time:.1f}s)")
            results.append({"level": level, "status": "pass", "speedup": speedup})
        else:
            err = eval_res.get("compiler_error", "wrong output")
            print(f"  ~ Compiled but wrong output (gen={gen_time:.1f}s): {err[:80]}")
            results.append({"level": level, "status": "wrong_output"})

    # ── 5. Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for level in LEVELS:
        lvl_results = [r for r in results if r["level"] == level]
        if not lvl_results:
            continue
        passes = sum(1 for r in lvl_results if r["status"] == "pass")
        compiles = sum(1 for r in lvl_results if r["status"] in ("pass", "wrong_output"))
        total = len(lvl_results)
        print(f"  {level}: {passes}/{total} pass, {compiles}/{total} compile")
    print()
    total_pass = sum(1 for r in results if r["status"] == "pass")
    total_compile = sum(1 for r in results if r["status"] in ("pass", "wrong_output"))
    print(f"  TOTAL: {total_pass}/{len(results)} pass, {total_compile}/{len(results)} compile")

    _shutdown_sglang()


if __name__ == "__main__":
    main()
