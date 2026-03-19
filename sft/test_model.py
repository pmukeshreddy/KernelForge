"""
test_model.py - Evaluate kernelforge-qwen3-14b-lora-v2 from HuggingFace.

Loads: Qwen/Qwen3-14B (base) + mukeshreddy/kernelforge-qwen3-14b-lora-v2 (LoRA)
Tests: 8 diverse CUDA kernel prompts
Reports: generation quality, CUDA construct presence, nvcc compile rate
"""

import os
import re
import subprocess
import tempfile
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL   = "Qwen/Qwen3-14B"
HF_ADAPTER   = "mukeshreddy/kernelforge-qwen3-14b-lora-v2"

SYSTEM_PROMPT = (
    "You are an expert NVIDIA CUDA Systems Engineer. "
    "Your objective is to write optimized CUDA C++ kernels to replace PyTorch operations. "
    "Output only the CUDA C++ implementation inside a ```cpp block."
)

TEST_PROMPTS = [
    {
        "name": "ReLU",
        "pytorch": """\
import torch, torch.nn as nn
class Model(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)""",
    },
    {
        "name": "GELU",
        "pytorch": """\
import torch, torch.nn as nn
class Model(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(x)""",
    },
    {
        "name": "Vector Add",
        "pytorch": """\
import torch, torch.nn as nn
class Model(nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b""",
    },
    {
        "name": "Softmax",
        "pytorch": """\
import torch, torch.nn as nn
class Model(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(x, dim=-1)""",
    },
    {
        "name": "Layer Norm",
        "pytorch": """\
import torch, torch.nn as nn
class Model(nn.Module):
    def __init__(self): super().__init__(); self.ln = nn.LayerNorm(512)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x)""",
    },
    {
        "name": "MatMul",
        "pytorch": """\
import torch, torch.nn as nn
class Model(nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.matmul(A, B)""",
    },
    {
        "name": "Max Pooling",
        "pytorch": """\
import torch, torch.nn as nn
class Model(nn.Module):
    def __init__(self): super().__init__(); self.pool = nn.MaxPool2d(2)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)""",
    },
    {
        "name": "SiLU (Swish)",
        "pytorch": """\
import torch, torch.nn as nn
class Model(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)""",
    },
]

CUDA_KEYWORDS = ["__global__", "__device__", "<<<", "blockIdx", "threadIdx", "cudaMalloc"]


def make_prompt(pytorch_code: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Write an optimized CUDA kernel for this PyTorch model:\n\n```python\n{pytorch_code}\n```\n"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def extract_code(text: str) -> str:
    # Strip <think>...</think> blocks first
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    for lang in ["cpp", "cuda", "c++"]:
        m = re.search(rf"```{lang}\s*(.*?)```", text, re.DOTALL)
        if m:
            return m.group(1).strip()
    m = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def check_nvcc() -> bool:
    try:
        subprocess.run(["nvcc", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def compile_kernel(code: str) -> tuple[bool, str]:
    import torch
    import torch.utils.cpp_extension
    import sysconfig
    torch_include  = torch.utils.cpp_extension.include_paths()
    torch_lib      = torch.utils.cpp_extension.library_paths()
    python_include = sysconfig.get_path("include")

    includes = [f"-I{p}" for p in torch_include] + [f"-I{python_include}"]
    libs     = [f"-L{p}" for p in torch_lib]

    with tempfile.NamedTemporaryFile(suffix=".cu", mode="w", delete=False) as f:
        f.write(code)
        cu_path = f.name
    obj_path = cu_path.replace(".cu", ".o")
    try:
        cmd = ["nvcc", "-c", cu_path, "-o", obj_path,
               "-Wno-deprecated-gpu-targets", "-std=c++17"] + includes + libs
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        err = result.stderr.strip()
        # Filter out pure warnings, keep only errors
        error_lines = [l for l in err.splitlines()
                       if "error:" in l.lower() or "fatal" in l.lower()]
        return result.returncode == 0, "\n".join(error_lines) if error_lines else err[:300]
    except subprocess.TimeoutExpired:
        return False, "Compilation timeout"
    except Exception as e:
        return False, str(e)
    finally:
        for p in [cu_path, obj_path]:
            if os.path.exists(p):
                os.remove(p)


def main():
    print(f"Base model  : {BASE_MODEL}")
    print(f"LoRA adapter: {HF_ADAPTER}\n")

    nvcc_ok = check_nvcc()
    if not nvcc_ok:
        print("WARNING: nvcc not found — compilation checks will be skipped.\n")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(HF_ADAPTER)

    print("Loading base model (bfloat16)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, HF_ADAPTER)
    model.eval()
    print("Model ready.\n")
    print("=" * 60)

    results = []

    for item in TEST_PROMPTS:
        name = item["name"]
        print(f"\n[{name}]")

        prompt = make_prompt(item["pytorch"])
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=768,
                do_sample=True,
                temperature=0.2,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
        code = extract_code(generated)

        has_cuda = any(kw in code for kw in CUDA_KEYWORDS)
        compiled = False
        compile_err = ""

        if has_cuda and nvcc_ok and len(code) > 50:
            compiled, compile_err = compile_kernel(code)

        status = "PASS" if compiled else ("NO_NVCC" if not nvcc_ok else ("CUDA_PRESENT" if has_cuda else "FAIL"))
        print(f"  Status       : {status}")
        print(f"  CUDA keywords: {has_cuda}")
        if nvcc_ok:
            print(f"  nvcc compile : {compiled}")
        if compile_err:
            print(f"  Error        : {compile_err[:200]}")
        print(f"  Code preview :\n{code[:300]}\n{'...' if len(code) > 300 else ''}")

        results.append({
            "name": name,
            "has_cuda": has_cuda,
            "compiled": compiled,
            "code_len": len(code),
        })

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total         = len(results)
    cuda_count    = sum(r["has_cuda"] for r in results)
    compile_count = sum(r["compiled"] for r in results)

    for r in results:
        tag = "+" if (r["compiled"] or (r["has_cuda"] and not nvcc_ok)) else "x"
        print(f"  [{tag}] {r['name']:<20} cuda={r['has_cuda']}  compiled={r['compiled']}  len={r['code_len']}")

    print(f"\nHas CUDA constructs : {cuda_count}/{total}  ({cuda_count/total*100:.0f}%)")
    if nvcc_ok:
        print(f"nvcc compile pass   : {compile_count}/{total}  ({compile_count/total*100:.0f}%)")
    else:
        print("nvcc compile pass   : skipped (nvcc not available)")


if __name__ == "__main__":
    main()
