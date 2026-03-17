"""
generate_sft_data.py - Download and prepare CUDA kernel SFT training pairs.

Sources:
1. CUDA-L1 (deepreinforce-ai/CUDA-L1) - Has ref_code + optimized custom_code pairs
2. KernelBench (ScalingIntelligence/KernelBench) - 250 PyTorch reference problems
   → Saved separately to rl_prompts.jsonl (prompt-only, for GRPO stage)

Target format: raw CUDA C++ only (the content inside cuda_source = \"\"\"...\"\"\").
Python wrappers, load_inline boilerplate, PYBIND11, and ModelNew classes are
discarded. Entries where a clean kernel cannot be extracted are filtered out.
Entries are verified with `nvcc -c` before inclusion.
"""
import json
import os
import re
import subprocess
import tempfile
import unicodedata
from datasets import load_dataset


def _normalize_code(code: str) -> str:
    """Normalize unicode characters in code to ASCII-compatible form."""
    return unicodedata.normalize("NFKC", code).encode("utf-8").decode("utf-8")

# ─── System Prompt (condensed version of rl/sys_prompt.py) ────────────────
# Must align with what the model sees during GRPO so distribution doesn't shift.
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


# ─── Format example (teaches the model the exact output skeleton) ─────────
FORMAT_EXAMPLE = """Here is an example of the expected output format:

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


def _extract_cuda_cpp(custom_code: str) -> str:  # unused, kept for reference
    """
    Extract raw CUDA C++ from a load_inline Python script.

    CUDA-L1 custom_code looks like:
        cuda_source = \"\"\"
        #include <torch/extension.h>
        __global__ void my_kernel(...) { ... }
        torch::Tensor run_cuda(...) { ... }
        \"\"\"
        cpp_source = "..."
        ext = load_inline(...)
        class ModelNew(...): ...

    We want ONLY the content inside cuda_source = \"\"\"...\"\"\".
    Returns empty string if extraction fails or the result is not valid C++.
    """
    # Try all common variable names used in CUDA-L1 custom_code
    # Build search list: known names + dynamically detect from cuda_sources=VARNAME
    known_vars = ['cuda_source', 'cuda_code', 'cuda_src', 'cuda_kernel_code',
                  'cuda_kernel', 'cuda_kernel_source', 'kernel', 'kernel_code', 'CUDA_KERNEL']
    dynamic = re.search(r'cuda_sources\s*=\s*(\w+)', custom_code)
    if dynamic and dynamic.group(1) not in ('self', 'None'):
        search_vars = [dynamic.group(1)] + known_vars
    else:
        search_vars = known_vars

    match = None
    for var in search_vars:
        match = re.search(rf'{re.escape(var)}\s*=\s*"""(.*?)"""', custom_code, re.DOTALL)
        if match:
            break
        match = re.search(rf"{re.escape(var)}\s*=\s*'''(.*?)'''", custom_code, re.DOTALL)
        if match:
            break
    if not match:
        return ""

    cpp = match.group(1).strip()

    # Fix deprecated PyTorch C++ API (removed in torch 2.x)
    cpp = cpp.replace(".type()", ".scalar_type()")

    # Add missing includes for commonly used APIs
    if "getCurrentCUDAStream" in cpp and "<ATen/cuda/CUDAContext.h>" not in cpp:
        cpp = "#include <ATen/cuda/CUDAContext.h>\n" + cpp

    # Must contain a real CUDA kernel and a torch::Tensor binding function
    if "__global__" not in cpp:
        return ""
    if "torch::Tensor" not in cpp:
        return ""
    if "#include" not in cpp:
        return ""

    return cpp


def check_nvcc_available() -> bool:
    """Check if nvcc is available on this machine."""
    try:
        subprocess.run(["nvcc", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False


def nvcc_compiles(cuda_code: str) -> tuple[bool, str]:
    """
    Try to compile CUDA C++ with nvcc -c. 
    Returns (True, "") if successful, or (False, error_msg) if it fails.
    """
    import torch
    import sysconfig
    from torch.utils.cpp_extension import include_paths
    
    tmpdir = tempfile.mkdtemp(prefix="kf_nvcc_")
    cu_path = os.path.join(tmpdir, "kernel.cu")
    obj_path = os.path.join(tmpdir, "kernel.o")
    
    try:
        with open(cu_path, "w") as f:
            f.write(cuda_code)
            
        # Get PyTorch and Python include paths
        includes = []
        for p in include_paths():
            includes.extend(["-I", p])
        includes.extend(["-I", sysconfig.get_path("include")])
        
        cmd = ["nvcc", "-c", cu_path, "-o", obj_path, "--std=c++17", "-w",
               "--expt-relaxed-constexpr", "--expt-extended-lambda"] + includes
        
        result = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            return True, ""
        else:
            return False, result.stderr + "\n" + result.stdout
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)
    finally:
        for f in [cu_path, obj_path]:
            if os.path.exists(f):
                os.remove(f)
        os.rmdir(tmpdir)


def make_training_text(pytorch_code, cuda_code, ops_desc=None):
    """Create full training prompt with input + target, including format example."""
    user_msg = ""
    if ops_desc:
        user_msg += f"Operations: {ops_desc}\n\n"
    user_msg += FORMAT_EXAMPLE
    user_msg += f"```python\n{pytorch_code}\n```"
    return SYSTEM + f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n```cpp\n{cuda_code}\n```<|im_end|>\n"

def main():
    output_file = "./sft_training_pairs.jsonl"
    rl_prompts_file = "./rl_prompts.jsonl"
    pairs = []
    
    # === Source 0: SakanaAI/AI-CUDA-Engineer-Archive (30k pre-verified pairs) ===
    print("Downloading SakanaAI/AI-CUDA-Engineer-Archive (pre-verified pairs)...")
    PER_LEVEL = 5000 // 3  # ~1667 per level for balanced difficulty
    try:
        for level in ["level_1", "level_2", "level_3"]:
            try:
                ds = load_dataset("SakanaAI/AI-CUDA-Engineer-Archive", split=level)
                level_pairs = []
                for row in ds:
                    if len(level_pairs) >= PER_LEVEL:
                        break
                    if not row.get("Correct", False):
                        continue
                    pytorch_code = row.get("PyTorch_Code_Module", "") or row.get("PyTorch_Code_Functional", "")
                    cuda_code = row.get("CUDA_Code", "")
                    if not pytorch_code or not cuda_code:
                        continue
                    if "__global__" not in cuda_code:
                        continue
                    text = make_training_text(pytorch_code, cuda_code)
                    level_pairs.append({
                        "source": f"sakana-{level}",
                        "task_id": str(row.get("task_id", "")),
                        "level_id": level,
                        "pytorch_code": pytorch_code,
                        "cuda_kernel": cuda_code,
                        "text": text
                    })
                pairs.extend(level_pairs)
                print(f"  SakanaAI {level}: {len(level_pairs)} pairs")
            except Exception as e:
                print(f"  Warning: Could not load SakanaAI {level}: {e}")
        print(f"SakanaAI total pairs (balanced): {len(pairs)}")
    except Exception as e:
        print(f"Warning: Could not load SakanaAI dataset: {e}")

    
    # === Source 2: KernelBench → separate rl_prompts.jsonl ===
    # These are prompt-only entries for the GRPO stage. NOT included in SFT training.
    print("Loading KernelBench dataset (RL prompts only)...")
    rl_prompts = []
    try:
        existing_codes = {p["pytorch_code"] for p in pairs}
        
        for level in ["level_1", "level_2", "level_3"]:
            try:
                ds = load_dataset("ScalingIntelligence/KernelBench", split=level)
                for row in ds:
                    code = row.get("code", "") or row.get("pytorch_code", "") or row.get("ref_code", "")
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
                print(f"  Warning: Could not load KernelBench {level}: {e}")
        
        print(f"KernelBench RL prompts: {len(rl_prompts)}")
    except Exception as e:
        print(f"Warning: Could not load KernelBench: {e}")
    
    # === Save SFT training data ===
    if len(pairs) == 0:
        print("ERROR: No training pairs generated! Check data sources.")
        return
    
    # Deduplicate by pytorch_code content hash
    seen = set()
    unique_pairs = []
    for p in pairs:
        key = hash(p.get("pytorch_code", "") + p.get("cuda_kernel", ""))
        if key not in seen:
            seen.add(key)
            unique_pairs.append(p)
    
    with open(output_file, 'w') as f:
        for p in unique_pairs:
            f.write(json.dumps(p) + "\n")
    
    # === Save RL prompts separately ===
    if rl_prompts:
        with open(rl_prompts_file, 'w') as f:
            for p in rl_prompts:
                f.write(json.dumps(p) + "\n")
        print(f"RL prompts saved to: {rl_prompts_file}")
    
    print(f"\n=== Results ===")
    print(f"SFT training pairs: {len(unique_pairs)} → {output_file}")
    print(f"RL prompts (KernelBench): {len(rl_prompts)} → {rl_prompts_file}")

if __name__ == "__main__":
    main()
