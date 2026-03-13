import os
import subprocess
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

def check_nvcc():
    """Check if nvcc is available."""
    try:
        subprocess.run(["nvcc", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False

def compile_kernel(kernel_code, filename="temp_kernel.cu"):
    """Compile with nvcc. Returns (success, error)."""
    with open(filename, "w") as f:
        f.write(kernel_code)
    try:
        result = subprocess.run(
            ["nvcc", "-c", filename, "-o", "temp_kernel.o"], 
            capture_output=True, text=True, timeout=15
        )
        return result.returncode == 0, result.stderr
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)
    finally:
        for f in [filename, "temp_kernel.o"]:
            if os.path.exists(f):
                os.remove(f)

SYSTEM = """<|im_start|>system
You are an expert GPU kernel developer. Rewrite PyTorch operations into optimized CUDA C++ code with __global__ kernels, proper thread indexing, shared memory, and memory coalescing. Output only the CUDA C++ code.
<|im_end|>
"""

def make_prompt(pytorch_code):
    return SYSTEM + f"<|im_start|>user\nRewrite this PyTorch model as an optimized CUDA kernel with __global__ functions, thread indexing, and proper memory access:\n```python\n{pytorch_code}\n```<|im_end|>\n<|im_start|>assistant\n```cpp\n"

def main():
    if not check_nvcc():
        print("Warning: 'nvcc' not found.")
        return

    model_path = "./sft_qwen3_8b_cuda"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Using base model.")
        model_path = "Qwen/Qwen3-8B"

    print(f"Loading Model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    
    # Use KernelBench level_1 as eval set (same format as training)
    print("Loading KernelBench level_1 for evaluation...")
    ds = load_dataset("ScalingIntelligence/KernelBench", split="level_1")
    
    # Take last 20 as eval
    eval_ds = ds.select(range(max(0, len(ds)-20), len(ds)))
    
    success_count = 0
    has_cuda_count = 0
    total = len(eval_ds)

    print(f"Generating and testing kernels for {total} examples...")
    pbar = tqdm(eval_ds, total=total)
    for example in pbar:
        pytorch_code = example.get("code", "")
        prompt = make_prompt(pytorch_code)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=2048, temperature=0.2, 
                do_sample=True, pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
        
        # Extract code
        code = ""
        for lang in ["cpp", "cuda", "c++"]:
            marker = f"```{lang}"
            if marker in response:
                parts = response.split(marker, 1)[1]
                if "```" in parts:
                    code = parts.split("```", 1)[0].strip()
                    break
        if not code and "```" in response and response.count("```") >= 2:
            code = response.split("```", 1)[1].split("```", 1)[0].strip()
        
        # Check for CUDA constructs
        if len(code) < 50:
            pbar.set_postfix({"pass": success_count, "cuda": has_cuda_count, "fail": pbar.n+1-success_count})
            continue
        
        has_cuda = any(kw in code for kw in ["__global__", "__device__", "<<<", "blockIdx", "threadIdx"])
        if has_cuda:
            has_cuda_count += 1
        else:
            pbar.set_postfix({"pass": success_count, "cuda": has_cuda_count, "fail": pbar.n+1-success_count})
            continue
            
        success, err = compile_kernel(code)
        if success:
            success_count += 1
        
        pbar.set_postfix({"pass": success_count, "cuda": has_cuda_count, "fail": pbar.n+1-success_count})
            
    print(f"\n=== Final Results ===")
    print(f"Has CUDA constructs: {has_cuda_count}/{total} ({has_cuda_count/total*100:.1f}%)")
    print(f"Compiles with nvcc:  {success_count}/{total} ({success_count/total*100:.1f}%)")

if __name__ == "__main__":
    main()
