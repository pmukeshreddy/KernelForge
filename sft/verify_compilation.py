import os
import subprocess
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from dataset_hf import format_cuda_prompt

def check_nvcc():
    """Check if nvcc is available."""
    try:
        subprocess.run(["nvcc", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False

def compile_kernel(kernel_code, filename="temp_kernel.cu"):
    """
    Writes the kernel to a file and attempts to compile it with nvcc.
    Returns (success: bool, error_message: str)
    """
    with open(filename, "w") as f:
        f.write(kernel_code)
    
    try:
        # We only compile (-c) to an object file without linking or executing
        # This checks syntax and basic CUDA semantics
        result = subprocess.run(
            ["nvcc", "-c", filename, "-o", "temp_kernel.o"], 
            capture_output=True, 
            text=True, 
            timeout=15
        )
        success = result.returncode == 0
        return success, result.stderr
    except subprocess.TimeoutExpired:
        return False, "Timeout during compilation"
    except Exception as e:
        return False, str(e)
    finally:
        # Cleanup
        if os.path.exists(filename):
            os.remove(filename)
        if os.path.exists("temp_kernel.o"):
            os.remove("temp_kernel.o")

def main():
    if not check_nvcc():
        print("Warning: 'nvcc' not found. Ensure CUDA toolkit is installed.")
        return

    model_path = "./sft_qwen3_8b_cuda" # Path to SFT model
    
    if not os.path.exists(model_path):
        print(f"Warning: Model not found at {model_path}. Maybe SFT has not completed yet.")
        print("Using base model for demonstration...")
        model_path = "Qwen/Qwen3-8B"

    print(f"Loading Tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    print(f"Loading Model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True
    )
    
    # A small proxy eval set to verify cold-start ability
    instructions = [
        "Write a CUDA kernel to perform element-wise addition of two arrays.",
        "Write an optimized CUDA kernel for SGEMM (Single Precision Matrix Multiplication) using shared memory.",
        "Write a CUDA kernel to perform a parallel reduction (sum) over an array.",
        "Write a CUDA kernel to compute the sigmoid of every element in a 2D tensor."
    ]
    
    success_count = 0
    results = []

    print("Generating and testing kernels...")
    for instr in tqdm(instructions):
        prompt = format_cuda_prompt(instr)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Determine offset to extract only the generated tokens
        prompt_len = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=1024, 
                temperature=0.2, 
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
        
        # Extract the cpp code block
        code = ""
        if "```cpp" in response:
            code = response.split("```cpp")[1].split("```")[0].strip()
        elif "```cuda" in response:
            code = response.split("```cuda")[1].split("```")[0].strip()
        elif "```" in response:
             code = response.split("```")[1].strip()
        else:
            code = response.strip()
            
        # Basic sanity check wrapping for incomplete generations (e.g. missing includes)
        if "#include" not in code:
            code = "#include <cuda_runtime.h>\n#include <stdio.h>\n" + code
            
        success, err = compile_kernel(code)
        
        results.append({
            "instruction": instr,
            "success": success,
            "error": err
        })
        
        if success:
            success_count += 1
        else:
            print(f"\n[FAIL] {instr}")
            print(f"Compiler Error:\n{err}")
            
    compilation_rate = (success_count / len(instructions)) * 100
    print(f"\nFinal Compilation Rate: {success_count}/{len(instructions)} ({compilation_rate:.2f}%)")

if __name__ == "__main__":
    main()
