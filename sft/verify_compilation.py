import os
import subprocess
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
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
    
    # A proxy eval set of 50 examples from the dataset to verify cold-start ability
    print("Loading evaluation dataset...")
    # Load a small selection from the dataset (e.g. taking the last 50 from train as a pseudo-val split)
    try:
        ds = load_dataset("BytedTsinghua-SIA/CUDA-Agent-Ops-6K", split="train[-50:]")
    except Exception as e:
        print(f"Failed to load dataset for evaluation: {e}")
        return
    
    success_count = 0
    results = []

    print(f"Generating and testing kernels for {len(ds)} examples...")
    for example in tqdm(ds):
        ops_desc = example.get("ops", "")
        model_py = example.get("model.py", "")
        
        prompt = format_cuda_prompt(ops_desc, model_py)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Determine offset to extract only the generated tokens
        prompt_len = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=4096, # Increased to allow for longer kernel generation
                temperature=0.2, 
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
        
        # Robust code extraction
        code = ""
        if "```cpp" in response and "```" in response.split("```cpp", 1)[1]:
            code = response.split("```cpp", 1)[1].split("```", 1)[0].strip()
        elif "```cuda" in response and "```" in response.split("```cuda", 1)[1]:
            code = response.split("```cuda", 1)[1].split("```", 1)[0].strip()
        elif "```c++" in response and "```" in response.split("```c++", 1)[1]:
            code = response.split("```c++", 1)[1].split("```", 1)[0].strip()
        elif "```" in response and response.count("```") >= 2:
             # Fallback to the first generic code block
             code = response.split("```", 1)[1].split("```", 1)[0].strip()
             # Optionally strip a language identifier like 'c' if mistakenly generated
             if code.startswith("c\n"): code = code[2:].strip()
        else:
            # Absolute fallback: just try to compile whatever it hallucinated
            code = response.strip()
            
        # Basic sanity check wrapping for incomplete generations (e.g. missing includes)
        if "#include" not in code:
            code = "#include <cuda_runtime.h>\n#include <stdio.h>\n" + code
            
        success, err = compile_kernel(code)
        
        results.append({
            "ops": ops_desc,
            "success": success,
            "error": err
        })
        
        if success:
            success_count += 1
        else:
            # We don't print every failure to avoid console spam over 50 examples,
            # but you can log them to a file if needed.
            pass
            
    compilation_rate = (success_count / len(ds)) * 100
    print(f"\nFinal Compilation Rate: {success_count}/{len(ds)} ({compilation_rate:.2f}%)")

if __name__ == "__main__":
    main()
