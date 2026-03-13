"""
generate_sft_data.py - Generate CUDA kernel SFT training pairs via rejection sampling.

Takes PyTorch problems from CUDA-Agent-Ops-6K and uses the base Qwen3-8B model
to generate CUDA kernel attempts. Filters for ones that compile with nvcc.
Saves the successful pairs as the SFT training dataset.
"""
import os
import json
import subprocess
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

from dataset_hf import format_cuda_prompt

def compile_kernel(kernel_code, filename="temp_kernel.cu"):
    """Attempt to compile CUDA code with nvcc. Returns (success, error)."""
    with open(filename, "w") as f:
        f.write(kernel_code)
    try:
        result = subprocess.run(
            ["nvcc", "-c", filename, "-o", "temp_kernel.o"],
            capture_output=True, text=True, timeout=15
        )
        success = result.returncode == 0
        return success, result.stderr
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)
    finally:
        for f in [filename, "temp_kernel.o"]:
            if os.path.exists(f):
                os.remove(f)

def extract_code(response):
    """Extract code from model response."""
    for lang in ["cpp", "cuda", "c++"]:
        marker = f"```{lang}"
        if marker in response:
            parts = response.split(marker, 1)[1]
            if "```" in parts:
                return parts.split("```", 1)[0].strip()
    if "```" in response and response.count("```") >= 2:
        code = response.split("```", 1)[1].split("```", 1)[0].strip()
        if code.startswith("c\n"): code = code[2:].strip()
        return code
    return response.strip()

def main():
    model_id = "Qwen/Qwen3-8B"
    output_file = "./sft_training_pairs.jsonl"
    num_attempts_per_problem = 1  # Generate 1 attempt per problem (faster)
    max_problems = 1000  # 1000 good pairs is plenty for SFT
    
    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
    
    print("Loading dataset...")
    ds = load_dataset("BytedTsinghua-SIA/CUDA-Agent-Ops-6K", split="train")
    
    successful_pairs = []
    total_attempts = 0
    compile_successes = 0
    
    print(f"Generating CUDA kernels for {min(len(ds), max_problems)} problems...")
    for i, example in enumerate(tqdm(ds)):
        if i >= max_problems:
            break
            
        ops_desc = example.get("ops", "")
        pytorch_code = example.get("code", "")
        
        # Create prompt (input only, no target)
        prompt = format_cuda_prompt(ops_desc, pytorch_code)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = inputs.input_ids.shape[1]
        
        best_kernel = None
        
        for attempt in range(num_attempts_per_problem):
            total_attempts += 1
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    temperature=0.7,  # Higher temp for diversity
                    do_sample=True,
                    top_p=0.95,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
            kernel_code = extract_code(response)
            
            # Skip empty or trivially short code
            if len(kernel_code) < 50:
                continue
            
            # Must have actual CUDA kernel indicators
            has_kernel = any(kw in kernel_code for kw in ["__global__", "__device__", "<<<", "blockIdx", "threadIdx"])
            if not has_kernel:
                continue
                
            # Try to compile
            success, error = compile_kernel(kernel_code)
            if success:
                compile_successes += 1
                best_kernel = kernel_code
                break  # Got a good one, move to next problem
        
        if best_kernel:
            # Save the successful pair
            pair = {
                "ops": ops_desc,
                "pytorch_code": pytorch_code,
                "cuda_kernel": best_kernel,
                "text": format_cuda_prompt(ops_desc, pytorch_code, best_kernel)
            }
            successful_pairs.append(pair)
            
            # Save incrementally
            with open(output_file, "a") as f:
                f.write(json.dumps(pair) + "\n")
        
        # Progress report every 100 problems
        if (i + 1) % 100 == 0:
            rate = len(successful_pairs) / (i + 1) * 100
            print(f"\nProgress: {i+1} problems, {len(successful_pairs)} successful ({rate:.1f}%), "
                  f"{compile_successes}/{total_attempts} compilations succeeded")
    
    print(f"\n=== Final Results ===")
    print(f"Problems processed: {min(len(ds), max_problems)}")
    print(f"Successful pairs: {len(successful_pairs)}")
    print(f"Success rate: {len(successful_pairs)/min(len(ds), max_problems)*100:.1f}%")
    print(f"Compile attempts: {total_attempts}")
    print(f"Compile successes: {compile_successes}")
    print(f"Data saved to: {output_file}")

if __name__ == "__main__":
    main()
