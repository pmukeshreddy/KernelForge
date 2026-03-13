"""
generate_sft_data.py - Generate CUDA kernel SFT pairs using vLLM batch inference.

Uses KernelBench (250 problems) + CUDA-Agent-Ops-6K (750 problems) = ~1000 prompts.
vLLM batches them all on GPU for 10-50x faster generation vs sequential HuggingFace.
Filters for real CUDA constructs (__global__, blockIdx, etc).
"""
import json
import subprocess
import os
from datasets import load_dataset, concatenate_datasets
from vllm import LLM, SamplingParams

SYSTEM = """<|im_start|>system
You are an expert GPU kernel developer. Rewrite PyTorch operations into optimized CUDA C++ code with __global__ kernels, proper thread indexing, shared memory, and memory coalescing. Output only the CUDA C++ code.
<|im_end|>
"""

def make_prompt(pytorch_code, ops_desc=None):
    user_msg = ""
    if ops_desc:
        user_msg += f"Operations: {ops_desc}\n\n"
    user_msg += f"Rewrite this PyTorch model as an optimized CUDA kernel with __global__ functions, thread indexing, and proper memory access:\n```python\n{pytorch_code}\n```"
    return SYSTEM + f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n```cpp\n"

def extract_cuda(text):
    """Extract CUDA code, stopping at closing fence."""
    if "```" in text:
        return text.split("```")[0].strip()
    return text.strip()

def compile_check(code, filename="/tmp/temp_kernel.cu"):
    """Quick nvcc compile check."""
    with open(filename, "w") as f:
        f.write(code)
    try:
        result = subprocess.run(
            ["nvcc", "-c", filename, "-o", "/tmp/temp_kernel.o"],
            capture_output=True, text=True, timeout=15
        )
        return result.returncode == 0
    except:
        return False
    finally:
        for f in [filename, "/tmp/temp_kernel.o"]:
            if os.path.exists(f):
                os.remove(f)

def main():
    output_file = "./sft_training_pairs.jsonl"
    
    # Load KernelBench (250 problems across 3 levels)
    print("Loading KernelBench dataset...")
    kb1 = load_dataset("ScalingIntelligence/KernelBench", split="level_1")
    kb2 = load_dataset("ScalingIntelligence/KernelBench", split="level_2")
    kb3 = load_dataset("ScalingIntelligence/KernelBench", split="level_3")
    
    # Load CUDA-Agent-Ops (first 750 for total ~1000)
    print("Loading CUDA-Agent-Ops-6K dataset...")
    ops = load_dataset("BytedTsinghua-SIA/CUDA-Agent-Ops-6K", split="train[:750]")
    
    # Build all prompts
    prompts = []
    metadata = []
    
    for ds_split in [kb1, kb2, kb3]:
        for ex in ds_split:
            prompts.append(make_prompt(ex["code"]))
            metadata.append({
                "source": "kernelbench",
                "name": ex.get("name", ""),
                "level": ex.get("level", ""),
                "pytorch_code": ex["code"]
            })
    
    for ex in ops:
        prompts.append(make_prompt(ex["code"], ex.get("ops", "")))
        metadata.append({
            "source": "cuda-agent-ops",
            "ops": ex.get("ops", ""),
            "pytorch_code": ex["code"]
        })
    
    print(f"Total prompts: {len(prompts)}")
    
    # Batch generate with vLLM
    print("Loading model with vLLM...")
    llm = LLM(
        model="Qwen/Qwen3-8B",
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.90
    )
    params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=2048,
        repetition_penalty=1.1
    )
    
    print("Generating CUDA kernels (batch mode)...")
    outputs = llm.generate(prompts, params)
    
    # Filter and save
    total = 0
    has_cuda = 0
    compiles = 0
    
    # Clear output file
    open(output_file, 'w').close()
    
    for out, meta in zip(outputs, metadata):
        total += 1
        text = out.outputs[0].text
        cuda_code = extract_cuda(text)
        
        # Must have real CUDA constructs
        if not any(kw in cuda_code for kw in ["__global__", "__device__", "blockIdx", "threadIdx"]):
            continue
        has_cuda += 1
        
        # Must be substantial
        if len(cuda_code) < 100:
            continue
        
        # Compile check
        if not compile_check(cuda_code):
            continue
        compiles += 1
        
        # Build full training text
        full_text = make_prompt(meta["pytorch_code"], meta.get("ops")) + cuda_code + "\n```<|im_end|>\n"
        meta["cuda_kernel"] = cuda_code
        meta["text"] = full_text
        
        with open(output_file, "a") as f:
            f.write(json.dumps(meta) + "\n")
    
    print(f"\n=== Results ===")
    print(f"Total generated: {total}")
    print(f"Has CUDA constructs: {has_cuda}")
    print(f"Compiles with nvcc: {compiles}")
    print(f"Success rate: {compiles/total*100:.1f}%")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    main()
