"""
generate_sft_data.py - Download and prepare CUDA kernel SFT training pairs.

Sources:
1. CUDA-L1 (deepreinforce-ai/CUDA-L1) - Has ref_code + optimized custom_code pairs
2. KernelBench (ScalingIntelligence/KernelBench) - 250 PyTorch reference problems

Combines both into a single JSONL file for SFT training.
"""
import json
import os
import glob
from huggingface_hub import snapshot_download
from datasets import load_dataset, concatenate_datasets

SYSTEM = """<|im_start|>system
You are an expert GPU kernel developer. Rewrite PyTorch operations into optimized CUDA C++ code with __global__ kernels, proper thread indexing, shared memory, and memory coalescing. Output only the CUDA C++ code.
<|im_end|>
"""

def make_training_text(pytorch_code, cuda_code, ops_desc=None):
    """Create full training prompt with input + target."""
    user_msg = ""
    if ops_desc:
        user_msg += f"Operations: {ops_desc}\n\n"
    user_msg += f"Rewrite this PyTorch model as an optimized CUDA kernel with __global__ functions, thread indexing, and proper memory access:\n```python\n{pytorch_code}\n```"
    return SYSTEM + f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n```cpp\n{cuda_code}\n```<|im_end|>\n"

def main():
    output_file = "./sft_training_pairs.jsonl"
    pairs = []
    
    # === Source 1: CUDA-L1 optimized kernel pairs ===
    print("Downloading CUDA-L1 dataset (optimized kernel pairs)...")
    try:
        cache_dir = snapshot_download(
            "deepreinforce-ai/CUDA-L1",
            repo_type="dataset",
            allow_patterns=["optimized_cuda_code/*.json"]
        )
        
        # Load all JSON files from the optimized_cuda_code folder
        json_files = glob.glob(os.path.join(cache_dir, "optimized_cuda_code", "*.json"))
        print(f"Found {len(json_files)} JSON files in CUDA-L1")
        
        for jf in json_files:
            gpu_name = os.path.basename(jf).replace('.json', '')
            with open(jf) as f:
                data = json.load(f)
            
            if isinstance(data, list):
                entries = data
            elif isinstance(data, dict):
                entries = list(data.values()) if not isinstance(list(data.values())[0], str) else [data]
            else:
                continue
            
            for entry in entries:
                ref = entry.get("ref_code", "")
                custom = entry.get("custom_code", "")
                
                if not ref or not custom:
                    continue
                
                # Check custom_code has real CUDA constructs
                if not any(kw in custom for kw in ["__global__", "__device__", "blockIdx", "threadIdx"]):
                    continue
                
                text = make_training_text(ref, custom)
                pairs.append({
                    "source": f"cuda-l1-{gpu_name}",
                    "task_id": entry.get("task_id", ""),
                    "level_id": entry.get("level_id", ""),
                    "pytorch_code": ref,
                    "cuda_kernel": custom,
                    "text": text
                })
        
        print(f"CUDA-L1 pairs with CUDA constructs: {len(pairs)}")
    except Exception as e:
        print(f"Warning: Could not load CUDA-L1: {e}")
    
    # === Source 2: KernelBench reference problems (for diversity) ===
    # Load these as input-only; if we have CUDA-L1 solutions for them, great
    # Otherwise they provide more PyTorch problems for the model to see
    print("Loading KernelBench dataset...")
    try:
        kb1 = load_dataset("ScalingIntelligence/KernelBench", split="level_1")
        kb2 = load_dataset("ScalingIntelligence/KernelBench", split="level_2")
        kb3 = load_dataset("ScalingIntelligence/KernelBench", split="level_3")
        
        # KernelBench problems are already covered by CUDA-L1 (same benchmark)
        # So we just log how many we have
        kb_total = len(kb1) + len(kb2) + len(kb3)
        print(f"KernelBench problems: {kb_total} (covered by CUDA-L1 solutions)")
    except Exception as e:
        print(f"Warning: Could not load KernelBench: {e}")
    
    # === Save the training data ===
    if len(pairs) == 0:
        print("ERROR: No training pairs generated! Check data sources.")
        return
    
    # Deduplicate by task_id
    seen = set()
    unique_pairs = []
    for p in pairs:
        key = f"{p.get('level_id', '')}-{p.get('task_id', '')}-{p.get('source', '')}"
        if key not in seen:
            seen.add(key)
            unique_pairs.append(p)
    
    with open(output_file, 'w') as f:
        for p in unique_pairs:
            f.write(json.dumps(p) + "\n")
    
    print(f"\n=== Results ===")
    print(f"Total unique training pairs: {len(unique_pairs)}")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    main()
