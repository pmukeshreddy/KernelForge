"""
generate_sft_data.py - Download and prepare CUDA kernel SFT training pairs.

Sources:
1. CUDA-L1 (deepreinforce-ai/CUDA-L1) - Has ref_code + optimized custom_code pairs
2. KernelBench (ScalingIntelligence/KernelBench) - 250 PyTorch reference problems

Combines both into a single JSONL file for SFT training.

Target format: raw CUDA C++ only (the content inside cuda_source = \"\"\"...\"\"\").
Python wrappers, load_inline boilerplate, PYBIND11, and ModelNew classes are
discarded. Entries where a clean kernel cannot be extracted are filtered out.
"""
import json
import os
import re
import glob
from huggingface_hub import snapshot_download
from datasets import load_dataset

SYSTEM = """<|im_start|>system
You are an expert GPU kernel developer. Rewrite PyTorch operations into optimized CUDA C++ code with __global__ kernels, proper thread indexing, shared memory, and memory coalescing. Output only the CUDA C++ code.
<|im_end|>
"""

def extract_cuda_cpp(custom_code: str) -> str:
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
    # Some entries use cuda_source, others use cuda_code, cuda_src, etc.
    for var in [r'cuda_source', r'cuda_code', r'cuda_src', r'cuda_kernel_code']:
        match = re.search(rf'{var}\s*=\s*"""(.*?)"""', custom_code, re.DOTALL)
        if match:
            break
        match = re.search(rf"{var}\s*=\s*'''(.*?)'''", custom_code, re.DOTALL)
        if match:
            break
    if not match:
        return ""

    cpp = match.group(1).strip()

    # Must contain a real CUDA kernel and a torch::Tensor binding function
    if "__global__" not in cpp:
        return ""
    if "torch::Tensor" not in cpp:
        return ""
    if "#include" not in cpp:
        return ""

    return cpp


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
                content = f.read().strip()
            
            # Try as single JSON first, then as JSONL
            entries = []
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    entries = data
                elif isinstance(data, dict):
                    entries = list(data.values()) if all(isinstance(v, dict) for v in data.values()) else [data]
            except json.JSONDecodeError:
                # Try JSONL (one JSON object per line)
                for line in content.split('\n'):
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            
            print(f"  {gpu_name}: {len(entries)} entries")
            
            skipped = 0
            for entry in entries:
                ref = entry.get("ref_code", "")
                custom = entry.get("custom_code", "")

                if not ref or not custom:
                    skipped += 1
                    continue

                # Extract raw C++ from inside cuda_source = """..."""
                cuda_cpp = extract_cuda_cpp(custom)
                if not cuda_cpp:
                    skipped += 1
                    continue

                text = make_training_text(ref, cuda_cpp)
                pairs.append({
                    "source": f"cuda-l1-{gpu_name}",
                    "task_id": entry.get("task_id", ""),
                    "level_id": entry.get("level_id", ""),
                    "pytorch_code": ref,
                    "cuda_kernel": cuda_cpp,
                    "text": text
                })
            if skipped:
                print(f"    {gpu_name}: skipped {skipped} entries (no clean kernel)")
        
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
