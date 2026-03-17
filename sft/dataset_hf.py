import os
from datasets import load_dataset
from transformers import AutoTokenizer

# A richer system prompt informed by CUDA Agent's approach
SYSTEM_PROMPT = """<|im_start|>system
You are an expert GPU kernel developer and optimizer. Your goal is to rewrite PyTorch operations into high-performance, optimized CUDA C++ code using appropriate kernel launch parameters, thread block dimensions, shared memory, and memory coalescing techniques.

When given a PyTorch operator implementation and a description of the operation, you must output the equivalent, heavily optimized CUDA C++ implementation with proper kernel launches, thread indexing, and memory access patterns.
<|im_end|>
"""

def format_cuda_prompt(ops_description, pytorch_code, cuda_code=None):
    """
    Format the prompt using the Qwen chat template. 
    Input is the baseline PyTorch code and operation description. 
    Target is the optimized CUDA code.
    """
    prompt = SYSTEM_PROMPT
    prompt += f"<|im_start|>user\nOperation description:\n{ops_description}\n\nPyTorch implementation:\n```python\n{pytorch_code}\n```<|im_end|>\n"
    prompt += f"<|im_start|>assistant\n```cpp\n"
    if cuda_code:
        prompt += f"{cuda_code}\n```<|im_end|>\n"
    return prompt

def prepare_cuda_agent_dataset(dataset_name="BytedTsinghua-SIA/CUDA-Agent-Ops-6K", split="train"):
    """
    Loads and formats the CUDA Agent-Ops-6K dataset.
    Actual columns are: 'ops', 'data_source', 'code'
    - ops: list of operation names (e.g. ["nn.BatchNorm3d", "torch.diag"])
    - data_source: source identifier
    - code: PyTorch Model class implementation
    
    NOTE: This dataset only contains PyTorch problems, NOT CUDA solutions.
    For SFT we need to pair these with CUDA kernel solutions generated 
    via rejection sampling or a teacher model.
    """
    print(f"Loading dataset: {dataset_name} ({split})")
    
    try:
        dataset = load_dataset(dataset_name, split=split)
    except Exception as e:
        print(f"Failed to load dataset from HF: {e}")
        print("Please ensure the dataset name is correct or login with `huggingface-cli login`.")
        raise
    
    print(f"Dataset columns: {dataset.column_names}")
    print(f"Dataset size: {len(dataset)}")
        
    def format_example(example):
        # ops is a string representation of a list of op names
        ops_desc = example.get("ops", "")
        # code contains the PyTorch Model class 
        pytorch_code = example.get("code", "")
        
        # For now, format as input-only (no CUDA target)
        # This will be used for generation/rejection sampling
        formatted_text = format_cuda_prompt(ops_desc, pytorch_code)
        return {"text": formatted_text}
        
    formatted_dataset = dataset.map(format_example)
    return formatted_dataset

if __name__ == "__main__":
    # Test script locally
    try:
        ds = prepare_cuda_agent_dataset("BytedTsinghua-SIA/CUDA-Agent-Ops-6K", "train")
        print("\n=== Sample prompt (first 500 chars) ===")
        print(ds[0]['text'][:500])
        print(f"\n=== Total length: {len(ds[0]['text'])} chars ===")
    except Exception as e:
        print(f"Error: {e}")
