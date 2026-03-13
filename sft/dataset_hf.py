import os
from datasets import load_dataset
from transformers import AutoTokenizer

# A richer system prompt informed by CUDA Agent's approach
SYSTEM_PROMPT = """<|im_start|>system
You are an expert GPU kernel developer and optimizer. Your goal is to rewrite PyTorch operations into high-performance, optimized CUDA C++ code using appropriate kernel launch parameters, thread block dimensions, shared memory, and memory coalescing techniques.

When given a PyTorch operator implementation (model.py) and a description of the operation (ops), you must output the equivalent, heavily optimized CUDA C++ implementation (model_new.py).
<|im_end|>
"""

def format_cuda_prompt(ops_description, model_py, model_new_py=None):
    """
    Format the prompt using the Qwen chat template. 
    Input is the baseline PyTorch and operation string. 
    Target is the optimized CUDA code.
    """
    prompt = SYSTEM_PROMPT
    prompt += f"<|im_start|>user\nOperation description:\n{ops_description}\n\nPyTorch implementation (model.py):\n```python\n{model_py}\n```<|im_end|>\n"
    prompt += f"<|im_start|>assistant\n```cpp\n"
    if model_new_py:
        prompt += f"{model_new_py}\n```<|im_end|>\n"
    return prompt

def prepare_cuda_agent_dataset(dataset_name="BytedTsinghua-SIA/CUDA-Agent-Ops-6K", split="train"):
    """
    Loads and formats the CUDA Agent-Ops-6K dataset.
    Columns are: 'ops', 'model.py', 'model_new.py'
    """
    print(f"Loading dataset: {dataset_name} ({split})")
    
    try:
        dataset = load_dataset(dataset_name, split=split)
    except Exception as e:
        print(f"Failed to load dataset from HF: {e}")
        print("Please ensure the dataset name is correct or login with `huggingface-cli login`.")
        raise
        
    def format_example(example):
        ops_desc = example.get("ops", "")
        model_py = example.get("model.py", "")
        model_new_py = example.get("model_new.py", "")
        
        formatted_text = format_cuda_prompt(ops_desc, model_py, model_new_py)
        return {"text": formatted_text}
        
    # We may want to filter out extremely long examples during mapping, 
    # but the trainer's max_seq_length logic will handle truncation.
    formatted_dataset = dataset.map(format_example)
    return formatted_dataset

if __name__ == "__main__":
    # Test script locally
    try:
        ds = prepare_cuda_agent_dataset("BytedTsinghua-SIA/CUDA-Agent-Ops-6K", "train")

        print("Sample:", ds[0]['text'])
    except Exception as e:
        pass
