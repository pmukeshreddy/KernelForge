import os
from datasets import load_dataset
from transformers import AutoTokenizer

def format_cuda_prompt(instruction, kernel=None):
    """
    Given an instruction (and optionally the kernel), format the prompt 
    in the chat template expected by Qwen3 (or a standard instruction format).
    """
    prompt = f"<|im_start|>system\nYou are an expert GPU programmer. Write a correct and optimized CUDA kernel for the requested operation.<|im_end|>\n"
    prompt += f"<|im_start|>user\n{instruction}<|im_end|>\n"
    prompt += f"<|im_start|>assistant\n```cpp\n"
    if kernel:
        prompt += f"{kernel}\n```<|im_end|>\n"
    return prompt

def prepare_cuda_agent_dataset(dataset_name="UCSC-VLAA/cuda-agent-sft", split="train"):
    """
    Loads and formats the CUDA Agent 6K operations dataset for SFT.
    Users should replace the dataset_name with the actual huggingface repo ID.
    """
    print(f"Loading dataset: {dataset_name} ({split})")
    
    # We use a placeholder local path if HF repo isn't public, otherwise load directly.
    # For now, we attempt to load from HF.
    try:
        dataset = load_dataset(dataset_name, split=split)
    except Exception as e:
        print(f"Failed to load dataset from HF: {e}")
        print("Please ensure the dataset name is correct or login with `huggingface-cli login`.")
        raise
        
    def format_example(example):
        instruction = example.get("instruction", example.get("prompt", ""))
        kernel = example.get("kernel", example.get("completion", ""))
        
        formatted_text = format_cuda_prompt(instruction, kernel)
        return {"text": formatted_text}
        
    formatted_dataset = dataset.map(format_example)
    return formatted_dataset

if __name__ == "__main__":
    # Test script locally
    try:
        ds = prepare_cuda_agent_dataset("Salesforce/wikitext", "train") # Replace with actual 6K ops dataset when ready
        print("Sample:", ds[0]['text'])
    except Exception as e:
        pass
