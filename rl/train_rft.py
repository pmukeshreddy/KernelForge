"""
train_rft.py - Stage 1 of the RL Pipeline: Rejection Fine-Tuning.

This script uses the autonomous ReAct agent (SP6) to generate multiple candidate
CUDA kernels for each PyTorch operation in the dataset. It then filters out any 
kernel that fails to compile, gets the wrong answer, or does not achieve a 
performance speedup. The remaining successful trajectories are saved to disk
so the model can be fine-tuned on them, locking in formatting constraints 
before Phase 2 (Full RL).
"""

import json
import random
import os
from agent import KernelForgeAgent

NUM_SAMPLES_PER_PROMPT = 8
MAX_REACT_STEPS = 2
MIN_SPEEDUP_REWARD = 1.05

def load_dataset(filepath: str) -> list:
    """Loads a JSON or JSONL dataset of PyTorch reference implementations."""
    with open(filepath, 'r') as f:
        if filepath.endswith('.jsonl'):
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)

def run_rft_collection(dataset_path: str, output_path: str):
    """Samples the LLM, filters bad generations, and saves the RFT dataset."""
    print(f"Initializing Rejection Fine-Tuning (RFT) Sampling...")
    agent = KernelForgeAgent(model_name="Qwen/Qwen2.5-Coder-3B-Instruct")
    
    # Load raw PyTorch -> CUDA prompts (SP1/SFT base)
    dataset = load_dataset(dataset_path)
    print(f"Loaded {len(dataset)} prompts.")
    
    rft_dataset = []
    
    for i, data in enumerate(dataset):
        prompt = data["pytorch_code"]
        print(f"\n[{i+1}/{len(dataset)}] Generating {NUM_SAMPLES_PER_PROMPT} candidates for target...")
        
        candidates_passed = 0
        for fallback_attempt in range(NUM_SAMPLES_PER_PROMPT):
            print(f"  Attempt {fallback_attempt + 1}/{NUM_SAMPLES_PER_PROMPT}...")
            # Run the autonomous ReAct loop
            optimized_code, reward = agent.run_react_loop(prompt, max_steps=MAX_REACT_STEPS)
            
            # Rejection Sampling: We only keep kernels that achieved a real speedup
            if reward >= MIN_SPEEDUP_REWARD:
                rft_dataset.append({
                    "prompt": prompt,
                    "optimized_cuda": optimized_code,
                    "reward": reward
                })
                candidates_passed += 1
                print(f"    ✅ Accepted! (Reward: {reward:.2f}x)")
            else:
                print(f"    ❌ Rejected. (Reward: {reward:.2f}x)")
        
        print(f"  Saved {candidates_passed} verified trajectories for this prompt.")
        
    print(f"\n🏁 RFT Sampling Complete. Saved {len(rft_dataset)} verified high-performance kernels.")
    
    # Save the filtered Dataset
    with open(output_path, 'w') as f:
        json.dump(rft_dataset, f, indent=4)
        print(f"Dataset written to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run KernelForge RFT Collection")
    parser.add_argument("--dataset", type=str, required=True, help="Path to input PyTorch dataset (JSON)")
    parser.add_argument("--output", type=str, default="data/rft_dataset.json", help="Path to save generated RFT dataset")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    run_rft_collection(args.dataset, args.output)
