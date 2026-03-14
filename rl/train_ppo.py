"""
train_ppo.py - Stage 2 of the RL Pipeline: Policy Optimization (GRPO/DAPO)

This script executes the full Reinforcement Learning phase. 
Once the model has "locked in" the format via RFT (Stage 1), we use PPO/GRPO
to explore the optimization space and strictly maximize hardware rewards
calculated by the Nsight Compute profiler.

It uses a HuggingFace generic GRPO implementation optimized for code-generation.
"""

from dataclasses import dataclass
import json
import torch
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from agent import KernelForgeAgent

@dataclass
class KernelForgeConfig:
    model_id: str = "Qwen/Qwen2.5-Coder-3B-Instruct"
    rft_dataset_path: str = "data/rft_dataset.json" # Baseline prompts
    output_dir: str = "checkpoints/kernelforge_ppo"
    batch_size: int = 4
    num_generations: int = 4 # N answers per input for GRPO
    max_prompt_length: int = 512
    max_completion_length: int = 1536
    learning_rate: float = 1e-6 # Very low LR to prevent format collapse

def kernel_reward_func(prompts, completions, **kwargs) -> list[float]:
    """
    The actual Reward Function invoked by the RL Trainer.
    We pass the generated completion to our verified Sandbox (SP1-3).
    If it compiles and is correct, we pass it to the Profiler (SP4) for Speedup.
    """
    import os
    from sandbox import evaluate
    from reward import calculate_reward
    
    rewards = []
    
    for prompt, completion in zip(prompts, completions):
        # The prompt is just the PyTorch operation.
        target_program = prompt
        
        # The completion is the LLM's `load_inline` python code block
        candidate_code = KernelForgeAgent().extract_code_block(completion)
        
        if not candidate_code:
            # Format Collapse Penalty
            rewards.append(-1.0)
            continue
            
        print(f"🛠️ Evaluating generation on Sandbox...")
        eval_result = evaluate(candidate_code, target_program)
        
        if not eval_result["correct"]:
            # Syntax / Logic Error Penalty
            rewards.append(-0.5)
            print(f"❌ Failed: Penalty -0.5")
            continue
            
        # Success! Calculate true hardware speedup mapping
        reward = calculate_reward(eval_result, target_program)
        print(f"🏆 Speedup Achieved: {reward:.2f}x")
        rewards.append(reward)
        
    return rewards

def train():
    config = KernelForgeConfig()
    print(f"Loading Model for GRPO: {config.model_id}")
    
    # Enable deepspeed / FSDP depending on server hardware
    training_args = GRPOConfig(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        num_generations=config.num_generations,
        max_prompt_length=config.max_prompt_length,
        max_completion_length=config.max_completion_length,
        learning_rate=config.learning_rate,
        beta=0.01, # KL penalty margin - keep small so it doesn't penalize long CUDA code too hard
        logging_steps=1,
    )
    
    # Load the RFT Dataset. We only extract the 'prompt' (PyTorch baseline)
    print(f"Loading Base Prompts from {config.rft_dataset_path}...")
    try:
        raw_dataset = load_dataset('json', data_files=config.rft_dataset_path, split="train")
    except Exception as e:
        print(f"Error loading RFT base dataset. Please run train_rft.py first.\n{e}")
        return
        
    # The GRPO Trainer expects columns `prompt` and `completion`
    
    print("Setting up GRPO Trainer (Reward Sandbox Attached)...")
    trainer = GRPOTrainer(
        model=config.model_id,
        reward_funcs=[kernel_reward_func],
        args=training_args,
        train_dataset=raw_dataset,
    )
    
    print("🚀 Launching Hardware-in-the-Loop PPO Training...")
    trainer.train()

if __name__ == "__main__":
    train()
