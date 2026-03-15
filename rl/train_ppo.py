"""
train_ppo.py - Stage 3 of the KernelForge Pipeline: Multi-Turn Agentic RL via GRPO.

NOTE: TRL 0.29.0 removed PPOTrainer entirely. We use GRPOTrainer with a custom
rollout_func — the TRL-supported path for multi-turn agentic RL.

GRPO vs PPO difference: GRPO uses group-normalized rewards as the baseline
(no critic/value head needed). PPOTrainer no longer exists in TRL 0.29.0.

Each RL episode is a full ReAct loop that mirrors agent.py exactly:
  - prefill: "```cpp\\n#include <torch/extension.h>\\n"  (matches SFT training format)
  - model outputs raw CUDA C++
  - extract_cuda_code() finds the ```cpp block
  - build_load_inline_wrapper() wraps it in Python for the sandbox
  - sandbox compiles, checks correctness, times execution
  - profiler reports hardware bottleneck
  - feedback returned for next turn
  - final reward = best speedup across all turns

Only agent-generated tokens go into completion_ids/logprobs.
Environment tokens (errors, profiler text) are context-only — masked from loss.

Reference: TRL GRPOTrainer rollout_func API (huggingface.co/docs/trl/main/en/openenv)
           "A Practitioner's Guide to Multi-Turn Agentic RL" (arXiv:2510.01132)
"""

import re
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

from agent import build_load_inline_wrapper
from profiler import profile_kernel
from reward import calculate_reward
from sandbox import evaluate
from sys_prompt import get_system_prompt


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class KernelForgeConfig:
    model_id: str = "mukeshreddy/kernelforge-sft-qwen3-8b"
    rft_dataset_path: str = "data/rft_dataset.json"
    output_dir: str = "checkpoints/kernelforge_grpo"

    # Episode settings
    max_react_steps: int = 2

    # GRPO settings
    num_generations: int = 4
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_completion_length: int = 4096
    learning_rate: float = 5e-7
    beta: float = 0.01
    num_train_epochs: int = 1
    logging_steps: int = 1
    save_steps: int = 50


# ---------------------------------------------------------------------------
# Format helpers — must match agent.py exactly
# ---------------------------------------------------------------------------

PREFILL = "```cpp\n#include <torch/extension.h>\n"


def extract_cuda_code(text: str) -> str:
    """
    Extract CUDA C++ from model response.
    Mirrors KernelForgeAgent.extract_cuda_code() in agent.py exactly.
    """
    for lang in ["cpp", "cuda", r"c\+\+"]:
        match = re.search(rf"```{lang}(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    clean = re.sub(r"<think>.*", "", clean, flags=re.DOTALL)
    stripped = clean.strip()
    if any(kw in stripped for kw in ["__global__", "torch::Tensor", "#include"]):
        return stripped
    return ""


# ---------------------------------------------------------------------------
# Generation — one turn
# ---------------------------------------------------------------------------

def _generate_turn(
    model,
    tokenizer,
    messages: list[dict],
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
) -> tuple[list[int], list[float], str]:
    """
    Run one generation turn. Returns:
        generated_ids  — agent-generated token IDs only (no prompt)
        token_logprobs — log π_θ(token | context) per generated token
        decoded_text   — full response string including prefill
    """
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    text += PREFILL

    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

    generated_ids = outputs.sequences[0][prompt_len:].tolist()

    # Log probs from raw logits — true π_θ, not temperature-scaled distribution
    token_logprobs: list[float] = []
    for idx, score in enumerate(outputs.scores):
        lp = F.log_softmax(score[0], dim=-1)
        token_logprobs.append(lp[generated_ids[idx]].item())

    decoded = PREFILL + tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_ids, token_logprobs, decoded


# ---------------------------------------------------------------------------
# Episode — one full ReAct loop = one RL trajectory
# ---------------------------------------------------------------------------

def _run_react_episode(
    prompt_text: str,
    model,
    tokenizer,
    max_steps: int = 2,
) -> tuple[list[int], list[int], list[float], float]:
    """
    Run one full ReAct episode using the same format as agent.py.

    Returns:
        prompt_ids     — token IDs of the initial prompt
        completion_ids — all agent-generated token IDs concatenated across turns
        logprobs       — per-token log probs for completion_ids
        reward         — best speedup achieved across all turns
    """
    system_prompt = get_system_prompt()

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Write an optimized CUDA C++ kernel to replace this PyTorch implementation. "
                "Output only the C++ code.\n\n"
                f"Reference Program:\n```python\n{prompt_text}\n```"
            ),
        },
    ]

    initial_text = (
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        + PREFILL
    )
    prompt_ids = tokenizer([initial_text], return_tensors="pt").input_ids[0].tolist()

    all_completion_ids: list[int] = []
    all_logprobs: list[float] = []
    best_reward = 0.0

    for step in range(max_steps):
        print(f"\n--- Step {step + 1}/{max_steps} ---")
        print("🧠 Generating Kernel...")

        generated_ids, token_logprobs, response_text = _generate_turn(
            model, tokenizer, messages
        )

        all_completion_ids.extend(generated_ids)
        all_logprobs.extend(token_logprobs)
        messages.append({"role": "assistant", "content": response_text})

        # Step 1: Extract CUDA C++ (same as agent.py)
        cuda_code = extract_cuda_code(response_text)
        if not cuda_code:
            print("❌ No ```cpp block found.")
            messages.append({
                "role": "user",
                "content": "Error: No ```cpp block found. Output CUDA C++ in a ```cpp code block.",
            })
            continue

        # Step 2: Wrap in load_inline (same as agent.py)
        candidate_code = build_load_inline_wrapper(cuda_code, prompt_text)
        if not candidate_code:
            print("❌ No torch::Tensor binding function found.")
            messages.append({
                "role": "user",
                "content": "Error: No `torch::Tensor` binding function found. Add a C++ function returning `torch::Tensor`.",
            })
            continue

        # Step 3: Sandbox evaluation
        print("🛠️  Compiling and Evaluating in Sandbox...")
        eval_result = evaluate(candidate_code, prompt_text)

        if not eval_result["correct"]:
            error = eval_result.get("compiler_error", "Outputs do not match reference.")
            print(f"❌ Failed: {error.strip()[:120]}...")
            messages.append({
                "role": "user",
                "content": (
                    f"Your kernel failed.\n\nError:\n```\n{error[:500]}\n```\n\n"
                    "Fix the bug and output the corrected C++ code."
                ),
            })
            continue

        # Step 4: Reward + profiler feedback
        reward = calculate_reward(eval_result)
        best_reward = max(best_reward, reward)
        runtime_ms = eval_result["runtime_ms"]
        print(f"✅ Passed! {runtime_ms:.3f} ms | {reward:.2f}x speedup")

        if step < max_steps - 1:
            print("🔬 Profiling...")
            profiler_feedback = profile_kernel(candidate_code, prompt_text)
            messages.append({
                "role": "user",
                "content": (
                    f"Success! {runtime_ms:.3f} ms ({reward:.2f}x speedup).\n\n"
                    f"Profiler:\n{profiler_feedback}\n\n"
                    "Optimize further. Output improved ```cpp code."
                ),
            })

    print(f"\n🏁 Episode done. Best reward: {best_reward:.2f}x")
    return prompt_ids, all_completion_ids, all_logprobs, best_reward


# ---------------------------------------------------------------------------
# rollout_func — plugs into GRPOTrainer (TRL 0.29.0 experimental API)
# ---------------------------------------------------------------------------

def make_rollout_func(max_react_steps: int):
    """
    Returns a rollout_func compatible with TRL 0.29.0 GRPOTrainer.

    Signature: fn(prompts: list[str], trainer: GRPOTrainer) -> dict[str, list]

    Required keys: prompt_ids, completion_ids, logprobs
    Extra keys forwarded as kwargs to reward_funcs: reward
    """
    def rollout_func(prompts: list[str], trainer) -> dict[str, Any]:
        model = trainer.model
        tokenizer = trainer.processing_class

        prompt_ids_batch: list[list[int]] = []
        completion_ids_batch: list[list[int]] = []
        logprobs_batch: list[list[float]] = []
        rewards_batch: list[float] = []

        for prompt_text in prompts:
            p_ids, c_ids, lps, reward = _run_react_episode(
                prompt_text, model, tokenizer, max_steps=max_react_steps
            )
            prompt_ids_batch.append(p_ids)
            completion_ids_batch.append(c_ids)
            logprobs_batch.append(lps)
            rewards_batch.append(reward)

        return {
            "prompt_ids": prompt_ids_batch,
            "completion_ids": completion_ids_batch,
            "logprobs": logprobs_batch,
            "reward": rewards_batch,
        }

    return rollout_func


def reward_from_rollout(completions, reward, **kwargs) -> list[float]:
    """Forward pre-computed sandbox reward to GRPOTrainer."""
    return list(reward)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config: KernelForgeConfig = None):
    if config is None:
        config = KernelForgeConfig()

    print(f"Loading dataset from {config.rft_dataset_path}...")
    try:
        raw_dataset = load_dataset("json", data_files=config.rft_dataset_path, split="train")
        # Keep only entries that have a pytorch_code prompt
        raw_dataset = raw_dataset.filter(lambda x: bool(x.get("pytorch_code", "").strip()))
    except Exception as e:
        print(f"Failed to load dataset.\n{e}")
        return
    print(f"Loaded {len(raw_dataset)} prompts.")

    training_args = GRPOConfig(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_generations=config.num_generations,
        max_completion_length=config.max_completion_length,
        learning_rate=config.learning_rate,
        beta=config.beta,
        num_train_epochs=config.num_train_epochs,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
    )

    rollout_func = make_rollout_func(max_react_steps=config.max_react_steps)

    print(f"Initializing GRPOTrainer (max_react_steps={config.max_react_steps}, "
          f"num_generations={config.num_generations})...")

    trainer = GRPOTrainer(
        model=config.model_id,
        reward_funcs=[reward_from_rollout],
        args=training_args,
        train_dataset=raw_dataset,
        rollout_func=rollout_func,
    )

    print("🚀 Launching multi-turn agentic RL training...")
    trainer.train()
    trainer.save_model(config.output_dir)
    print(f"Model saved to {config.output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="KernelForge Multi-Turn GRPO Training")
    parser.add_argument("--dataset", type=str, default="../sft/sft_training_pairs.jsonl")
    parser.add_argument("--output_dir", type=str, default="checkpoints/kernelforge_grpo")
    parser.add_argument("--model", type=str, default="mukeshreddy/kernelforge-sft-qwen3-8b")
    parser.add_argument("--max_react_steps", type=int, default=2)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--beta", type=float, default=0.01)
    args = parser.parse_args()

    config = KernelForgeConfig(
        model_id=args.model,
        rft_dataset_path=args.dataset,
        output_dir=args.output_dir,
        max_react_steps=args.max_react_steps,
        num_generations=args.num_generations,
        learning_rate=args.lr,
        beta=args.beta,
    )
    train(config)
