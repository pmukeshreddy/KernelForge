"""
train_ppo.py - Stage 3 of the KernelForge Pipeline: Multi-Turn Agentic PPO.

Each RL episode is a full ReAct loop that mirrors agent.py exactly:
  - Model outputs raw CUDA C++ (prefill: "```cpp\\n#include <torch/extension.h>\\n")
  - extract_cuda_code() finds the ```cpp block
  - build_load_inline_wrapper() wraps it in Python for the sandbox
  - Sandbox compiles, checks correctness, times execution
  - Profiler (ncu) reports the hardware bottleneck
  - Feedback is returned to the model for the next turn
  - Final reward = best speedup (baseline_ms / kernel_ms) across all turns

All agent-generated tokens across turns are concatenated into a single
response tensor. Environment tokens (profiler text, error messages) are
context-only — they never enter the response tensor or the loss.

Algorithm: PPO with GAE via TRL PPOTrainer + AutoModelForCausalLMWithValueHead.
  - The value head provides per-token baselines for advantage estimation (GAE).
  - A frozen reference model enforces the KL penalty.
  - Uses PPO clipped objective with adaptive KL coefficient.

Why PPO over GRPO:
  - GRPO uses group-normalized rewards as the baseline (no critic).
  - PPO has a learned value function — better credit assignment for
    sparse rewards in multi-turn trajectories (arXiv:2510.01132 shows
    PPO significantly outperforms GRPO/RLOO on multi-turn sparse tasks).
"""

import re
from dataclasses import dataclass

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

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
    output_dir: str = "checkpoints/kernelforge_ppo"

    # Episode settings
    max_react_steps: int = 2       # turns per episode: generate → profiler → improve

    # PPO hyperparameters
    batch_size: int = 4            # prompts collected before each PPO update
    mini_batch_size: int = 1       # mini-batch size inside PPO update
    ppo_epochs: int = 4            # PPO update passes over each collected batch
    learning_rate: float = 5e-7
    init_kl_coef: float = 0.01     # initial KL penalty coefficient (adaptive)
    target_kl: float = 6.0         # adaptive KL target — keeps model near SFT
    gamma: float = 1.0             # discount factor (1.0 = no decay, reward at end)
    lam: float = 0.95              # GAE lambda
    cliprange: float = 0.2         # PPO policy clip epsilon
    cliprange_value: float = 0.2   # value function clip epsilon
    vf_coef: float = 0.1           # weight of value function loss

    # Generation settings
    max_new_tokens: int = 2048
    temperature: float = 0.7


# ---------------------------------------------------------------------------
# Format helpers — must match agent.py exactly
# ---------------------------------------------------------------------------

# This prefill matches agent.py line 115
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
    # Fallback: strip <think> blocks, check if remainder looks like CUDA C++
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    clean = re.sub(r"<think>.*", "", clean, flags=re.DOTALL)
    stripped = clean.strip()
    if any(kw in stripped for kw in ["__global__", "torch::Tensor", "#include"]):
        return stripped
    return ""


# ---------------------------------------------------------------------------
# Generation — one turn of the ReAct loop
# ---------------------------------------------------------------------------

def _generate_turn(
    trainer: PPOTrainer,
    tokenizer: AutoTokenizer,
    messages: list[dict],
    config: KernelForgeConfig,
) -> tuple[torch.Tensor, str]:
    """
    Run one generation turn using trainer.generate() for correct device handling.

    Returns:
        generated_ids — 1D CPU tensor of agent-generated token IDs only.
        decoded_text  — full decoded response string (includes PREFILL).
    """
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    text += PREFILL

    input_ids = tokenizer.encode(text, return_tensors="pt")

    # trainer.generate() handles device placement and works with the value-head model
    output_ids = trainer.generate(
        input_ids,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )[0]  # returns list of tensors; [0] for single input

    prompt_len = input_ids.shape[1]
    generated_ids = output_ids[prompt_len:].cpu()
    decoded = PREFILL + tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_ids, decoded


# ---------------------------------------------------------------------------
# Episode — one full ReAct loop = one PPO trajectory
# ---------------------------------------------------------------------------

def _run_react_episode(
    prompt_text: str,
    trainer: PPOTrainer,
    tokenizer: AutoTokenizer,
    config: KernelForgeConfig,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Run one full ReAct episode.

    The format is identical to agent.py (KernelForgeAgent.run_react_loop):
      user   → "write an optimized CUDA C++ kernel for this op"
      agent  → generates raw C++ (prefill forces ```cpp format)
      [wrap] → build_load_inline_wrapper() builds the Python/sandbox script
      [eval] → sandbox compiles, checks correctness, times it
      user   → profiler feedback OR error message
      agent  → generates improved C++ kernel
      ...

    Only agent-generated tokens go into response_tensor.
    Environment tokens (errors, profiler) stay in messages for context only.

    Returns:
        query_tensor    — 1D tensor: initial prompt token IDs (for PPOTrainer.step)
        response_tensor — 1D tensor: all agent turns concatenated (for PPOTrainer.step)
        reward          — float: best speedup across all turns
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

    # Encode the initial prompt for PPOTrainer.step()
    initial_text = (
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        + PREFILL
    )
    query_tensor = tokenizer.encode(initial_text, return_tensors="pt")[0]

    all_response_ids: list[int] = []
    best_reward = 0.0

    for step in range(config.max_react_steps):
        print(f"\n--- Step {step + 1}/{config.max_react_steps} ---")
        print("🧠 Generating Kernel...")

        generated_ids, response_text = _generate_turn(trainer, tokenizer, messages, config)

        # Accumulate ONLY agent-generated tokens — never environment tokens
        all_response_ids.extend(generated_ids.tolist())

        messages.append({"role": "assistant", "content": response_text})

        # --- Same flow as agent.py ---

        # Step 1: Extract CUDA C++ from the ```cpp block
        cuda_code = extract_cuda_code(response_text)
        if not cuda_code:
            print("❌ No ```cpp block found.")
            messages.append({
                "role": "user",
                "content": (
                    "Error: Could not find ```cpp block in your response. "
                    "Output the CUDA C++ code in a ```cpp code block."
                ),
            })
            continue

        # Step 2: Wrap raw C++ in load_inline Python script
        candidate_code = build_load_inline_wrapper(cuda_code, prompt_text)
        if not candidate_code:
            print("❌ Could not parse torch::Tensor binding function.")
            messages.append({
                "role": "user",
                "content": (
                    "Error: Could not find a `torch::Tensor` binding function. "
                    "You must include a C++ function that returns `torch::Tensor` "
                    "and calls your CUDA kernel."
                ),
            })
            continue

        # Step 3: Sandbox evaluation (compile, correctness, timing)
        print("🛠️  Compiling and Evaluating in Sandbox...")
        eval_result = evaluate(candidate_code, prompt_text)

        if not eval_result["correct"]:
            error = eval_result.get("compiler_error", "Outputs do not match reference.")
            print(f"❌ Evaluation Failed: {error.strip()[:120]}...")
            messages.append({
                "role": "user",
                "content": (
                    f"Your CUDA C++ code failed.\n\nError Log:\n```\n{error[:500]}\n```\n\n"
                    "Analyze the root cause carefully. Fix the bug and output the corrected C++ code."
                ),
            })
            continue

        # Step 4: Reward + profiler feedback
        reward = calculate_reward(eval_result)
        best_reward = max(best_reward, reward)
        runtime_ms = eval_result["runtime_ms"]
        print(f"✅ Sandbox Passed! Latency: {runtime_ms:.3f} ms | Reward: {reward:.2f}x")

        if step < config.max_react_steps - 1:
            print("🔬 Profiling Hardware Metrics...")
            profiler_feedback = profile_kernel(candidate_code, prompt_text)
            messages.append({
                "role": "user",
                "content": (
                    f"Success! Your kernel ran in {runtime_ms:.3f} ms "
                    f"({reward:.2f}x speedup over baseline).\n\n"
                    f"Hardware Profiler Report:\n{profiler_feedback}\n\n"
                    "Can you optimize the C++ kernel further? Output the improved ```cpp code."
                ),
            })

    print(f"\n🏁 Episode Complete. Best Reward: {best_reward:.2f}x")

    # If model generated nothing at all (should not happen), create a dummy token
    if not all_response_ids:
        all_response_ids = [tokenizer.eos_token_id]

    response_tensor = torch.tensor(all_response_ids, dtype=torch.long)
    return query_tensor, response_tensor, best_reward


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config: KernelForgeConfig = None):
    if config is None:
        config = KernelForgeConfig()

    print(f"Loading dataset from {config.rft_dataset_path}...")
    try:
        raw_dataset = load_dataset("json", data_files=config.rft_dataset_path, split="train")
    except Exception as e:
        print(f"Failed to load dataset. Run train_rft.py first.\n{e}")
        return
    print(f"Loaded {len(raw_dataset)} prompts.")

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Policy model with value head (PPO critic learns V(s) for GAE)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Frozen reference model for KL penalty
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    ppo_config = PPOConfig(
        model_name=config.model_id,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        mini_batch_size=config.mini_batch_size,
        ppo_epochs=config.ppo_epochs,
        kl_penalty="kl",
        init_kl_coef=config.init_kl_coef,
        target=config.target_kl,
        gamma=config.gamma,
        lam=config.lam,
        cliprange=config.cliprange,
        cliprange_value=config.cliprange_value,
        vf_coef=config.vf_coef,
        log_with=None,
    )

    trainer = PPOTrainer(ppo_config, model, ref_model, tokenizer)

    dataloader = DataLoader(
        raw_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # avoid multiprocessing conflicts with sandbox subprocesses
    )

    print(
        f"🚀 Starting PPO training "
        f"(max_react_steps={config.max_react_steps}, batch_size={config.batch_size})..."
    )

    for batch in tqdm(dataloader, desc="PPO Batches"):
        prompts = batch["prompt"]

        query_tensors = []
        response_tensors = []
        rewards = []

        for prompt_text in prompts:
            q, r, reward = _run_react_episode(prompt_text, trainer, tokenizer, config)
            query_tensors.append(q)
            response_tensors.append(r)
            rewards.append(torch.tensor(reward, dtype=torch.float))

        # PPO step: computes log probs (policy + ref), value estimates,
        # GAE advantages, clipped policy loss + value loss, KL penalty
        stats = trainer.step(query_tensors, response_tensors, rewards)
        trainer.log_stats(stats, {"query": prompts}, rewards)

    trainer.save_pretrained(config.output_dir)
    print(f"Model saved to {config.output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="KernelForge Multi-Turn PPO Training")
    parser.add_argument("--dataset", type=str, default="data/rft_dataset.json")
    parser.add_argument("--output_dir", type=str, default="checkpoints/kernelforge_ppo")
    parser.add_argument("--max_react_steps", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--init_kl_coef", type=float, default=0.01)
    args = parser.parse_args()

    config = KernelForgeConfig(
        rft_dataset_path=args.dataset,
        output_dir=args.output_dir,
        max_react_steps=args.max_react_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        init_kl_coef=args.init_kl_coef,
    )
    train(config)
