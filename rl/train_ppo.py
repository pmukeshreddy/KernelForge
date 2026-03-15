"""
train_ppo.py - Stage 3 of the KernelForge Pipeline: Multi-Turn Agentic RL via GRPO.

Each RL episode is a full ReAct loop:
  Turn 1: Model generates CUDA kernel → Sandbox compiles + checks correctness → Profiler runs
  Turn 2: Model sees profiler feedback → Generates improved kernel → Sandbox re-evaluates
  ...
  Final reward: best speedup (baseline_ms / kernel_ms) achieved across all turns.

The full multi-turn trajectory is treated as a single flat token sequence for GRPO.
Only agent-generated tokens contribute to the loss. Environment tokens (profiler
feedback, error messages) stay in the conversation context but are masked out.

Algorithm: GRPO with rollout_func (TRL experimental multi-turn API).
  - num_generations rollouts are collected per prompt.
  - Rewards are normalized within the group (zero-mean, unit-variance).
  - KL penalty against the reference model prevents policy collapse.

Reference:
  "A Practitioner's Guide to Multi-Turn Agentic RL" (arXiv:2510.01132)
  TRL GRPOTrainer rollout_func API: huggingface.co/docs/trl/main/en/openenv
"""

import re
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

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
    max_react_steps: int = 2          # turns per episode (generate → profiler → improve)

    # GRPO settings
    num_generations: int = 4          # G: parallel rollouts per prompt for group normalization
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_prompt_length: int = 512
    max_completion_length: int = 4096  # long enough to cover concatenated multi-turn completions
    learning_rate: float = 5e-7
    beta: float = 0.01                # KL penalty — small so long CUDA completions aren't over-penalized
    num_train_epochs: int = 1
    logging_steps: int = 1
    save_steps: int = 50


# ---------------------------------------------------------------------------
# Generation utilities
# ---------------------------------------------------------------------------

def _extract_code_block(text: str) -> str:
    """Extract the Python code block from model output."""
    match = re.search(r"```python(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: strip <think> blocks and return the rest
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    clean = re.sub(r"<think>.*", "", clean, flags=re.DOTALL)
    return clean.strip()


def _generate_turn(
    model,
    tokenizer,
    messages: list[dict],
    prefill: str,
) -> tuple[list[int], list[float], str]:
    """
    Run one generation turn.

    Args:
        model: The policy model (trainer.model).
        tokenizer: The tokenizer (trainer.processing_class).
        messages: Current conversation history (list of role/content dicts).
        prefill: Text to force-prepend to the assistant response.

    Returns:
        generated_ids  — token IDs for the agent's response (not the prompt).
        token_logprobs — log π_θ(token_i | context) for each generated token.
        decoded_text   — full decoded response string including prefill.
    """
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    text += prefill

    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,   # one raw logit tensor per generated token
        )

    generated_ids = outputs.sequences[0][prompt_len:].tolist()

    # Compute log π_θ from raw logits (temperature NOT applied — we want true model probs)
    token_logprobs: list[float] = []
    for idx, score in enumerate(outputs.scores):
        lp = F.log_softmax(score[0], dim=-1)
        token_logprobs.append(lp[generated_ids[idx]].item())

    decoded = prefill + tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_ids, token_logprobs, decoded


# ---------------------------------------------------------------------------
# Episode (one full ReAct loop = one RL trajectory)
# ---------------------------------------------------------------------------

def _run_react_episode(
    prompt_text: str,
    model,
    tokenizer,
    max_steps: int = 2,
) -> tuple[list[int], list[int], list[float], float]:
    """
    Run one full ReAct episode.

    The episode is a multi-turn conversation:
      user   → "write a kernel for this PyTorch op"
      agent  → generates kernel (Turn 1)
      user   → profiler feedback OR error (environment observation)
      agent  → generates improved kernel (Turn 2)
      ...

    Only agent-generated tokens are collected in completion_ids / logprobs.
    Environment observations are added to messages for context but masked out.

    Returns:
        prompt_ids     — token IDs of the initial user prompt.
        completion_ids — all agent-generated token IDs concatenated across turns.
        logprobs       — per-token log probs for completion_ids.
        reward         — best speedup (≥1.0 = faster than baseline, 0.0 = failed).
    """
    prefill = "```python\nimport torch\n"
    system_prompt = get_system_prompt()

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Please write a highly optimized custom CUDA kernel to replace "
                "this PyTorch reference implementation. Output the complete "
                "load_inline build script.\n\n"
                f"Reference Program:\n```python\n{prompt_text}\n```"
            ),
        },
    ]

    # Tokenize the initial prompt — used as prompt_ids for TRL loss masking
    initial_text = (
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        + prefill
    )
    prompt_ids = tokenizer([initial_text], return_tensors="pt").input_ids[0].tolist()

    all_completion_ids: list[int] = []
    all_logprobs: list[float] = []
    best_reward = 0.0

    for step in range(max_steps):
        print(f"\n--- Step {step + 1}/{max_steps} ---")
        print("🧠 Generating Kernel...")

        generated_ids, token_logprobs, response_text = _generate_turn(
            model, tokenizer, messages, prefill
        )

        # Accumulate ONLY agent tokens — environment tokens are never added here
        all_completion_ids.extend(generated_ids)
        all_logprobs.extend(token_logprobs)

        # Add agent response to conversation history
        messages.append({"role": "assistant", "content": response_text})

        # Extract the load_inline Python script from the response
        candidate_code = _extract_code_block(response_text)
        if not candidate_code:
            print("❌ No ```python block found. Requesting fix...")
            messages.append({
                "role": "user",
                "content": (
                    "Error: Could not find a ```python block in your response. "
                    "Output the complete load_inline script inside a single ```python block."
                ),
            })
            continue

        # Evaluate in sandbox (compile, correctness, timing)
        print("🛠️  Compiling and Evaluating in Sandbox...")
        eval_result = evaluate(candidate_code, prompt_text)

        if not eval_result["correct"]:
            error = eval_result.get("compiler_error", "Outputs do not match the reference.")
            print(f"❌ Evaluation Failed: {error.strip()[:120]}...")
            messages.append({
                "role": "user",
                "content": (
                    f"Your kernel failed evaluation.\n\nError Log:\n```\n{error[:500]}\n```"
                    "\n\nPlease fix the errors and rewrite the complete kernel."
                ),
            })
            continue

        # Kernel passed — compute reward and prepare profiler feedback for next turn
        reward = calculate_reward(eval_result)
        best_reward = max(best_reward, reward)
        runtime_ms = eval_result["runtime_ms"]
        print(f"✅ Sandbox Passed! Latency: {runtime_ms:.3f} ms | Reward: {reward:.2f}x")

        if step < max_steps - 1:
            print("🔬 Profiling Hardware Metrics...")
            profiler_feedback = profile_kernel(candidate_code, prompt_text)
            messages.append({
                "role": "user",
                "content": (
                    f"Success! Your kernel ran in {runtime_ms:.3f} ms "
                    f"({reward:.2f}x speedup over baseline).\n\n"
                    f"Hardware Profiler Report:\n{profiler_feedback}\n\n"
                    "Apply the optimization playbook strategies to push the speedup further."
                ),
            })

    print(f"\n🏁 Episode Complete. Best Reward: {best_reward:.2f}x")
    return prompt_ids, all_completion_ids, all_logprobs, best_reward


# ---------------------------------------------------------------------------
# rollout_func — plugs into GRPOTrainer
# ---------------------------------------------------------------------------

def make_rollout_func(max_react_steps: int):
    """
    Factory that returns a rollout_func compatible with TRL's GRPOTrainer.

    TRL experimental API (rollout_func signature):
        fn(prompts: list[str], trainer: GRPOTrainer) -> dict[str, list]

    Required return keys (consumed by GRPOTrainer for loss computation):
        prompt_ids     — list[list[int]]
        completion_ids — list[list[int]]
        logprobs       — list[list[float]]

    Extra return keys are forwarded as kwargs to reward_funcs:
        reward         — list[float]  (pre-computed speedup from sandbox)
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


# ---------------------------------------------------------------------------
# Reward function — just forwards the pre-computed reward from the rollout
# ---------------------------------------------------------------------------

def reward_from_rollout(completions, reward, **kwargs) -> list[float]:
    """
    Reward function for GRPOTrainer.

    The actual reward (speedup ratio) was computed inside rollout_func by the
    sandbox. We simply forward it here so GRPOTrainer can normalize it across
    the group and compute advantages.
    """
    return list(reward)


# ---------------------------------------------------------------------------
# Training entry point
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

    training_args = GRPOConfig(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_generations=config.num_generations,
        max_prompt_length=config.max_prompt_length,
        max_completion_length=config.max_completion_length,
        learning_rate=config.learning_rate,
        beta=config.beta,
        num_train_epochs=config.num_train_epochs,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        bf16=True,
        gradient_checkpointing=True,
        # Avoid nested multiprocessing conflicts with the sandbox subprocess
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
    parser.add_argument("--dataset", type=str, default="data/rft_dataset.json")
    parser.add_argument("--output_dir", type=str, default="checkpoints/kernelforge_grpo")
    parser.add_argument("--max_react_steps", type=int, default=2)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--beta", type=float, default=0.01)
    args = parser.parse_args()

    config = KernelForgeConfig(
        rft_dataset_path=args.dataset,
        output_dir=args.output_dir,
        max_react_steps=args.max_react_steps,
        num_generations=args.num_generations,
        learning_rate=args.lr,
        beta=args.beta,
    )
    train(config)
