"""
train_ppo.py - Stage 3 of the KernelForge Pipeline: Multi-Turn Agentic RL via PPO.

Uses trl.experimental.ppo (PPOTrainer + AutoModelForCausalLMWithValueHead).
PPO requires a critic (value head) to estimate baselines — unlike GRPO which uses
group-normalized rewards. AutoModelForCausalLMWithValueHead adds the value head on
top of the SFT-fine-tuned Qwen3-8B.

Each RL episode is a full ReAct loop that mirrors agent.py exactly:
  - prefill: "```cpp\\n#include <torch/extension.h>\\n"  (matches SFT training format)
  - model outputs raw CUDA C++
  - extract_cuda_code() finds the ```cpp block
  - build_load_inline_wrapper() wraps it in Python for the sandbox
  - sandbox compiles, checks correctness, times execution
  - profiler reports hardware bottleneck
  - feedback returned for next turn
  - final reward = best speedup across all turns

Only agent-generated tokens go into the PPO response tensors.
Environment tokens (errors, profiler text) are context-only — never in response_tensors.

Reference: trl.experimental.ppo (TRL 0.29.0)
           "A Practitioner's Guide to Multi-Turn Agentic RL" (arXiv:2510.01132)
"""

import random
import re
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig

try:
    from trl import AutoModelForCausalLMWithValueHead
except ImportError:
    from trl.experimental.ppo import AutoModelForCausalLMWithValueHead

from trl.experimental.ppo import PPOConfig, PPOTrainer
from transformers import AutoTokenizer

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
    max_react_steps: int = 2

    # PPO settings
    batch_size: int = 4           # Number of prompts per PPO update step
    mini_batch_size: int = 1      # per_device_train_batch_size for gradient updates
    ppo_epochs: int = 4           # Number of optimization epochs per batch
    learning_rate: float = 1.4e-5
    kl_coef: float = 0.2          # KL penalty coefficient (controls drift from ref)
    cliprange: float = 0.2        # PPO clip range for policy ratio
    vf_coef: float = 0.1          # Value function loss coefficient
    cliprange_value: float = 0.2  # Clip range for value function
    max_new_tokens: int = 2048
    temperature: float = 0.7
    num_train_epochs: int = 1
    save_steps: int = 50

    # LoRA settings (same target modules as SFT)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


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
) -> tuple[list[int], str]:
    """
    Run one generation turn. Returns:
        generated_ids — agent-generated token IDs only (no prompt)
        decoded_text  — full response string including prefill
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
        )

    generated_ids = outputs[0][prompt_len:].tolist()
    decoded = PREFILL + tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_ids, decoded


# ---------------------------------------------------------------------------
# Episode — one full ReAct loop = one RL trajectory
# ---------------------------------------------------------------------------

def _run_react_episode(
    prompt_text: str,
    model,
    tokenizer,
    max_steps: int = 2,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run one full ReAct episode using the same format as agent.py.

    Returns:
        query_tensor    — token IDs of the initial prompt (1D LongTensor)
        response_tensor — all agent-generated token IDs concatenated across turns (1D LongTensor)
        reward_tensor   — scalar reward (best speedup achieved across all turns)

    Only agent-generated tokens go into response_tensor.
    Environment feedback messages are appended to `messages` for context but
    their tokens are never included in the response tensor.
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

    # Encode the initial prompt (query) — includes PREFILL since we force-start with it
    initial_text = (
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        + PREFILL
    )
    query_ids = tokenizer(initial_text, return_tensors="pt").input_ids[0]

    all_response_ids: list[int] = []
    best_reward = 0.0

    for step in range(max_steps):
        print(f"\n--- Step {step + 1}/{max_steps} ---")
        print("🧠 Generating Kernel...")

        generated_ids, response_text = _generate_turn(
            model, tokenizer, messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        # Only agent tokens go into the PPO response
        all_response_ids.extend(generated_ids)
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

    response_tensor = torch.tensor(all_response_ids, dtype=torch.long)
    reward_tensor = torch.tensor(best_reward, dtype=torch.float32)
    return query_ids, response_tensor, reward_tensor


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config: KernelForgeConfig = None):
    if config is None:
        config = KernelForgeConfig()

    print(f"Loading dataset from {config.rft_dataset_path}...")
    try:
        raw_dataset = load_dataset("json", data_files=config.rft_dataset_path, split="train")
        raw_dataset = raw_dataset.filter(lambda x: bool(x.get("pytorch_code", "").strip()))
    except Exception as e:
        print(f"Failed to load dataset.\n{e}")
        return
    prompts = [row["pytorch_code"] for row in raw_dataset]
    print(f"Loaded {len(prompts)} prompts.")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # LoRA config — same target modules as SFT stage, value head excluded (trained fully)
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Policy model with value head — LoRA adapters + value head are the only trained params
    # ref_model=None: TRL uses the frozen base layers (pre-LoRA weights) as implicit reference
    print(f"Loading policy model with value head + LoRA: {config.model_id}...")
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    ppo_config = PPOConfig(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.mini_batch_size,
        num_ppo_epochs=config.ppo_epochs,
        kl_coef=config.kl_coef,
        cliprange=config.cliprange,
        vf_coef=config.vf_coef,
        cliprange_value=config.cliprange_value,
        response_length=config.max_new_tokens,
        temperature=config.temperature,
        bf16=True,
        logging_steps=1,
        save_steps=config.save_steps,
    )

    trainer = PPOTrainer(
        args=ppo_config,
        processing_class=tokenizer,
        model=model,
        ref_model=None,       # None = use frozen base weights as reference (LoRA path)
        train_dataset=raw_dataset,
        peft_config=lora_config,
    )

    # Use the underlying pretrained LM for generation (not the value-head wrapper)
    gen_model = model.pretrained_model

    print(
        f"🚀 Launching multi-turn agentic PPO training...\n"
        f"   max_react_steps={config.max_react_steps}, "
        f"batch_size={config.batch_size}, "
        f"ppo_epochs={config.ppo_epochs}"
    )

    global_step = 0
    for epoch in range(config.num_train_epochs):
        epoch_prompts = prompts.copy()
        random.shuffle(epoch_prompts)

        for batch_start in range(0, len(epoch_prompts), config.batch_size):
            batch_prompts = epoch_prompts[batch_start : batch_start + config.batch_size]

            query_tensors: list[torch.Tensor] = []
            response_tensors: list[torch.Tensor] = []
            rewards: list[torch.Tensor] = []

            for prompt_text in batch_prompts:
                q, r, rew = _run_react_episode(
                    prompt_text,
                    gen_model,
                    tokenizer,
                    max_steps=config.max_react_steps,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                )
                query_tensors.append(q)
                response_tensors.append(r)
                rewards.append(rew)

            # PPO update — computes advantages via GAE, clips policy ratio, updates value head
            stats = trainer.step(query_tensors, response_tensors, rewards)

            global_step += 1
            mean_reward = sum(r.item() for r in rewards) / len(rewards)
            print(
                f"[Epoch {epoch + 1} | Step {global_step}] "
                f"mean_reward={mean_reward:.3f}x | "
                f"ppo/loss/total={stats.get('ppo/loss/total', 0):.4f} | "
                f"ppo/policy/kl={stats.get('ppo/policy/kl', 0):.4f}"
            )

            if global_step % config.save_steps == 0:
                ckpt = f"{config.output_dir}/step_{global_step}"
                trainer.save_model(ckpt)
                print(f"Checkpoint saved → {ckpt}")

    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print(f"Model saved to {config.output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="KernelForge Multi-Turn PPO Training")
    parser.add_argument("--dataset", type=str, default="../sft/sft_training_pairs.jsonl")
    parser.add_argument("--output_dir", type=str, default="checkpoints/kernelforge_ppo")
    parser.add_argument("--model", type=str, default="mukeshreddy/kernelforge-sft-qwen3-8b")
    parser.add_argument("--max_react_steps", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.4e-5)
    parser.add_argument("--kl_coef", type=float, default=0.2)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    args = parser.parse_args()

    config = KernelForgeConfig(
        model_id=args.model,
        rft_dataset_path=args.dataset,
        output_dir=args.output_dir,
        max_react_steps=args.max_react_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        kl_coef=args.kl_coef,
        ppo_epochs=args.ppo_epochs,
    )
    train(config)
