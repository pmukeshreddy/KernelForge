"""
train_ppo.py - Stage 3 of the KernelForge Pipeline: Multi-Turn Agentic RL via PPO.

trl.experimental.ppo.PPOTrainer requires a separate reward_model and value_model —
designed for RLHF with a learned reward model. Our reward is a CUDA sandbox, not a
neural network. So we implement the PPO update loop directly:

  1. Run ReAct episodes → collect (query_ids, response_ids, terminal_reward)
  2. Forward pass through actor-critic → get log probs + value estimates
  3. GAE advantage estimation (terminal reward at last response token)
  4. PPO clipped objective + value loss + KL penalty
  5. Backprop through LoRA adapters + value head only

Model architecture:
  - Policy+Critic: AutoModelForCausalLMWithValueHead with LoRA
    (LoRA adapters + v_head are the only trained parameters)
  - Reference: plain AutoModelForCausalLM, fully frozen
    (used only for per-token KL divergence penalty)

Each RL episode mirrors agent.py:
  - prefill "```cpp\\n#include <torch/extension.h>\\n"
  - extract_cuda_code() → build_load_inline_wrapper() → sandbox → profiler
  - final reward = best speedup across all turns
  - only agent-generated tokens enter the PPO gradient
"""

import random
import re
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.experimental.ppo import AutoModelForCausalLMWithValueHead

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

    # PPO hyperparameters
    batch_size: int = 4       # prompts per PPO update
    ppo_epochs: int = 4       # optimization passes per batch
    learning_rate: float = 1.4e-5
    kl_coef: float = 0.2      # KL penalty weight (keeps policy near reference)
    cliprange: float = 0.2    # PPO policy ratio clip
    vf_coef: float = 0.1      # value function loss weight
    cliprange_value: float = 0.2
    gamma: float = 1.0        # discount (1.0 = episodic, no discounting)
    lam: float = 0.95         # GAE lambda
    max_grad_norm: float = 1.0
    max_new_tokens: int = 2048
    temperature: float = 0.7
    num_train_epochs: int = 1
    save_steps: int = 50

    # LoRA settings — same as SFT stage
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


# ---------------------------------------------------------------------------
# Format helpers — must match agent.py exactly
# ---------------------------------------------------------------------------

PREFILL = "```cpp\n#include <torch/extension.h>\n"


def extract_cuda_code(text: str) -> str:
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
    Returns:
        generated_ids — agent-only token IDs (no prompt)
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
    gen_model,
    tokenizer,
    max_steps: int = 2,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Returns:
        query_ids    — initial prompt token IDs (1D LongTensor)
        response_ids — all agent-generated token IDs across all turns (1D LongTensor)
        reward       — best speedup achieved (scalar float)

    Environment feedback messages (errors, profiler) are appended to messages
    as context but their tokens are NEVER included in response_ids.
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
    query_ids = tokenizer(initial_text, return_tensors="pt").input_ids[0]

    all_response_ids: list[int] = []
    best_reward = 0.0

    for step in range(max_steps):
        print(f"\n--- Step {step + 1}/{max_steps} ---")
        print("🧠 Generating Kernel...")

        generated_ids, response_text = _generate_turn(
            gen_model, tokenizer, messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        all_response_ids.extend(generated_ids)
        messages.append({"role": "assistant", "content": response_text})

        cuda_code = extract_cuda_code(response_text)
        if not cuda_code:
            print("❌ No ```cpp block found.")
            messages.append({
                "role": "user",
                "content": "Error: No ```cpp block found. Output CUDA C++ in a ```cpp code block.",
            })
            continue

        candidate_code = build_load_inline_wrapper(cuda_code, prompt_text)
        if not candidate_code:
            print("❌ No torch::Tensor binding function found.")
            messages.append({
                "role": "user",
                "content": "Error: No `torch::Tensor` binding function found. Add a C++ function returning `torch::Tensor`.",
            })
            continue

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
    return query_ids, torch.tensor(all_response_ids, dtype=torch.long), best_reward


# ---------------------------------------------------------------------------
# PPO core — forward pass, GAE, update
# ---------------------------------------------------------------------------

def _get_log_probs_and_values(
    model,
    query_ids: torch.Tensor,
    response_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass through AutoModelForCausalLMWithValueHead.
    Returns:
        token_log_probs — log π_θ(r_t | context) for each response token [R]
        values          — V(s_t) for each response token position [R]
    """
    device = next(model.parameters()).device
    input_ids = torch.cat([query_ids, response_ids]).unsqueeze(0).to(device)
    Q = len(query_ids)
    R = len(response_ids)
    r_ids = response_ids.to(device)

    lm_logits, _, values = model(input_ids=input_ids)
    # lm_logits: [1, Q+R, vocab_size]
    # values:    [1, Q+R] or [1, Q+R, 1]

    # Positions Q-1 .. Q+R-2 predict response tokens r_0 .. r_{R-1}
    resp_logits = lm_logits[0, Q - 1 : Q + R - 1, :]          # [R, vocab_size]
    log_probs = F.log_softmax(resp_logits, dim=-1)             # [R, vocab_size]
    token_log_probs = log_probs[range(R), r_ids]               # [R]

    v = values[0, Q - 1 : Q + R - 1]
    if v.dim() == 2:
        v = v.squeeze(-1)                                      # [R]

    return token_log_probs, v


@torch.no_grad()
def _get_ref_log_probs(
    ref_model,
    query_ids: torch.Tensor,
    response_ids: torch.Tensor,
) -> torch.Tensor:
    """Log probs from the frozen reference model. [R]"""
    device = next(ref_model.parameters()).device
    input_ids = torch.cat([query_ids, response_ids]).unsqueeze(0).to(device)
    Q = len(query_ids)
    R = len(response_ids)
    r_ids = response_ids.to(device)

    outputs = ref_model(input_ids=input_ids)
    resp_logits = outputs.logits[0, Q - 1 : Q + R - 1, :]     # [R, vocab_size]
    log_probs = F.log_softmax(resp_logits, dim=-1)
    return log_probs[range(R), r_ids]                          # [R]


def _compute_gae(
    rewards: torch.Tensor,   # [R]  shaped rewards (KL-penalized)
    values: torch.Tensor,    # [R]  value estimates
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE advantage + returns. Both [R]."""
    R = len(rewards)
    advantages = torch.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(R)):
        next_value = values[t + 1].item() if t < R - 1 else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


def _ppo_update(
    model,
    ref_model,
    optimizer,
    trajectories: list[tuple[torch.Tensor, torch.Tensor, float]],
    old_log_probs_list: list[torch.Tensor],
    config: KernelForgeConfig,
):
    """
    Run `ppo_epochs` passes of the PPO objective over the collected batch.
    Loss = policy clip loss + vf_coef * value loss + kl_coef * KL penalty
    """
    for ppo_ep in range(config.ppo_epochs):
        batch_loss = torch.tensor(0.0)

        for (q_ids, r_ids, reward), old_lp in zip(trajectories, old_log_probs_list):
            if len(r_ids) == 0:
                continue  # empty response (all steps failed) — skip

            # Current policy log probs + values
            token_lp, values = _get_log_probs_and_values(model, q_ids, r_ids)
            R = len(r_ids)

            # Reference log probs for KL penalty
            ref_lp = _get_ref_log_probs(ref_model, q_ids, r_ids)

            # Per-token KL: log(π/π_ref)
            kl = token_lp - ref_lp.to(token_lp.device)       # [R]

            # Shaped rewards: terminal reward + KL penalty at every token
            shaped = -config.kl_coef * kl.detach()
            shaped[-1] = shaped[-1] + reward

            # GAE
            advantages, returns = _compute_gae(
                shaped, values.detach(), config.gamma, config.lam
            )

            # Normalize advantages across this trajectory
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # PPO clipped policy loss
            old_lp_dev = old_lp.to(token_lp.device)
            ratio = torch.exp(token_lp - old_lp_dev)
            pg1 = -advantages * ratio
            pg2 = -advantages * torch.clamp(ratio, 1 - config.cliprange, 1 + config.cliprange)
            pg_loss = torch.max(pg1, pg2).mean()

            # Value function loss (clipped)
            v_clipped = old_lp_dev.detach() + torch.clamp(
                values - old_lp_dev.detach(),
                -config.cliprange_value,
                config.cliprange_value,
            )
            vf_loss = torch.max(
                F.mse_loss(values, returns),
                F.mse_loss(v_clipped, returns),
            )

            loss = pg_loss + config.vf_coef * vf_loss
            batch_loss = batch_loss + loss

        optimizer.zero_grad()
        (batch_loss / max(len(trajectories), 1)).backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            config.max_grad_norm,
        )
        optimizer.step()

        print(f"  [PPO epoch {ppo_ep + 1}/{config.ppo_epochs}] loss={batch_loss.item():.4f}")


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

    tokenizer = AutoTokenizer.from_pretrained(config.model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # ── Actor-Critic: AutoModelForCausalLMWithValueHead + LoRA ──────────────
    print(f"Loading policy model (actor-critic): {config.model_id}...")
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model.pretrained_model = get_peft_model(model.pretrained_model, lora_config)
    model.pretrained_model.print_trainable_parameters()

    # ── Reference model: frozen base LM (no value head, no LoRA) ────────────
    print("Loading frozen reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # Only LoRA adapters + value head are trained
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate)

    # For generation we use the underlying pretrained model (avoids value head interference)
    gen_model = model.pretrained_model

    print(
        f"🚀 Launching multi-turn agentic PPO training...\n"
        f"   max_react_steps={config.max_react_steps}, "
        f"batch_size={config.batch_size}, ppo_epochs={config.ppo_epochs}"
    )

    import os
    os.makedirs(config.output_dir, exist_ok=True)

    global_step = 0
    for epoch in range(config.num_train_epochs):
        epoch_prompts = prompts.copy()
        random.shuffle(epoch_prompts)

        for batch_start in range(0, len(epoch_prompts), config.batch_size):
            batch_prompts = epoch_prompts[batch_start : batch_start + config.batch_size]

            # ── Rollout phase: collect trajectories ──────────────────────────
            trajectories: list[tuple[torch.Tensor, torch.Tensor, float]] = []
            for prompt_text in batch_prompts:
                q, r, rew = _run_react_episode(
                    prompt_text, gen_model, tokenizer,
                    max_steps=config.max_react_steps,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                )
                trajectories.append((q, r, rew))

            mean_reward = sum(t[2] for t in trajectories) / len(trajectories)
            print(f"\n[Epoch {epoch + 1} | Step {global_step + 1}] "
                  f"mean_reward={mean_reward:.3f}x  ({len(trajectories)} episodes)")

            # ── Old log probs (for PPO importance ratio) ─────────────────────
            old_log_probs_list = []
            model.eval()
            for q, r, _ in trajectories:
                if len(r) == 0:
                    old_log_probs_list.append(torch.zeros(0))
                    continue
                with torch.no_grad():
                    lp, _ = _get_log_probs_and_values(model, q, r)
                old_log_probs_list.append(lp.detach())
            model.train()

            # ── PPO update ───────────────────────────────────────────────────
            _ppo_update(model, ref_model, optimizer, trajectories, old_log_probs_list, config)

            global_step += 1
            if global_step % config.save_steps == 0:
                ckpt = f"{config.output_dir}/step_{global_step}"
                model.pretrained_model.save_pretrained(ckpt)
                tokenizer.save_pretrained(ckpt)
                print(f"Checkpoint saved → {ckpt}")

    model.pretrained_model.save_pretrained(config.output_dir)
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
