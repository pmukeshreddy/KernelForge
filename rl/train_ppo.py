"""
train_ppo.py - Stage 3 of the KernelForge Pipeline: Multi-Turn Agentic RL via PPO.

Self-contained PPO (no PPOTrainer) because trl.experimental.ppo.PPOTrainer
requires a neural reward_model — our reward is a CUDA sandbox.

Architecture:
  Policy+Critic : AutoModelForCausalLMWithValueHead (existing SFT LoRA + new v_head)
  Reference     : AutoModelForCausalLM frozen at SFT checkpoint (KL anchor)

Multi-turn correctness:
  Each turn stores its OWN context_ids (initial prompt + all prior assistant/env
  messages). Log probs and values are computed per-turn against the context that
  was actually used during generation — not a flat concatenation that ignores
  intervening environment feedback.

Fixes applied vs. first draft:
  1. Value clipping uses old_values (not old_log_probs) — was numerically broken
  2. old_values collected during rollout alongside old_log_probs
  3. Per-trajectory backward (no CPU batch_loss accumulation) — gradient device safe
  4. Per-turn (context_ids, response_ids) tracking — correct credit assignment
  5. Negative reward for failures (contrast signal for PPO)
  6. Entropy bonus — prevents policy collapse
  7. ref_model = frozen SFT checkpoint (correct RL-from-SFT anchor; NOT base Qwen)
"""

import os
import random
import re
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer
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
    batch_size: int = 4
    ppo_epochs: int = 4
    learning_rate: float = 1.4e-5
    kl_coef: float = 0.2
    cliprange: float = 0.2
    vf_coef: float = 0.1
    cliprange_value: float = 0.2
    entropy_coef: float = 0.01   # entropy bonus to prevent policy collapse
    gamma: float = 1.0
    lam: float = 0.95
    max_grad_norm: float = 1.0
    max_new_tokens: int = 2048
    temperature: float = 0.7
    num_train_epochs: int = 1
    save_steps: int = 50

    # Reward shaping
    reward_failure: float = -1.0   # compile error or wrong output
    reward_no_code: float = -1.0   # no ```cpp block at all


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
# Generation helpers
# ---------------------------------------------------------------------------

def _generate_from_context(
    model,
    tokenizer,
    context_ids: torch.Tensor,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
) -> tuple[list[int], str]:
    """
    Generate from pre-tokenized context_ids.
    Returns generated_ids (agent tokens only) and decoded text including prefill.
    """
    input_ids = context_ids.unsqueeze(0).to(next(model.parameters()).device)
    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][prompt_len:].tolist()
    decoded = PREFILL + tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_ids, decoded


# ---------------------------------------------------------------------------
# Episode — multi-turn ReAct loop
# ---------------------------------------------------------------------------

# Each turn stores the context that was used + the agent response
TurnData = tuple[torch.Tensor, torch.Tensor]  # (context_ids, response_ids)


def _run_react_episode(
    prompt_text: str,
    gen_model,
    tokenizer,
    max_steps: int = 2,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    reward_failure: float = -1.0,
    reward_no_code: float = -1.0,
) -> tuple[list[TurnData], float]:
    """
    Run one full ReAct episode.

    Returns:
        turns  — list of (context_ids, response_ids) ONE entry per turn.
                 context_ids is the actual tokenized prompt fed to the model
                 for that specific turn (includes prior env feedback).
                 NEVER a flat concatenation across turns.
        reward — terminal reward: best speedup achieved, or negative for failure.

    Reward scale:
        compile/correctness failure : reward_failure  (negative — contrast signal)
        no ```cpp block             : reward_no_code   (negative)
        success                     : calculate_reward() speedup ratio (≥ 1.0)
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

    turns: list[TurnData] = []
    best_reward = reward_failure  # default: penalise if nothing ever compiles

    for step in range(max_steps):
        print(f"\n--- Step {step + 1}/{max_steps} ---")

        # Build THIS turn's context and tokenize it
        context_text = (
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            + PREFILL
        )
        context_ids = tokenizer(context_text, return_tensors="pt").input_ids[0]

        print("🧠 Generating Kernel...")
        generated_ids, response_text = _generate_from_context(
            gen_model, tokenizer, context_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        # Store THIS turn: (context used for generation, tokens the agent produced)
        turns.append((context_ids, torch.tensor(generated_ids, dtype=torch.long)))
        messages.append({"role": "assistant", "content": response_text})

        # Extract + wrap
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

        # Sandbox
        print("🛠️  Compiling and Evaluating in Sandbox...")
        eval_result = evaluate(candidate_code, prompt_text)

        if not eval_result["correct"]:
            error = eval_result.get("compiler_error", "Outputs do not match reference.")
            print(f"❌ Failed:\n{error.strip()[:600]}")
            messages.append({
                "role": "user",
                "content": (
                    f"Your kernel failed.\n\nError:\n```\n{error[:1500]}\n```\n\n"
                    "Fix the bug and output the corrected C++ code."
                ),
            })
            # best_reward stays at reward_failure unless a later step succeeds
            continue

        # Success
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

    print(f"\n🏁 Episode done. Best reward: {best_reward:.2f}")
    return turns, best_reward


# ---------------------------------------------------------------------------
# PPO core
# ---------------------------------------------------------------------------

def _get_log_probs_values_entropy(
    model,
    context_ids: torch.Tensor,
    response_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward pass for one turn.
    Returns:
        token_log_probs — log π_θ(r_t | context) [R]
        values          — V(s_t) at each response position [R]
        entropy         — mean token entropy scalar (for bonus term)
    """
    device = next(model.parameters()).device
    input_ids = torch.cat([context_ids, response_ids]).unsqueeze(0).to(device)
    Q = len(context_ids)
    R = len(response_ids)
    r_ids = response_ids.to(device)

    lm_logits, _, values = model(input_ids=input_ids)

    resp_logits = lm_logits[0, Q - 1 : Q + R - 1, :]       # [R, V]
    log_probs_all = F.log_softmax(resp_logits, dim=-1)       # [R, V]
    token_log_probs = log_probs_all[range(R), r_ids]         # [R]

    # Entropy = -E[log p] = -sum(p * log_p)
    entropy = -(log_probs_all.exp() * log_probs_all).sum(-1).mean()  # scalar

    v = values[0, Q - 1 : Q + R - 1]
    if v.dim() == 2:
        v = v.squeeze(-1)                                    # [R]

    return token_log_probs, v, entropy


@torch.no_grad()
def _get_ref_log_probs(
    model,
    context_ids: torch.Tensor,
    response_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Reference log probs for one turn — base model only (LoRA disabled).
    Shares weights with the policy model: no second model copy, no extra VRAM.
    KL anchor = pretrained base distribution (before any SFT/RL fine-tuning).
    """
    peft_model = model.pretrained_model
    device = next(model.parameters()).device
    input_ids = torch.cat([context_ids, response_ids]).unsqueeze(0).to(device)
    Q = len(context_ids)
    R = len(response_ids)
    r_ids = response_ids.to(device)

    with peft_model.disable_adapter():
        outputs = peft_model(input_ids=input_ids)

    resp_logits = outputs.logits[0, Q - 1 : Q + R - 1, :]
    log_probs = F.log_softmax(resp_logits, dim=-1)
    return log_probs[range(R), r_ids]


def _compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE advantage + returns. Both [R]."""
    R = len(rewards)
    advantages = torch.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(R)):
        next_v = values[t + 1].item() if t < R - 1 else 0.0
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    return advantages, advantages + values


def _compute_trajectory_loss(
    model,
    turns: list[TurnData],
    reward: float,
    old_log_probs: list[torch.Tensor],
    old_values: list[torch.Tensor],
    config: KernelForgeConfig,
) -> torch.Tensor:
    """
    Compute PPO loss for one multi-turn episode.
    Each turn is evaluated against its own actual context — no flat concatenation.
    GAE runs over the flattened (time-ordered) sequence of all turns.
    """
    all_lp, all_val, all_entropy = [], [], []
    all_ref_lp = []

    for (ctx, resp) in turns:
        if len(resp) == 0:
            continue
        lp, val, ent = _get_log_probs_values_entropy(model, ctx, resp)
        ref_lp = _get_ref_log_probs(model, ctx, resp)
        all_lp.append(lp)
        all_val.append(val)
        all_entropy.append(ent)
        all_ref_lp.append(ref_lp.to(lp.device))

    if not all_lp:
        return torch.tensor(0.0, requires_grad=True)

    token_lp = torch.cat(all_lp)                # [T_total]
    values    = torch.cat(all_val)               # [T_total]
    ref_lp    = torch.cat(all_ref_lp)            # [T_total]
    entropy   = torch.stack(all_entropy).mean()  # scalar
    T = len(token_lp)

    # KL per token: log(π / π_ref)
    kl = token_lp - ref_lp                              # [T]

    # Shaped rewards: KL penalty at every token, terminal reward at last token
    shaped = -config.kl_coef * kl.detach()
    shaped[-1] = shaped[-1] + reward

    # GAE
    advantages, returns = _compute_gae(shaped, values.detach(), config.gamma, config.lam)

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Flatten old tensors (same order as all_lp)
    old_lp_cat = torch.cat([lp.to(token_lp.device) for lp in old_log_probs])
    old_val_cat = torch.cat([v.to(values.device) for v in old_values])

    # PPO clipped policy loss
    ratio = torch.exp(token_lp - old_lp_cat.detach())
    pg1 = -advantages * ratio
    pg2 = -advantages * torch.clamp(ratio, 1 - config.cliprange, 1 + config.cliprange)
    pg_loss = torch.max(pg1, pg2).mean()

    # Value function loss — clipped around OLD VALUES (not log probs)
    v_clipped = old_val_cat.detach() + torch.clamp(
        values - old_val_cat.detach(),
        -config.cliprange_value,
        config.cliprange_value,
    )
    vf_loss = torch.max(
        F.mse_loss(values, returns),
        F.mse_loss(v_clipped, returns),
    )

    # Entropy bonus (negative in loss = maximise entropy)
    total_loss = pg_loss + config.vf_coef * vf_loss - config.entropy_coef * entropy
    return total_loss


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

    # ── Policy + Critic ──────────────────────────────────────────────────────
    # Reuse existing SFT LoRA adapters — no second LoRA stacked on top.
    # Trainable: SFT LoRA adapter weights + v_head (value head).
    print(f"Loading policy model (actor-critic): {config.model_id}...")
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.pretrained_model.enable_adapter_layers()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {n_trainable:,} / {n_total:,} params ({100*n_trainable/n_total:.2f}%)")

    # No separate reference model needed — _get_ref_log_probs disables LoRA
    # adapters on the policy model for reference forward passes. Saves ~16GB VRAM.
    # KL anchor = base model weights (pre-SFT), which is standard for PPO-on-LoRA.

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate)

    # Use underlying LM (without value head) for generation
    gen_model = model.pretrained_model

    os.makedirs(config.output_dir, exist_ok=True)

    print(
        f"🚀 Launching multi-turn agentic PPO training...\n"
        f"   max_react_steps={config.max_react_steps}, "
        f"batch_size={config.batch_size}, ppo_epochs={config.ppo_epochs}"
    )

    global_step = 0
    for epoch in range(config.num_train_epochs):
        epoch_prompts = prompts.copy()
        random.shuffle(epoch_prompts)

        for batch_start in range(0, len(epoch_prompts), config.batch_size):
            batch_prompts = epoch_prompts[batch_start : batch_start + config.batch_size]

            # ── Rollout phase ────────────────────────────────────────────────
            all_turns:   list[list[TurnData]] = []
            all_rewards: list[float] = []

            for prompt_text in batch_prompts:
                turns, reward = _run_react_episode(
                    prompt_text, gen_model, tokenizer,
                    max_steps=config.max_react_steps,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    reward_failure=config.reward_failure,
                    reward_no_code=config.reward_no_code,
                )
                all_turns.append(turns)
                all_rewards.append(reward)

            mean_reward = sum(all_rewards) / len(all_rewards)
            print(f"\n[Epoch {epoch+1} | Step {global_step+1}] mean_reward={mean_reward:.3f}")

            # ── Collect old log probs + old values (before any gradient update) ──
            all_old_lp:  list[list[torch.Tensor]] = []
            all_old_val: list[list[torch.Tensor]] = []

            model.eval()
            for turns in all_turns:
                ep_lp, ep_val = [], []
                for ctx, resp in turns:
                    if len(resp) == 0:
                        ep_lp.append(torch.zeros(0))
                        ep_val.append(torch.zeros(0))
                        continue
                    with torch.no_grad():
                        lp, val, _ = _get_log_probs_values_entropy(model, ctx, resp)
                    ep_lp.append(lp.detach())
                    ep_val.append(val.detach())
                all_old_lp.append(ep_lp)
                all_old_val.append(ep_val)
            model.train()

            # ── PPO epochs ───────────────────────────────────────────────────
            for ppo_ep in range(config.ppo_epochs):
                optimizer.zero_grad()
                total_loss_val = 0.0
                n_valid = sum(1 for t in all_turns if any(len(r) > 0 for _, r in t))

                for turns, reward, old_lp, old_val in zip(
                    all_turns, all_rewards, all_old_lp, all_old_val
                ):
                    if not any(len(r) > 0 for _, r in turns):
                        continue

                    loss = _compute_trajectory_loss(
                        model, turns, reward, old_lp, old_val, config
                    )
                    # Gradient accumulation: backward per-trajectory, normalize by batch size
                    (loss / max(n_valid, 1)).backward()
                    total_loss_val += loss.item()

                torch.nn.utils.clip_grad_norm_(trainable_params, config.max_grad_norm)
                optimizer.step()
                print(f"  [PPO epoch {ppo_ep+1}/{config.ppo_epochs}] loss={total_loss_val:.4f}")

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
