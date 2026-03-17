"""
train_grpo.py - Multi-Turn Agentic RL via GRPO + DAPO fixes.

Replaces train_ppo.py. Based on:
  - Kevin (Cognition): GRPO with multi-turn ReAct, γ=0.4 discount, G=16 per task
  - Dr. Kernel: TRLOO unbiased advantage for multi-turn
  - DAPO (ByteDance): Token-level loss, Clip-Higher, no KL penalty

Key differences from PPO:
  - No value head (no critic) — saves ~16GB VRAM
  - No reference model / KL penalty — DAPO showed it's unnecessary
  - No GAE — advantages from group-relative reward normalization
  - Token-level loss — prevents vanishing gradients on long sequences
  - Asymmetric clipping (Clip-Higher) — prevents entropy collapse
"""

import os
import re
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from concurrent.futures import ProcessPoolExecutor

from agent import build_load_inline_wrapper, _extract_cuda_code
from profiler import profile_kernel
from reward import calculate_reward
from sandbox import evaluate
from sys_prompt import get_system_prompt


def _worker_run_eval(args):
    cand, prompt_text = args
    if cand is None: return None
    return evaluate(cand, prompt_text)


def _worker_eval_pair(pair):
    cuda_code, p_code = pair
    if not cuda_code: return None
    candidate = build_load_inline_wrapper(cuda_code, p_code)
    if not candidate: return None
    return evaluate(candidate, p_code)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GRPOConfig:
    model_id: str = "Qwen/Qwen3-14B"
    adapter_path: str = "checkpoints/kernelforge_redi"
    dataset_path: str = "data/rft_dataset.json"
    output_dir: str = "checkpoints/kernelforge_grpo"

    # Episode
    max_react_steps: int = 4          # turns per trajectory (Kevin uses 4)
    group_size: int = 16              # G trajectories per prompt (Kevin uses 16, gives more stable advantage estimates)

    # GRPO + DAPO hyperparameters
    grpo_epochs: int = 4              # gradient updates per batch
    batch_size: int = 4               # prompts per batch
    learning_rate: float = 1e-6
    cliprange_low: float = 0.2        # standard lower clip
    cliprange_high: float = 0.28      # DAPO Clip-Higher (asymmetric)
    max_grad_norm: float = 1.0
    reward_discount: float = 0.4      # multi-turn γ (Kevin's value)

    # Generation
    max_new_tokens: int = 2048
    temperature: float = 0.7
    mock_mode: bool = False

    # Training
    num_train_epochs: int = 1
    save_steps: int = 50
    eval_steps: int = 50
    wandb_project: str = "kernelforge-rl"
    wandb_run_name: str = "grpo-qwen-14b"

    # Reward shaping (graduated — creates gradient signal at every failure stage)
    reward_no_code: float = -1.0      # no ```cpp block found at all
    reward_compile_fail: float = -0.5 # code found but fails to compile or wrap
    reward_wrong_output: float = -0.1 # compiles but produces wrong outputs
    reward_correct_base: float = 0.3  # base bonus for any correct kernel (Kevin's approach)

    # Dynamic Sampling: skip degenerate groups where all rewards are identical
    dynamic_sampling: bool = True
    max_resample_attempts: int = 3


# ---------------------------------------------------------------------------
# Format helpers (must match agent.py)
# ---------------------------------------------------------------------------

PREFILL = "```cpp\n#include <torch/extension.h>\n"



# ---------------------------------------------------------------------------
# Multi-turn episode
# ---------------------------------------------------------------------------

# Each turn: (context_ids, response_ids)
TurnData = tuple[torch.Tensor, torch.Tensor]


def _run_group_episodes(
    prompt_text: str,
    model,
    tokenizer,
    config: GRPOConfig,
) -> tuple[list[list[TurnData]], list[float]]:
    """
    Run a group of trajectories for a single prompt simultaneously.
    Batches the inference and parallelizes the sandbox evaluation.

    Returns:
      group_turns: list of G trajectory turn histories
      group_rewards: list of G scalar discounted rewards
    """
    G = config.group_size
    system_prompt = get_system_prompt()
    base_messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Write an optimized CUDA C++ kernel to replace this PyTorch "
                "implementation. Output only the C++ code.\n\n"
                f"Reference Program:\n```python\n{prompt_text}\n```"
            ),
        },
    ]

    # State per trajectory
    messages_list = [base_messages.copy() for _ in range(G)]
    turns_list: list[list[TurnData]] = [[] for _ in range(G)]
    per_turn_rewards_list: list[list[float]] = [[] for _ in range(G)]
    active_mask = [True] * G  # Track which trajectories are still generating

    for step in range(config.max_react_steps):
        active_indices = [i for i, active in enumerate(active_mask) if active]
        if not active_indices:
            break

        # 1. Prepare batch context
        context_texts = []
        for i in active_indices:
            ctx = tokenizer.apply_chat_template(
                messages_list[i], tokenize=False, add_generation_prompt=True
            ) + PREFILL
            context_texts.append(ctx)

        # Tokenize (left padding is needed for batched generation)
        if not config.mock_mode:
            tokenizer.padding_side = "left"
            inputs = tokenizer(context_texts, return_tensors="pt", padding=True)
            input_ids = inputs.input_ids.to(next(model.parameters()).device)
            attention_mask = inputs.attention_mask.to(next(model.parameters()).device)
    
            # 2. Batched Generation
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
        else:
            outputs = [None] * len(active_indices)
            input_ids = [None] * len(active_indices)

        # Extract generated portion
        generated_ids_list = []
        response_texts = []
        for batch_idx, traj_idx in enumerate(active_indices):
            if not config.mock_mode:
                seq_len = input_ids[batch_idx].shape[0]
                gen_ids = outputs[batch_idx][seq_len:]
                generated_ids_list.append(gen_ids)
                
                resp_text = PREFILL + tokenizer.decode(gen_ids, skip_special_tokens=True)
                response_texts.append(resp_text)
                
                # Save history
                exact_ctx_ids = tokenizer(context_texts[batch_idx], return_tensors="pt").input_ids[0]
                turns_list[traj_idx].append((exact_ctx_ids, gen_ids.cpu()))
                messages_list[traj_idx].append({"role": "assistant", "content": resp_text})
            else:
                resp_text = PREFILL + "import torch\ntorch.Tensor\n#include <torch/extension.h>\n__global__ void mykernel() {}\ntorch::Tensor foo(torch::Tensor a) { return a; }\n```\n"
                response_texts.append(resp_text)
                
                exact_ctx_ids = torch.tensor([1, 2, 3])
                gen_ids = torch.tensor([4, 5, 6])
                turns_list[traj_idx].append((exact_ctx_ids, gen_ids))
                messages_list[traj_idx].append({"role": "assistant", "content": resp_text})

        # 3. Parallel Evaluation
        # First, extract code quickly
        candidates = []
        error_msgs = []
        for resp in response_texts:
            cuda_code = _extract_cuda_code(resp)
            if not cuda_code:
                candidates.append(None)
                error_msgs.append("Error: No ```cpp block found. Output CUDA C++ in a ```cpp code block.")
                continue
            
            cand_code = build_load_inline_wrapper(cuda_code, prompt_text)
            if not cand_code:
                candidates.append(None)
                error_msgs.append("Error: No torch::Tensor binding function found.")
                continue
                
            candidates.append(cand_code)
            error_msgs.append(None)

        # Run valid candidates in process pool
        eval_results = [None] * len(active_indices)
        
        with ProcessPoolExecutor(max_workers=min(G, 16)) as pool:
            eval_results = list(pool.map(
                _worker_run_eval,
                [(c, prompt_text) for c in candidates]
            ))

        # Compile rate for this step (helps decide whether to run slow NCU)
        n_compiled = sum(1 for res in eval_results if res is not None and res.get("compiles", res.get("correct", False)))
        compile_rate = n_compiled / max(1, len(active_indices))

        # 4. Process results and update trajectories
        for batch_idx, traj_idx in enumerate(active_indices):
            eval_res = eval_results[batch_idx]
            
            # DAPO Soft overlong punishment: slight negative reward for generating excessively many tokens
            gen_len = len(turns_list[traj_idx][-1][1])
            length_penalty = -0.001 * gen_len
            
            if eval_res is None:
                # No code block or no binding function found
                base_r = config.reward_no_code if error_msgs[batch_idx].startswith("Error: No ```cpp") else config.reward_compile_fail
                per_turn_rewards_list[traj_idx].append(base_r + length_penalty)
                messages_list[traj_idx].append({
                    "role": "user",
                    "content": error_msgs[batch_idx],
                })
                continue

            if not eval_res["correct"]:
                # Distinguish compile failure from wrong output — different gradient signal
                if not eval_res.get("compiles", False):
                    base_r = config.reward_compile_fail   # -0.5: compiled but linker/nvcc error
                else:
                    base_r = config.reward_wrong_output   # -0.1: compiled, ran, but wrong answer
                per_turn_rewards_list[traj_idx].append(base_r + length_penalty)
                error = eval_res.get("compiler_error", "Outputs do not match.")
                messages_list[traj_idx].append({
                    "role": "user",
                    "content": (
                        f"Your kernel failed.\n\nError:\n```\n{error[:1500]}\n```\n\n"
                        "Fix the bug and output the corrected C++ code."
                    ),
                })
                continue

            # Success: base bonus + speedup (Kevin's approach — separates correct-but-slow from failures)
            reward = config.reward_correct_base + calculate_reward(eval_res) + length_penalty
            per_turn_rewards_list[traj_idx].append(reward)
            runtime_ms = eval_res["runtime_ms"]
            print(f"    ✅ Traj {traj_idx+1} Step {step+1}: {runtime_ms:.3f}ms, {reward:.2f}x")

            if step < config.max_react_steps - 1:
                # Need to profile to get feedback for next turn
                # Skip NCU if batch compile rate is < 50% to save time early in training
                if compile_rate >= 0.5 and not config.mock_mode:
                    profiler_feedback = profile_kernel(candidates[batch_idx], prompt_text)
                else:
                    profiler_feedback = "Profiler Skipped: Batch compile rate is < 50%. Focus on syntax and basic correctness first."
                    
                messages_list[traj_idx].append({
                    "role": "user",
                    "content": (
                        f"Success! {runtime_ms:.3f}ms ({reward:.2f}x speedup).\n\n"
                        f"Profiler:\n{profiler_feedback}\n\n"
                        "Optimize further. Output improved ```cpp code."
                    ),
                })
            else:
                active_mask[traj_idx] = False # Done optimizing


    # Calculate discounted rewards for all trajectories
    group_rewards = []
    for traj_idx in range(G):
        r_list = per_turn_rewards_list[traj_idx]
        total = sum(
            (config.reward_discount ** t) * r
            for t, r in enumerate(r_list)
        ) if r_list else config.reward_compile_fail
        group_rewards.append(total)

    return turns_list, group_rewards


# ---------------------------------------------------------------------------
# GRPO core
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 3072


def _get_token_log_probs(
    model,
    context_ids: torch.Tensor,
    response_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Forward pass for one turn. Returns per-token log probs [R].
    Truncates context from the left if total length exceeds MAX_SEQ_LEN.
    """
    R = len(response_ids)
    max_ctx = max(MAX_SEQ_LEN - R, 64)
    if len(context_ids) > max_ctx:
        context_ids = context_ids[-max_ctx:]

    device = next(model.parameters()).device
    input_ids = torch.cat([context_ids, response_ids]).unsqueeze(0).to(device)
    Q = len(context_ids)
    r_ids = response_ids.to(device)

    with torch.no_grad() if not torch.is_grad_enabled() else torch.enable_grad():
        outputs = model(input_ids=input_ids)

    resp_logits = outputs.logits[0, Q - 1 : Q + R - 1, :]  # [R, V]
    log_probs = F.log_softmax(resp_logits, dim=-1)           # [R, V]
    token_log_probs = log_probs[range(R), r_ids]              # [R]

    return token_log_probs


def _compute_grpo_loss(
    model,
    group_turns: list[list[TurnData]],
    group_rewards: list[float],
    old_log_probs: list[list[torch.Tensor]],
    config: GRPOConfig,
) -> torch.Tensor:
    """
    Compute GRPO loss for one group of G trajectories from the same prompt.

    1. Group-relative advantage: A_i = (r_i - mean) / (std + ε)
    2. Token-level clipped loss with DAPO Clip-Higher
    """
    G = len(group_rewards)
    rewards = torch.tensor(group_rewards, dtype=torch.float32)

    # Group-relative advantage normalization (the core GRPO idea)
    mean_r = rewards.mean()
    std_r = rewards.std() + 1e-8
    advantages = (rewards - mean_r) / std_r  # [G]

    total_loss = torch.tensor(0.0, device=next(model.parameters()).device,
                              requires_grad=True)
    n_tokens = 0

    for i in range(G):
        A_i = advantages[i].item()
        turns = group_turns[i]
        old_lps = old_log_probs[i]

        for turn_idx, (ctx, resp) in enumerate(turns):
            if len(resp) == 0:
                continue

            # Current policy log probs (with gradients)
            new_lp = _get_token_log_probs(model, ctx, resp)
            old_lp = old_lps[turn_idx].to(new_lp.device)

            # Policy ratio
            ratio = torch.exp(new_lp - old_lp.detach())

            # DAPO Clip-Higher: asymmetric clipping
            clipped = torch.clamp(
                ratio,
                1.0 - config.cliprange_low,
                1.0 + config.cliprange_high,
            )

            # Token-level loss (not sample-level mean then loss)
            token_loss = -torch.min(ratio * float(A_i), clipped * float(A_i))
            total_loss = total_loss + token_loss.sum()
            n_tokens += len(resp)

    if n_tokens > 0:
        total_loss = total_loss / float(n_tokens)

    return total_loss


def _run_evaluation(model, tokenizer, config: GRPOConfig, val_prompts: list[str]) -> dict:
    """Run an evaluation pass on the validation set."""
    print(f"\n--- Running Evaluation ({len(val_prompts)} prompts) ---")
    if config.mock_mode:
        return {"eval/pass_rate": 0.5, "eval/valid_rate": 0.8, "eval/avg_speedup": 1.1}

    model.eval()
    system_prompt = get_system_prompt()
    candidates_to_eval = []
    
    with torch.no_grad():
        for prompt_text in val_prompts:
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        "Write an optimized CUDA C++ kernel to replace this PyTorch "
                        "implementation. Output only the C++ code.\n\n"
                        f"Reference Program:\n```python\n{prompt_text}\n```"
                    ),
                },
            ]
            
            prompt_str = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            input_ids = tokenizer(prompt_str, return_tensors="pt").input_ids.to(model.device)
            
            # Generate single best guess (greedy, T=0.0)
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=config.max_new_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            resp_ids = output_ids[0][len(input_ids[0]):]
            response_text = tokenizer.decode(resp_ids, skip_special_tokens=True)
            
            cuda_code = _extract_cuda_code(response_text)
            candidates_to_eval.append((cuda_code, prompt_text))
            
    # Parallel eval over generated candidates
    success = 0
    valid = 0
    total_speedup = 0.0
    
    with ProcessPoolExecutor(max_workers=min(16, len(val_prompts))) as pool:
        eval_results = list(pool.map(_worker_eval_pair, candidates_to_eval))
        
    for res in eval_results:
        if res is not None:
            valid += 1
            if res["correct"]:
                success += 1
                total_speedup += calculate_reward(res)
                
    total = len(val_prompts)
    results = {
        "eval/pass_rate": success / total,
        "eval/valid_rate": valid / total,
        "eval/avg_speedup": total_speedup / max(1, success)
    }
    
    print(f"Eval Results: pass_rate={results['eval/pass_rate']:.2%}, "
          f"valid_rate={results['eval/valid_rate']:.2%}, "
          f"avg_speedup={results['eval/avg_speedup']:.2f}x")
    model.train()
    return results


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config: GRPOConfig = None):
    if config is None:
        config = GRPOConfig()

    print(f"Loading dataset from {config.dataset_path}...")
    try:
        raw = load_dataset("json", data_files=config.dataset_path, split="train")
        raw = raw.filter(lambda x: bool(x.get("pytorch_code", "").strip()))
    except Exception as e:
        print(f"Failed to load dataset.\n{e}")
        return
    prompts = [row["pytorch_code"] for row in raw]
    # Split to train/val (10% val, max 20)
    random.shuffle(prompts)
    val_size = min(int(len(prompts) * 0.1), 20)
    if val_size == 0 and len(prompts) > 1:
        val_size = 1
    
    val_prompts = prompts[:val_size]
    train_prompts = prompts[val_size:]
    
    if len(train_prompts) == 0:
        # Fallback for dummy datasets
        train_prompts = prompts
        val_prompts = prompts

    print(f"Loaded {len(train_prompts)} train prompts, {len(val_prompts)} val prompts.")

    if not config.mock_mode:
        tokenizer = AutoTokenizer.from_pretrained(config.model_id, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
    
        # Load base model + SFT LoRA adapter
        print(f"Loading base model: {config.model_id}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        print(f"Loading SFT adapter: {config.adapter_path}...")
        model = PeftModel.from_pretrained(base_model, config.adapter_path, is_trainable=True)
        
        # Enable gradient checkpointing to save VRAM
        model.base_model.model.model.gradient_checkpointing_enable()
    
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"Trainable: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.2f}%)")
    
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate)
    else:
        print("MOCK MODE: Skipping model and tokenizer loading.")
        class MockTokenizer:
            eos_token = "<|endoftext|>"
            eos_token_id = 0
            pad_token = "<|endoftext|>"
            pad_token_id = 0
            def save_pretrained(self, path): pass
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                return "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        tokenizer = MockTokenizer()
        
        # Simple mock model for parameter device tracking
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Parameter(torch.randn(10, 10))
            def forward(self, input_ids, **kwargs):
                B, S = input_ids.shape
                # return dummy logits of shape [1, S, Vocabulary]
                logits = torch.randn(B, S, 50256, device=self.dummy.device) * self.dummy.sum() * 0.0
                class Output:
                    pass
                out = Output()
                out.logits = logits
                return out
            def generate(self, *args, **kwargs): pass
            def save_pretrained(self, *args, **kwargs): pass
        model = MockModel().to(torch.device("cpu"))
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        trainable_params = list(model.parameters())

    os.makedirs(config.output_dir, exist_ok=True)

    print(
        f"\n🚀 Multi-Turn GRPO+DAPO Training\n"
        f"   group_size={config.group_size}, max_react_steps={config.max_react_steps}, "
        f"batch_size={config.batch_size}, grpo_epochs={config.grpo_epochs}\n"
        f"   clip=[{config.cliprange_low}, {config.cliprange_high}] (DAPO Clip-Higher)\n"
        f"   reward_discount={config.reward_discount} (Kevin multi-turn γ)\n"
    )

    global_step = 0

    if not config.mock_mode:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=config.__dict__,
        )

    for epoch in range(config.num_train_epochs):
        epoch_prompts = train_prompts.copy()
        random.shuffle(epoch_prompts)

        for batch_start in range(0, len(epoch_prompts), config.batch_size):
            batch = epoch_prompts[batch_start : batch_start + config.batch_size]

            # ── Rollout: G trajectories per prompt ──────────────────────
            all_group_turns:   list[list[list[TurnData]]] = []  # [B][G][turns]
            all_group_rewards: list[list[float]] = []            # [B][G]

            n_degenerate = 0
            for p_idx, prompt_text in enumerate(batch):
                print(f"\n[Prompt {p_idx+1}/{len(batch)}] Generating {config.group_size} trajectories (batched/parallel)...")
                group_turns, group_rewards = _run_group_episodes(prompt_text, model, tokenizer, config)

                # Dynamic Sampling: if all rewards are identical, resample (DAPO)
                if config.dynamic_sampling:
                    for attempt in range(1, config.max_resample_attempts):
                        reward_std = torch.tensor(group_rewards).std().item()
                        if reward_std > 1e-4:
                            break
                        n_degenerate += 1
                        print(f"  [Dynamic Sampling] Degenerate group (std={reward_std:.6f}), resampling attempt {attempt+1}/{config.max_resample_attempts}...")
                        group_turns, group_rewards = _run_group_episodes(prompt_text, model, tokenizer, config)

                all_group_turns.append(group_turns)
                all_group_rewards.append(group_rewards)

                mean_r = sum(group_rewards) / len(group_rewards)
                best_r = max(group_rewards)
                reward_std = torch.tensor(group_rewards).std().item()
                print(f"  Group: mean={mean_r:.2f}, best={best_r:.2f}, std={reward_std:.3f}")

            batch_mean = sum(
                sum(rs) / len(rs) for rs in all_group_rewards
            ) / len(all_group_rewards)
            print(f"\n[Epoch {epoch+1} | Step {global_step+1}] batch_mean_reward={batch_mean:.3f}")

            torch.cuda.empty_cache()

            # ── Collect old log probs (before gradient updates) ─────────
            model.eval()
            all_old_lps: list[list[list[torch.Tensor]]] = []  # [B][G][turns]

            for group_turns in all_group_turns:
                group_old = []
                for turns in group_turns:
                    turn_lps = []
                    for ctx, resp in turns:
                        if len(resp) == 0:
                            turn_lps.append(torch.zeros(0))
                        else:
                            with torch.no_grad():
                                lp = _get_token_log_probs(model, ctx, resp)
                            turn_lps.append(lp.detach().cpu())
                    group_old.append(turn_lps)
                all_old_lps.append(group_old)
            model.train()

            # ── GRPO epochs ─────────────────────────────────────────────
            for grpo_ep in range(config.grpo_epochs):
                optimizer.zero_grad()
                total_loss_val = 0.0

                for group_turns, group_rewards, group_old_lps in zip(
                    all_group_turns, all_group_rewards, all_old_lps
                ):
                    loss = _compute_grpo_loss(
                        model, group_turns, group_rewards, group_old_lps, config
                    )
                    loss.backward()
                    total_loss_val += loss.item()

                torch.nn.utils.clip_grad_norm_(trainable_params, config.max_grad_norm)
                optimizer.step()
                
                if not config.mock_mode:
                    # Compute mean policy entropy from old log probs as collapse indicator
                    all_lp_vals = [
                        lp for group in all_old_lps
                        for turn_lps in group
                        for lp in turn_lps
                        if len(lp) > 0
                    ]
                    mean_entropy = float(-torch.cat(all_lp_vals).mean()) if all_lp_vals else 0.0

                    wandb.log({
                        "train/loss": total_loss_val,
                        "train/batch_mean_reward": batch_mean,
                        "train/learning_rate": config.learning_rate,
                        "train/policy_entropy": mean_entropy,
                        "train/degenerate_groups": n_degenerate,
                        "epoch": epoch + (global_step / max(1, len(train_prompts) // config.batch_size))
                    }, step=global_step)

                print(f"  [GRPO epoch {grpo_ep+1}/{config.grpo_epochs}] loss={total_loss_val:.4f}")

            global_step += 1
            
            if global_step % config.eval_steps == 0 or global_step == 1:
                eval_metrics = _run_evaluation(model, tokenizer, config, val_prompts)
                if not config.mock_mode:
                    wandb.log(eval_metrics, step=global_step)

            if not config.mock_mode and global_step % config.save_steps == 0:
                ckpt = f"{config.output_dir}/step_{global_step}"
                model.save_pretrained(ckpt)
                tokenizer.save_pretrained(ckpt)
                print(f"  Checkpoint → {ckpt}")

    if not config.mock_mode:
        model.save_pretrained(config.output_dir)
        tokenizer.save_pretrained(config.output_dir)
        wandb.finish()
    print(f"\n✅ Training complete. Model saved to {config.output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="KernelForge Multi-Turn GRPO+DAPO")
    parser.add_argument("--dataset", type=str, default="../sft/rl_prompts.jsonl")
    parser.add_argument("--output_dir", type=str, default="checkpoints/kernelforge_grpo")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--adapter", type=str, default="../sft/sft_qwen3_14b_lora")
    parser.add_argument("--max_react_steps", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--grpo_epochs", type=int, default=2, help="Gradient updates per rollout (2 reduces staleness vs original 4)")
    parser.add_argument("--wandb_project", type=str, default="kernelforge-rl")
    parser.add_argument("--wandb_name", type=str, default="grpo-qwen-14b")
    parser.add_argument("--resume", action="store_true", help="Resume from output_dir if it exists")
    parser.add_argument("--mock_mode", action="store_true")
    parser.add_argument("--no_dynamic_sampling", action="store_true", help="Disable DAPO dynamic sampling")
    args = parser.parse_args()

    cfg = GRPOConfig(
        model_id=args.model,
        adapter_path=args.adapter,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        max_react_steps=args.max_react_steps,
        group_size=args.group_size,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        grpo_epochs=args.grpo_epochs,
        mock_mode=args.mock_mode,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_name,
        dynamic_sampling=not args.no_dynamic_sampling,
    )
    
    # Simple resume logic: swap base model for output_dir if resuming
    if args.resume and os.path.exists(args.output_dir):
        # find the highest step checkpoint, or output_dir itself
        import glob
        ckpts = glob.glob(f"{args.output_dir}/step_*")
        if ckpts:
            latest = sorted(ckpts, key=lambda x: int(x.split("_")[-1]))[-1]
            print(f"Resuming from checkpoint: {latest}")
            cfg.model_id = latest
            cfg.adapter_path = "" # weights are already merged in checkpoint
        else:
            print(f"Resuming from output dir: {args.output_dir}")
            cfg.model_id = args.output_dir
            cfg.adapter_path = ""

    train(cfg)
