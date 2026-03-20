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

import math
import time
import torch
from tqdm import tqdm
import torch.nn.functional as F
import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
_MP_SPAWN_CTX = multiprocessing.get_context("spawn")

# SGLang imports — optional, falls back to model.generate() if not installed
try:
    import sglang as sgl
    from sglang import RuntimeEndpoint
    SGLANG_AVAILABLE = True
except ImportError:
    SGLANG_AVAILABLE = False

from agent import _extract_cuda_code
from profiler import profile_kernel
from reward import calculate_reward
from sandbox import evaluate
from sys_prompt import get_system_prompt


def _worker_run_eval(args):
    cand, prompt_text = args
    if cand is None: return None
    return evaluate(cand, prompt_text)


def _extract_python_block(text: str) -> str:
    """Extract the first ```python ... ``` block from model output."""
    import re
    m = re.search(r'```python\s*(.*?)```', text, re.DOTALL)
    return m.group(1).strip() if m else ""


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
    group_size: int = 16              # G trajectories per prompt

    # GRPO + DAPO hyperparameters
    grpo_epochs: int = 2              # gradient updates per batch
    batch_size: int = 4               # prompts per batch
    learning_rate: float = 1e-6
    warmup_steps: int = 10            # cosine schedule warmup (noisy advantages early in training)
    cliprange_low: float = 0.2        # standard lower clip
    cliprange_high: float = 0.28      # DAPO Clip-Higher (asymmetric)
    max_grad_norm: float = 1.0

    # Generation
    max_new_tokens: int = 3000
    temperature: float = 0.3
    mock_mode: bool = False

    # SGLang server-mode generation (faster than model.generate())
    # SGLang MUST be installed in a separate venv to avoid dependency conflicts.
    # Set SGLANG_PYTHON=/path/to/sglang_env/bin/python before running, or pass --sglang_python.
    use_sglang: bool = False
    sglang_python: str = ""      # path to SGLang venv python; falls back to $SGLANG_PYTHON
    sglang_port: int = 30000
    sglang_tp: int = 1           # tensor parallel degree (set to GPU count for multi-GPU)

    # Training
    num_train_epochs: int = 1
    save_steps: int = 50
    eval_steps: int = 50
    wandb_project: str = "kernelforge-rl"
    wandb_run_name: str = "grpo-qwen-14b"

    # Reward shaping (graduated — creates gradient signal at every failure stage)
    reward_no_code: float = -1.0      # no ```python block found at all
    reward_compile_fail: float = -0.5 # code found but fails to compile or wrap
    reward_wrong_output: float = -0.1 # compiles but produces wrong outputs
    reward_correct_base: float = 0.3  # base bonus for any correct kernel (Kevin's approach)

    # Length penalty — scaled to be minor relative to reward signal
    # At 3000 tokens: -0.0001 * 3000 = -0.3 (won't swamp the correctness signal)
    length_penalty_coef: float = 0.0001

    # Entropy bonus — prevents entropy collapse (critical for coding tasks)
    # A small positive coefficient adds H(π) to the objective, keeping exploration alive.
    # Dr. Kernel and DAPO both recommend ~0.01-0.05 for code generation.
    entropy_coef: float = 0.02

    # Curriculum: weight sampling toward easier tasks (lower level_id) early in training
    # Set to True to use level_id field from dataset if available
    curriculum: bool = True

    # Dynamic Sampling: skip degenerate groups where all rewards are identical
    dynamic_sampling: bool = True
    max_resample_attempts: int = 3


# ---------------------------------------------------------------------------
# Format helpers (must match agent.py)
# ---------------------------------------------------------------------------

PREFILL = "```python\nimport torch\n"

FORMAT_EXAMPLE = """\
Here is an example of the expected output format:

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = \"\"\"
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void add_kernel(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] + b[idx];
}

torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b) {
    auto out = torch::empty_like(a);
    int n = a.numel();
    add_kernel<<<(n+255)/256, 256>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &add_cuda, "add");
}
\"\"\"

ext = load_inline(
    name="add_ext",
    cpp_sources="torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b);",
    cuda_sources=cuda_source,
    functions=["add_cuda"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, a, b):
        return ext.add_cuda(a, b)
```

Now write the complete model_new.py for the following operation:
"""




# ---------------------------------------------------------------------------
# SGLang server helpers
# ---------------------------------------------------------------------------

_sglang_server = None  # global handle so we can shut it down at exit
_weight_sync_group = None  # NCCL process group for weight sync
_NCCL_MASTER_PORT = 65501  # separate port from SGLang HTTP port


def launch_sglang_server(model_path: str, adapter_path: str, port: int, tp: int,
                         sglang_python: str = None):
    """
    Launch SGLang as an inference server in a subprocess.
    Uses LoRA-merged weights so the server sees the SFT-initialized model.

    SGLang must be installed in a separate venv to avoid dependency conflicts
    with the training stack. Pass sglang_python to point at that environment's
    Python, e.g. /root/sglang_env/bin/python. Defaults to the env variable
    SGLANG_PYTHON, then falls back to the current interpreter (not recommended).

    Returns the server process handle.
    """
    import subprocess, sys, time, requests, os

    # Resolve which Python to use for the SGLang subprocess
    python_bin = (
        sglang_python
        or os.environ.get("SGLANG_PYTHON")
        or sys.executable
    )
    if python_bin == sys.executable:
        print("[SGLang] WARNING: using training venv Python for SGLang server. "
              "Set SGLANG_PYTHON=/path/to/sglang_env/bin/python to avoid "
              "dependency conflicts.")

    # Merge LoRA into base weights and save to a temp dir for SGLang to load
    merged_path = os.path.join(os.path.dirname(adapter_path), "_sglang_merged")
    if not os.path.exists(merged_path):
        print(f"[SGLang] Merging LoRA into base model for server launch → {merged_path}")
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        base = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
        merged = PeftModel.from_pretrained(base, adapter_path).merge_and_unload()
        merged.save_pretrained(merged_path)
        AutoTokenizer.from_pretrained(model_path, trust_remote_code=True).save_pretrained(merged_path)
        # Fix Qwen3 tokenizer_config.json: extra_special_tokens saved as list, SGLang needs dict
        import json as _json
        tok_cfg_path = os.path.join(merged_path, "tokenizer_config.json")
        with open(tok_cfg_path) as _f:
            tok_cfg = _json.load(_f)
        if isinstance(tok_cfg.get("extra_special_tokens"), list):
            tok_cfg["extra_special_tokens"] = {}
            with open(tok_cfg_path, "w") as _f:
                _json.dump(tok_cfg, _f, indent=2)
        del base, merged
        torch.cuda.empty_cache()
        print(f"[SGLang] Merge complete.")

    cmd = [
        python_bin, "-m", "sglang.launch_server",
        "--model-path", merged_path,
        "--port", str(port),
        "--tp", str(tp),
        "--dtype", "bfloat16",
        "--trust-remote-code",
        "--mem-fraction-static", "0.4",  # training model ~42GB, SGLang gets 40% of 95GB = ~38GB
        "--log-level", "error",
    ]
    env = {**__import__("os").environ, "SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK": "1"}
    proc = subprocess.Popen(cmd, env=env)

    # Wait for server to be ready
    url = f"http://localhost:{port}/health"
    for _ in range(120):
        try:
            if requests.get(url, timeout=2).status_code == 200:
                print(f"[SGLang] Server ready on port {port}")
                return proc
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"SGLang server failed to start on port {port}")


def init_weight_sync_group(port: int, tp_size: int = 1) -> bool:
    """
    Initialize NCCL process group for weight sync between trainer and SGLang.
    This is the correct cross-process approach used by verl/OpenRLHF/TorchRL.

    - Trainer joins as rank 0
    - SGLang worker(s) join as ranks 1..tp_size
    - NCCL handles the actual tensor transfer — no serialization needed
    """
    import requests, time
    import torch.distributed as dist

    global _weight_sync_group

    if dist.is_initialized():
        print("[SGLang] torch.distributed already initialized — skipping NCCL group init.")
        return False

    master_addr = "localhost"
    master_port = _NCCL_MASTER_PORT
    world_size = tp_size + 1  # trainer + SGLang worker(s)

    # Tell SGLang to join the NCCL group (non-blocking on SGLang side)
    try:
        resp = requests.post(
            f"http://localhost:{port}/init_weights_update_group",
            json={
                "master_address": master_addr,
                "master_port": master_port,
                "rank_offset": 1,          # SGLang workers take ranks 1..tp_size
                "world_size": world_size,
                "group_name": "kf_weight_sync",
                "backend": "nccl",
            },
            timeout=120,
        )
        if resp.status_code != 200:
            print(f"[SGLang] init_weights_update_group failed: {resp.status_code} {resp.text[:200]}")
            return False
    except Exception as e:
        print(f"[SGLang] init_weights_update_group error: {e}")
        return False

    # Give SGLang time to start its NCCL init before we rendezvous
    time.sleep(5)

    # Trainer joins as rank 0 — this blocks until SGLang also connects
    try:
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_addr}:{master_port}",
            world_size=world_size,
            rank=0,
        )
        _weight_sync_group = dist.GroupMember.WORLD
        print(f"[SGLang] NCCL weight sync group initialized (world_size={world_size})")
        return True
    except Exception as e:
        print(f"[SGLang] NCCL group init failed: {e}")
        return False


def sync_weights_to_sglang(model, port: int):
    """
    Push updated LoRA-merged weights to SGLang via NCCL broadcast.

    Correct cross-process approach (used by verl, OpenRLHF, TorchRL):
      1. Signal SGLang via HTTP with parameter name/dtype/shape
      2. NCCL broadcast the merged tensor from trainer (rank 0) to SGLang (rank 1)
      3. Flush KV cache so stale cached prompts are invalidated

    No serialization, no pickle, no size limit.
    """
    import requests
    import torch.distributed as dist

    global _weight_sync_group
    if _weight_sync_group is None or not dist.is_initialized():
        print("[SGLang] NCCL group not initialized — skipping weight sync.")
        return

    params_to_sync = []
    for name, module in model.named_modules():
        if not (hasattr(module, 'lora_A') and hasattr(module, 'lora_B')
                and 'default' in getattr(module, 'lora_A', {})):
            continue
        if not hasattr(module, 'weight') or module.weight is None:
            continue
        try:
            with torch.no_grad():
                lora_A = module.lora_A['default'].weight
                lora_B = module.lora_B['default'].weight
                s = module.scaling
                scaling = s['default'] if isinstance(s, dict) else float(s)
                delta = (lora_B.float() @ lora_A.float()) * scaling
                # Keep on GPU for NCCL broadcast
                merged = (module.weight.data.float() + delta).contiguous().cuda()

            param_name = name + ".weight"
            for pfx in ("base_model.model.", "base_model."):
                if param_name.startswith(pfx):
                    param_name = param_name[len(pfx):]
                    break
            params_to_sync.append((param_name, merged))
        except Exception as e:
            print(f"[SGLang] Merge error ({name}): {e}")

    n_synced = 0
    for param_name, tensor in params_to_sync:
        try:
            # Signal SGLang to prepare to receive this parameter
            resp = requests.post(
                f"http://localhost:{port}/update_weights_from_distributed",
                json={
                    "name": param_name,
                    "dtype": "float32",
                    "shape": list(tensor.shape),
                },
                timeout=30,
            )
            if resp.status_code != 200:
                print(f"[SGLang] update signal failed ({param_name}): {resp.status_code}")
                continue
            # NCCL broadcast: trainer rank 0 → SGLang rank 1
            dist.broadcast(tensor, src=0, group=_weight_sync_group)
            torch.cuda.synchronize()
            n_synced += 1
        except Exception as e:
            print(f"[SGLang] Sync error ({param_name}): {e}")
            break

    try:
        requests.post(f"http://localhost:{port}/flush_cache", timeout=30)
    except Exception:
        pass

    print(f"[SGLang] Weight sync: {n_synced}/{len(params_to_sync)} params via NCCL")


def _generate_with_sglang(context_texts: list[str], config: "GRPOConfig") -> list[str]:
    """
    Generate completions for a batch of prompts via the SGLang server.
    Sends all prompts in one batch request — SGLang processes them in parallel.
    RadixAttention automatically caches the shared system prompt prefix.
    """
    import requests
    payload = {
        "text": context_texts,
        "sampling_params": {
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "top_p": 1.0 if config.temperature == 0.0 else 0.9,
            "stop": ["<|im_end|>"],
        },
    }
    resp = requests.post(
        f"http://localhost:{config.sglang_port}/generate",
        json=payload,
        timeout=300,
    )
    resp.raise_for_status()
    result = resp.json()
    # SGLang returns a list when input is a list
    if isinstance(result, list):
        return [r["text"] for r in result]
    return [result["text"]]


# ---------------------------------------------------------------------------
# Single-turn episode (standard GRPO — no ReAct loop)
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
    Generate G trajectories for one prompt, evaluate in parallel, return rewards.

    Returns:
      group_turns:   list[G] of single-element turn lists [(ctx_ids, resp_ids)]
      group_rewards: list[G] of scalar rewards
    """
    G = config.group_size

    user_msg = f"{FORMAT_EXAMPLE}Reference Program:\n```python\n{prompt_text}\n```"
    prompt_str = (
        get_system_prompt()
        + f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        + f"<|im_start|>assistant\n{PREFILL}"
    )
    context_texts = [prompt_str] * G

    # 1. Generate G completions
    if not config.mock_mode:
        if config.use_sglang and (SGLANG_AVAILABLE or config.sglang_python):
            t_gen = time.time()
            print(f"  Generating {G} responses...", end=" ", flush=True)
            raw_completions = _generate_with_sglang(context_texts, config)
            print(f"done ({time.time()-t_gen:.1f}s)")
            generated_texts = []
            for ctx, full in zip(context_texts, raw_completions):
                completion = full[len(ctx):] if full.startswith(ctx) else full
                generated_texts.append(PREFILL + completion)
        else:
            tokenizer.padding_side = "left"
            inputs = tokenizer(context_texts, return_tensors="pt", padding=True)
            input_ids_tensor = inputs.input_ids.to(next(model.parameters()).device)
            attention_mask = inputs.attention_mask.to(next(model.parameters()).device)
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids_tensor,
                    attention_mask=attention_mask,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated_texts = [
                PREFILL + tokenizer.decode(outputs[i][input_ids_tensor[i].shape[0]:], skip_special_tokens=True)
                for i in range(G)
            ]
    else:
        generated_texts = [
            "```python\nimport torch\nimport torch.nn as nn\nclass ModelNew(nn.Module):\n    def forward(self, a): return a\n```"
        ] * G

    # 2. Tokenize context + response for GRPO loss computation
    turns_list: list[list[TurnData]] = []
    for i, gen_text in enumerate(generated_texts):
        if config.mock_mode:
            ctx_ids = torch.tensor([1, 2, 3])
            resp_ids = torch.tensor([4, 5, 6])
        else:
            ctx_ids = tokenizer(context_texts[i], return_tensors="pt").input_ids[0]
            resp_ids = tokenizer(gen_text, return_tensors="pt").input_ids[0]
        turns_list.append([(ctx_ids.cpu(), resp_ids.cpu())])

    # 3. Extract complete model_new.py from model output — no wrapper needed
    candidates = []
    for i, gen_text in enumerate(generated_texts):
        model_new_py = _extract_python_block(gen_text)
        if i < 2:
            print(f"  [CODE DUMP traj={i}]:\n{(model_new_py or gen_text)[:800]}")
        candidates.append(model_new_py if model_new_py else None)

    # 4. Parallel evaluation
    n_valid = sum(1 for c in candidates if c is not None)
    t_eval = time.time()
    print(f"  Evaluating {n_valid}/{G} valid kernels...", end=" ", flush=True)
    with ProcessPoolExecutor(max_workers=min(G, 16), mp_context=_MP_SPAWN_CTX) as pool:
        eval_results = list(pool.map(
            _worker_run_eval,
            [(c, prompt_text) for c in candidates]
        ))
    n_compiled = sum(1 for r in eval_results if r is not None and r.get("compiles", r.get("correct", False)))
    print(f"done ({time.time()-t_eval:.1f}s) | compiled={n_compiled}/{n_valid}")

    for i, res in enumerate(eval_results):
        if res and not res.get("compiles", False) and res.get("compiler_error"):
            print(f"  [COMPILE ERROR traj={i}]: {res['compiler_error'][:400]}")

    # 5. Compute rewards
    group_rewards = []
    for i, eval_res in enumerate(eval_results):
        gen_len = len(turns_list[i][0][1])
        length_penalty = -config.length_penalty_coef * gen_len  # max ~-0.3 at 3000 tokens

        if candidates[i] is None:
            group_rewards.append(config.reward_no_code + length_penalty)
        elif eval_res is None or not eval_res.get("compiles", False):
            group_rewards.append(config.reward_compile_fail + length_penalty)
        elif not eval_res["correct"]:
            group_rewards.append(config.reward_wrong_output + length_penalty)
        else:
            reward = config.reward_correct_base + calculate_reward(eval_res) + length_penalty
            print(f"    ✅ Traj {i}: {eval_res['runtime_ms']:.3f}ms reward={reward:.3f}")
            group_rewards.append(reward)

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
    3. Entropy bonus: -entropy_coef * H(π) prevents entropy collapse
    """
    G = len(group_rewards)
    rewards = torch.tensor(group_rewards, dtype=torch.float32)

    # Group-relative advantage normalization (the core GRPO idea)
    mean_r = rewards.mean()
    std_r = rewards.std() + 1e-8
    advantages = (rewards - mean_r) / std_r  # [G]

    device = next(model.parameters()).device
    token_losses = []
    entropy_terms = []

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

            # Token-level policy loss
            token_losses.append(-torch.min(ratio * float(A_i), clipped * float(A_i)))

            # Entropy: H(π) = -E[log π] = -mean(new_lp)
            # Maximizing entropy = adding -entropy_coef * (-new_lp) = entropy_coef * new_lp
            # We subtract from loss (loss -= entropy_coef * H) = loss += entropy_coef * new_lp
            if config.entropy_coef > 0:
                entropy_terms.append(-new_lp)  # -log π per token (positive entropy)

    if not token_losses:
        return torch.tensor(0.0, device=device, requires_grad=True)

    policy_loss = torch.cat(token_losses).mean()

    if entropy_terms and config.entropy_coef > 0:
        # entropy = mean(-log π); we want to maximize it → subtract from loss
        entropy = torch.cat(entropy_terms).mean()
        return policy_loss - config.entropy_coef * entropy

    return policy_loss


def _run_evaluation(model, tokenizer, config: GRPOConfig, val_prompts: list[str]) -> dict:
    """Run an evaluation pass on the validation set."""
    print(f"\n--- Running Evaluation ({len(val_prompts)} prompts) ---")
    if config.mock_mode:
        return {"eval/pass_rate": 0.5, "eval/valid_rate": 0.8, "eval/avg_speedup": 1.1}

    model.eval()
    candidates_to_eval = []

    with torch.no_grad():
        for prompt_text in val_prompts:
            # Build prompt in exact SFT/rollout format
            user_msg = f"{FORMAT_EXAMPLE}Reference Program:\n```python\n{prompt_text}\n```"
            prompt_str = (
                get_system_prompt()
                + f"<|im_start|>user\n{user_msg}<|im_end|>\n"
                + f"<|im_start|>assistant\n{PREFILL}"
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

            model_new_py = _extract_python_block(PREFILL + response_text)
            candidates_to_eval.append((model_new_py or None, prompt_text))

    # Parallel eval over generated candidates
    success = 0
    valid = 0
    total_speedup = 0.0

    with ProcessPoolExecutor(max_workers=min(16, len(val_prompts)), mp_context=_MP_SPAWN_CTX) as pool:
        eval_results = list(pool.map(_worker_run_eval, candidates_to_eval))
        
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
    rows = list(raw)

    # Curriculum: sort by level_id (easy first) if available and enabled
    if config.curriculum and "level_id" in rows[0]:
        rows.sort(key=lambda r: (r.get("level_id", 99), random.random()))
        print(f"Curriculum enabled: sorted {len(rows)} tasks by level_id (easy-first).")
    else:
        random.shuffle(rows)

    prompts = [row["pytorch_code"] for row in rows]
    # Split to train/val (10% val, max 20)
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
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="kernels-community/flash-attn2",
        )
        print(f"Loading SFT adapter: {config.adapter_path}...")
        model = PeftModel.from_pretrained(base_model, config.adapter_path, is_trainable=True)
        
        # Enable gradient checkpointing to save VRAM
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    
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

    # Cosine LR schedule with linear warmup — stabilizes early steps when advantages are noisy
    total_steps = max(1, (len(train_prompts) // config.batch_size) * config.num_train_epochs)
    warmup_steps = config.warmup_steps

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))  # floor at 10% of peak lr

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(
        f"\n🚀 GRPO+DAPO Training\n"
        f"   group_size={config.group_size}, batch_size={config.batch_size}, grpo_epochs={config.grpo_epochs}\n"
        f"   clip=[{config.cliprange_low}, {config.cliprange_high}] (DAPO Clip-Higher)\n"
    )

    global_step = 0

    # Launch SGLang server if enabled
    if not config.mock_mode and config.use_sglang:
        if not SGLANG_AVAILABLE and not config.sglang_python:
            raise RuntimeError("use_sglang=True but sglang is not installed. Run: pip install sglang")
        import atexit
        global _sglang_server
        _sglang_server = launch_sglang_server(
            config.model_id, config.adapter_path, config.sglang_port, config.sglang_tp,
            sglang_python=config.sglang_python or None,
        )
        atexit.register(lambda: _sglang_server.terminate() if _sglang_server else None)
        print(f"[SGLang] Server running on port {config.sglang_port} (TP={config.sglang_tp})")
        # Initialize NCCL group for weight sync (correct cross-process approach)
        init_weight_sync_group(config.sglang_port, tp_size=config.sglang_tp)

    if not config.mock_mode:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=config.__dict__,
        )

    total_steps = config.num_train_epochs * (len(train_prompts) // config.batch_size)
    step_bar = tqdm(total=total_steps, desc="GRPO", unit="step", dynamic_ncols=True)

    for epoch in range(config.num_train_epochs):
        epoch_prompts = train_prompts.copy()
        random.shuffle(epoch_prompts)

        for batch_start in range(0, len(epoch_prompts), config.batch_size):
            batch = epoch_prompts[batch_start : batch_start + config.batch_size]

            # ── Rollout: G trajectories per prompt ──────────────────────
            all_group_turns:   list[list[list[TurnData]]] = []  # [B][G][turns]
            all_group_rewards: list[list[float]] = []            # [B][G]

            n_degenerate = 0
            t0 = time.time()
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
            elapsed = time.time() - t0
            step_bar.set_postfix(reward=f"{batch_mean:.3f}", step_time=f"{elapsed:.0f}s")
            step_bar.update(1)
            print(f"\n[Epoch {epoch+1} | Step {global_step+1}/{total_steps}] reward={batch_mean:.3f}  step_time={elapsed:.0f}s")

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

            # One scheduler step per batch step (not per grpo_epoch) so LR
            # schedule completes in the expected number of training steps.
            scheduler.step()

            # Sync updated weights to SGLang via NCCL broadcast
            if not config.mock_mode and config.use_sglang:
                sync_weights_to_sglang(model, config.sglang_port)

            # Log metrics and print loss (always, not just when using SGLang)
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
                    "train/mean_entropy": mean_entropy,  # track entropy — collapse = this dropping
                    "train/degenerate_groups": n_degenerate,
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "epoch": epoch + (global_step / max(1, len(train_prompts) // config.batch_size))
                }, step=global_step)

            print(f"  [Step {global_step+1}] loss={total_loss_val:.4f}")

            global_step += 1
            
            if global_step % config.eval_steps == 0:
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

    parser = argparse.ArgumentParser(description="KernelForge GRPO+DAPO")
    parser.add_argument("--dataset", type=str, default="../sft/rl_prompts.jsonl")
    parser.add_argument("--output_dir", type=str, default="checkpoints/kernelforge_grpo")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--adapter", type=str, default="../sft/sft_qwen3_14b_lora")
    parser.add_argument("--group_size", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--grpo_epochs", type=int, default=2, help="Gradient updates per rollout (2 reduces staleness vs original 4)")
    parser.add_argument("--wandb_project", type=str, default="kernelforge-rl")
    parser.add_argument("--wandb_name", type=str, default="grpo-qwen-14b")
    parser.add_argument("--resume", action="store_true", help="Resume from output_dir if it exists")
    parser.add_argument("--mock_mode", action="store_true")
    parser.add_argument("--no_dynamic_sampling", action="store_true", help="Disable DAPO dynamic sampling")
    parser.add_argument("--use_sglang", action="store_true", help="Use SGLang server for generation (faster)")
    parser.add_argument("--sglang_python", type=str, default="", help="Path to SGLang venv python (e.g. /root/sglang_env/bin/python). Can also set SGLANG_PYTHON env var.")
    parser.add_argument("--sglang_port", type=int, default=30000)
    parser.add_argument("--sglang_tp", type=int, default=1, help="SGLang tensor parallel degree")
    args = parser.parse_args()

    cfg = GRPOConfig(
        model_id=args.model,
        adapter_path=args.adapter,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        group_size=args.group_size,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        grpo_epochs=args.grpo_epochs,
        mock_mode=args.mock_mode,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_name,
        dynamic_sampling=not args.no_dynamic_sampling,
        use_sglang=args.use_sglang,
        sglang_python=args.sglang_python,
        sglang_port=args.sglang_port,
        sglang_tp=args.sglang_tp,
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
