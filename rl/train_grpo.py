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
    # Strip Qwen3 think blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    m = re.search(r'```python\s*(.*?)```', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: unclosed block (truncated by max_new_tokens)
    m = re.search(r'```python\s*(.*)', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


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
    learning_rate: float = 2e-6
    warmup_steps: int = 10            # cosine schedule warmup (noisy advantages early in training)
    cliprange_low: float = 0.2        # standard lower clip
    cliprange_high: float = 0.28      # DAPO Clip-Higher (asymmetric)
    max_grad_norm: float = 0.05

    # Multi-turn (Kevin's recipe): T refinement turns per trajectory
    # γ=0.4 discounts later turns so getting it right on turn 1 is worth more.
    # Kevin's ablation explicitly found 0.4 > 0.8 for CUDA kernel RL.
    num_turns: int = 4
    gamma: float = 0.4

    # Generation
    max_new_tokens: int = 6000        # total budget (thinking + code)
    think_budget: int = 2000          # phase-1 thinking cap; code gets the rest
    temperature: float = 0.7
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
    reward_no_code: float = -1.0        # no ```python block found at all
    reward_compile_fail: float = -0.5   # code found but fails to compile or wrap
    reward_shape_mismatch: float = -0.25 # compiles but output shape is wrong
    reward_mostly_wrong: float = -0.1   # >90% of values wrong (fundamental algo error)
    reward_partially_wrong: float = -0.05 # 30-90% of values wrong (indexing bug)
    reward_nearly_correct: float = 0.05  # <30% of values wrong (boundary/edge case)
    reward_wrong_output: float = -0.1   # fallback when wrong_frac unavailable
    reward_correct_base: float = 0.3    # base bonus for any correct kernel (Kevin's approach)

    # Regression penalty (SCoRe): if turn T was correct and turn T+1 breaks it,
    # subtract this from the turn T+1 reward so the model has a specific gradient
    # signal for "don't regress from a working state" (not just "be correct").
    reward_regression_penalty: float = -0.3

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

PREFILL = ""

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

Now write the complete model_new.py for the following operation. End your response with:
Reflection: <2-3 sentences covering: (1) your parallelization strategy and block/grid dimensions, (2) memory optimizations used (shared memory, vectorized loads, etc.), (3) what you would try next to improve performance>
"""




# ---------------------------------------------------------------------------
# Multi-turn helpers (Kevin's recipe)
# ---------------------------------------------------------------------------

def _strip_thinking(text: str) -> str:
    """
    Strip internal <think> blocks from a prior-turn response, keeping all
    code and the Reflection line intact.

    This is the correct ReAct approach: the model's "action" (the kernel it
    wrote) must remain visible in context so subsequent turns can make targeted
    fixes rather than hallucinating a new kernel from scratch.

    Context length is handled downstream by _get_token_log_probs which
    truncates sequences exceeding MAX_SEQ_LEN (DAPO overlong handling).
    """
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def _build_turn_feedback(eval_res: dict | None) -> str:
    """
    Build the user feedback message for the next turn (Kevin's format):
    - Full error message (not truncated mid-sentence)
    - Specific, actionable ask, scaled to how close the kernel is
    """
    if eval_res is None:
        return (
            "Your previous response could not be evaluated. "
            "Please write a complete, valid CUDA kernel using load_inline()."
        )
    if not eval_res.get("compiles", False):
        err = (eval_res.get("compiler_error") or "Unknown compile error")
        import re as _re
        name_match = _re.search(r"name '(\w+)' is not defined", err)
        if name_match:
            var = name_match.group(1)
            hint = (
                f"This is a Python scoping error: '{var}' is not available at module level. "
                f"load_inline() is called when the module is imported, before any __init__ runs, "
                f"so constructor parameters like '{var}' do not exist yet. "
                f"Fix: move load_inline() inside __init__ where '{var}' is accessible, "
                f"OR remove '{var}' from the cuda_source string and pass it as a kernel argument at runtime."
            )
        else:
            hint = "Fix the compile error above."
        return (
            f"Your previous kernel failed to compile:\n{err}\n\n"
            f"{hint} End your response with:\n"
            "Reflection: <2-3 sentences: (1) what caused the error, (2) what you changed to fix it, (3) your parallelization strategy>"
        )
    if not eval_res.get("correct", False):
        err = (eval_res.get("compiler_error") or "Outputs do not match reference")
        wrong_frac = eval_res.get("wrong_frac")
        shape_ok = eval_res.get("shape_ok")

        import re as _re
        if "TORCH_USE_CUDA_DSA" in err or "device-side assert" in err.lower():
            err = (
                "CUDA out-of-bounds memory access: a thread read/wrote past array bounds. "
                "Common causes: (1) threadIdx.y used as batch index but blockDim.y > batch_size — "
                "threads with threadIdx.y >= batch_size go out of bounds; "
                "(2) shared memory loaded by only threadIdx.x threads but indexed up to in_features; "
                "(3) output index calculation does not match tensor layout [batch, out_features]. "
                "Fix: ensure every thread index stays within the tensor dimensions it accesses. "
                "If you changed from a flat 1D kernel to a 2D block layout, verify blockDim.x/y "
                "and your index formula match the new launch configuration."
            )

        # Shape wrong — the algorithm may be fine, only dimensions need fixing
        if shape_ok is False:
            return (
                f"Your previous kernel compiled but produced incorrect outputs:\n{err}\n\n"
                "Fix the correctness issue. End your response with:\n"
                "Reflection: <2-3 sentences: (1) what was wrong in your previous kernel, (2) what you changed to fix it, (3) your parallelization strategy>"
            )

        # Nearly correct (<30% wrong) — small fix, don't rewrite
        if wrong_frac is not None and wrong_frac < 0.30:
            return (
                f"Your previous kernel compiled but produced incorrect outputs:\n{err}\n\n"
                "Fix the correctness issue. End your response with:\n"
                "Reflection: <2-3 sentences: (1) what was wrong in your previous kernel, (2) what you changed to fix it, (3) your parallelization strategy>"
            )

        # Partially wrong (30-90%) — indexing bug
        if wrong_frac is not None and wrong_frac < 0.90:
            return (
                f"Your previous kernel compiled but produced incorrect outputs:\n{err}\n\n"
                "Fix the correctness issue. End your response with:\n"
                "Reflection: <2-3 sentences: (1) what was wrong in your previous kernel, (2) what you changed to fix it, (3) your parallelization strategy>"
            )

        # Mostly wrong (>90%) — fundamental error, rewrite
        if "FUNDAMENTAL ALGORITHMIC ERROR" in err:
            return (
                f"Your previous kernel compiled but produced incorrect outputs:\n{err}\n\n"
                "Rewrite the kernel completely from scratch — do not modify the existing code. "
                "Study the reference implementation carefully before writing. End your response with:\n"
                "Reflection: <2-3 sentences: (1) what was fundamentally wrong, (2) what algorithm you are using in the rewrite, (3) your parallelization strategy>"
            )
        return (
            f"Your previous kernel compiled but produced incorrect outputs:\n{err}\n\n"
            "Fix the correctness issue. End your response with:\n"
            "Reflection: <2-3 sentences: (1) what was wrong in your previous kernel, (2) what you changed to fix it, (3) your parallelization strategy>"
        )

    rt = eval_res.get("runtime_ms")
    bt = eval_res.get("baseline_runtime_ms")
    if rt and bt:
        speedup = bt / rt
        return (
            f"Your previous kernel was correct at {speedup:.2f}x speedup over PyTorch. "
            "Try to optimize it further — shared memory, vectorized loads, better parallelism, tuned block size. "
            "IMPORTANT: preserve correctness. Only submit if your optimized version still produces correct outputs. "
            "End your response with:\n"
            "Reflection: <2-3 sentences: (1) what you changed vs your previous kernel, (2) why you expected it to be faster, (3) what you would try next>"
        )
    return (
        "Your previous kernel was correct. Try to optimize it for GPU performance. "
        "IMPORTANT: preserve correctness. "
        "End your response with:\n"
        "Reflection: <2-3 sentences: (1) what you changed vs your previous kernel, (2) why you expected it to be faster, (3) what you would try next>"
    )


# ---------------------------------------------------------------------------
# SGLang server helpers
# ---------------------------------------------------------------------------

_sglang_server = None  # global handle so we can shut it down at exit
_lora_temp_dir = None  # temp dir for LoRA sync between trainer and SGLang
_LORA_NAME = "kf_grpo_adapter"  # name SGLang uses for the live LoRA adapter


def launch_sglang_server(model_path: str, adapter_path: str, port: int, tp: int,
                         sglang_python: str = None):
    """
    Launch SGLang as an inference server in a subprocess using the base model
    + LoRA loaded dynamically via /load_lora_adapter (no merge needed).

    After each training step: save LoRA → /release_memory_occupation →
    backward pass → /resume_memory_occupation → /load_lora_adapter (hot-reload).
    No NCCL required.

    Returns the server process handle.
    """
    import subprocess, sys, time, requests, os

    python_bin = (
        sglang_python
        or os.environ.get("SGLANG_PYTHON")
        or sys.executable
    )
    if python_bin == sys.executable:
        print("[SGLang] WARNING: using training venv Python for SGLang server. "
              "Set SGLANG_PYTHON=/path/to/sglang_env/bin/python to avoid "
              "dependency conflicts.")

    cmd = [
        python_bin, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--port", str(port),
        "--tp", str(tp),
        "--dtype", "bfloat16",
        "--trust-remote-code",
        "--enable-lora",
        "--lora-paths", f"{_LORA_NAME}={adapter_path}",
        "--max-loras-per-batch", "1",
        "--mem-fraction-static", "0.45",
        "--context-length", "16384",
        "--log-level", "error",
    ]

    import glob as _glob
    cuda_homes = sorted(_glob.glob("/usr/local/cuda-*"), reverse=True) + ["/usr/local/cuda"]
    cuda_home = next((p for p in cuda_homes if os.path.isfile(f"{p}/bin/nvcc")), None)
    base_env = dict(os.environ)
    if cuda_home:
        cuda_bin = f"{cuda_home}/bin"
        cuda_inc = f"{cuda_home}/include"
        path = base_env.get("PATH", "")
        if cuda_bin not in path:
            base_env["PATH"] = f"{cuda_bin}:{path}"
        cpath = base_env.get("CPATH", "")
        if cuda_inc not in cpath:
            base_env["CPATH"] = f"{cuda_inc}:{cpath}"
    env = {**base_env, "SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK": "1"}
    proc = subprocess.Popen(cmd, env=env)

    # Wait for server to be ready
    url = f"http://localhost:{port}/health"
    for _ in range(120):
        try:
            if requests.get(url, timeout=2).status_code == 200:
                print(f"[SGLang] Server ready on port {port}")
                break
        except Exception:
            pass
        time.sleep(2)
    else:
        proc.terminate()
        raise RuntimeError(f"SGLang server failed to start on port {port}")

    # Initial LoRA is loaded at server startup via --lora-paths
    print(f"[SGLang] Server ready with LoRA '{_LORA_NAME}' loaded from {adapter_path}")
    return proc


def release_sglang_memory(port: int):
    """Release SGLang's GPU allocation so the trainer can use full VRAM for backward pass."""
    import requests
    try:
        r = requests.post(f"http://localhost:{port}/release_memory_occupation",
                          json={}, timeout=60)
        print(f"[SGLang] Memory released (status={r.status_code})")
    except Exception as e:
        print(f"[SGLang] release_memory_occupation failed: {e}")


def resume_sglang_memory(port: int):
    """Restore SGLang's GPU allocation after the training step is done."""
    import requests
    try:
        r = requests.post(f"http://localhost:{port}/resume_memory_occupation",
                          json={}, timeout=120)
        print(f"[SGLang] Memory resumed (status={r.status_code})")
    except Exception as e:
        print(f"[SGLang] resume_memory_occupation failed: {e}")


def sync_lora_to_sglang(model, port: int):
    """
    Save the updated LoRA adapter to a temp dir and hot-reload it in SGLang.
    Called after each optimizer step so SGLang always generates with the latest policy.
    """
    import requests, tempfile, os

    global _lora_temp_dir
    if _lora_temp_dir is None:
        import tempfile as _tmp
        _lora_temp_dir = _tmp.mkdtemp(prefix="kf_lora_sync_")

    lora_path = os.path.join(_lora_temp_dir, "adapter")
    os.makedirs(lora_path, exist_ok=True)

    # Save only the LoRA adapter weights (small — ~100MB)
    model.save_pretrained(lora_path)

    # Unload the old adapter, then load the updated one
    try:
        requests.post(f"http://localhost:{port}/unload_lora_adapter",
                      json={"lora_name": _LORA_NAME}, timeout=30)
    except Exception:
        pass  # best-effort; load will fail if still loaded

    try:
        r = requests.post(
            f"http://localhost:{port}/load_lora_adapter",
            json={"lora_name": _LORA_NAME, "lora_path": lora_path},
            timeout=60,
        )
        if r.status_code == 200:
            print(f"[SGLang] LoRA hot-reloaded (step saved → {lora_path})")
        else:
            print(f"[SGLang] load_lora_adapter: status={r.status_code} {r.text[:200]}")
    except Exception as e:
        print(f"[SGLang] load_lora_adapter failed: {e}")

    # Flush stale KV cache (entries computed with old weights are invalid)
    try:
        requests.post(f"http://localhost:{port}/flush_cache", timeout=30)
    except Exception:
        pass




def _sglang_post(port: int, contexts: list[str], max_new_tokens: int,
                 temperature: float, stop: list[str]) -> list[str]:
    """
    Raw SGLang /generate call. Returns full text (context + completion).
    If a batch request fails (e.g. context too long), falls back to
    per-request calls so one long context doesn't crash the whole batch.
    """
    import requests

    def _single(ctx):
        payload = {
            "text": ctx,
            "lora_name": _LORA_NAME,
            "sampling_params": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": 1.0 if temperature == 0.0 else 0.95,
                "stop": stop,
            },
        }
        try:
            r = requests.post(f"http://localhost:{port}/generate", json=payload, timeout=600)
            r.raise_for_status()
            res = r.json()
            return res["text"] if isinstance(res, dict) else res[0]["text"]
        except Exception as e:
            print(f"  [SGLang] single request failed (context too long?): {e}")
            return ctx  # return context unchanged → empty completion

    # Try batch first (fast path)
    payload = {
        "text": contexts,
        "lora_name": _LORA_NAME,
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": 1.0 if temperature == 0.0 else 0.95,
            "stop": stop,
        },
    }
    try:
        resp = requests.post(f"http://localhost:{port}/generate", json=payload, timeout=600)
        resp.raise_for_status()
        result = resp.json()
        if isinstance(result, list):
            return [r["text"] for r in result]
        return [result["text"]]
    except Exception:
        # Batch failed — fall back to per-request so one bad context doesn't kill all 8
        print("  [SGLang] Batch request failed, falling back to per-request mode...")
        return [_single(ctx) for ctx in contexts]


def _generate_with_sglang(context_texts: list[str], config: "GRPOConfig") -> list[str]:
    """
    Budget-forcing generation:
      Phase 1 — let the model think for up to think_budget tokens,
                 stop early if it writes ```python or </think> on its own.
      Phase 2 — if no code block started after phase 1, inject
                 </think>\n```python\n and generate the code.

    This prevents the model from spending all 6000 tokens in <think>
    and never writing any code.
    """
    think_budget = config.think_budget
    code_budget   = config.max_new_tokens - think_budget

    # ── Phase 1: thinking ───────────────────────────────────────────────────
    phase1_raw = _sglang_post(
        config.sglang_port, context_texts, think_budget, config.temperature,
        stop=["<|im_end|>", "```python"],
    )
    # phase1_raw contains the full text (context + partial completion)

    # ── Build phase-2 contexts ───────────────────────────────────────────────
    phase2_contexts = []
    phase1_completions = []   # what the model produced in phase 1 (no context prefix)

    for ctx, full in zip(context_texts, phase1_raw):
        comp = full[len(ctx):] if full.startswith(ctx) else full
        phase1_completions.append(comp)

        if "```python" in comp:
            # Model already started writing code — continue from exactly here
            idx = comp.index("```python")
            phase2_contexts.append(ctx + comp[:idx] + "```python\n")
        elif "</think>" in comp:
            # Model closed its thinking naturally — let it continue normally
            phase2_contexts.append(ctx + comp)
        else:
            # Thinking budget exhausted without closing — inject the transition
            phase2_contexts.append(ctx + comp + "\n</think>\n```python\n")

    # ── Phase 2: code ────────────────────────────────────────────────────────
    phase2_raw = _sglang_post(
        config.sglang_port, phase2_contexts, code_budget, config.temperature,
        stop=["<|im_end|>"],
    )

    # ── Combine: return context + phase1 + phase2 ────────────────────────────
    results = []
    for ctx, p2ctx, p2full in zip(context_texts, phase2_contexts, phase2_raw):
        # p2ctx already contains ctx + phase1 prefix; p2full = p2ctx + phase2 completion
        full_with_ctx = p2full if p2full.startswith(ctx) else (p2ctx + p2full)
        results.append(full_with_ctx)
    return results


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
) -> tuple[list[list[TurnData]], list[list[float]]]:
    """
    Generate G trajectories × T turns (Kevin's multi-turn recipe).

    Each turn: generate → evaluate → build feedback → build next-turn context.
    Thinking is stripped from inter-turn context so the model only sees its code + feedback.

    Returns:
      group_turns:   list[G] of list[T] of (ctx_ids, resp_ids)
      group_rewards: list[G] of list[T] of scalar rewards
    """
    G = config.group_size
    T = config.num_turns

    sys_content = get_system_prompt().replace("<|im_start|>system\n", "").replace("<|im_end|>\n", "").strip()
    user_msg = f"{FORMAT_EXAMPLE}Reference Program:\n```python\n{prompt_text}\n```"
    base_messages = [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": user_msg},
    ]

    group_turns:   list[list[TurnData]]   = [[] for _ in range(G)]
    group_rewards: list[list[float]]      = [[] for _ in range(G)]

    # Per-trajectory conversation state across turns
    traj_responses: list[list[str]]  = [[] for _ in range(G)]  # raw completions (keep <think>)
    traj_evals:     list[list[dict]] = [[] for _ in range(G)]  # eval results per turn

    # ── DEBUG: what operation is this prompt? ───────────────────────────────
    # Extract first torch call from prompt to label the operation
    op_match = re.search(r'torch\.\w+|nn\.\w+', prompt_text)
    op_label = op_match.group(0) if op_match else "unknown_op"
    prompt_lines = prompt_text.strip().count('\n') + 1
    print(f"  [DEBUG] Prompt: {op_label}, {prompt_lines} lines, {len(prompt_text)} chars")
    # ── END DEBUG ────────────────────────────────────────────────────────────

    for turn_idx in range(T):
        turn_label = f"Turn {turn_idx+1}/{T}"

        # Build context texts: base prompt + all prior turns [code + feedback].
        # Kevin, Dr. Kernel, and MURPHY all keep full history — the model needs to
        # see all prior attempts to identify which approach was best and avoid repeating mistakes.
        context_texts = []
        for i in range(G):
            msgs = list(base_messages)
            for t in range(turn_idx):
                stripped = _strip_thinking(traj_responses[i][t])
                if i == 0:
                    has_ref = "Reflection:" in stripped
                    print(f"  [DEBUG] Turn {t+1}→{turn_idx+1} traj=0: "
                          f"reflection={'YES' if has_ref else 'NO'}, stripped_len={len(stripped)} chars")
                msgs.append({"role": "assistant", "content": stripped})
                feedback = _build_turn_feedback(traj_evals[i][t])
                if i == 0:
                    print(f"  [DEBUG] Feedback traj=0 turn {t+1}→{turn_idx+1}:\n{feedback}")
                elif (traj_evals[i][t] is not None
                      and not traj_evals[i][t].get("correct", False)
                      and t == turn_idx - 1):
                    # Print feedback for first wrong traj so we see what error msg it received
                    is_first_wrong = not any(
                        traj_evals[j][t] is not None
                        and not traj_evals[j][t].get("correct", False)
                        and t == turn_idx - 1
                        for j in range(1, i)
                    )
                    if is_first_wrong:
                        print(f"  [DEBUG] Feedback traj={i} turn {t+1}→{turn_idx+1}:\n{feedback}")
                msgs.append({"role": "user", "content": feedback})
            ctx_str = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            if i == 0:
                ctx_tokens = len(tokenizer(ctx_str).input_ids)
                print(f"  [DEBUG] Context traj=0 turn {turn_idx+1}: {ctx_tokens} tokens")
            context_texts.append(ctx_str)

        # Generate G completions for this turn
        if not config.mock_mode:
            if config.use_sglang and (SGLANG_AVAILABLE or config.sglang_python):
                t_gen = time.time()
                print(f"  [{turn_label}] Generating {G} responses...", end=" ", flush=True)
                raw_completions = _generate_with_sglang(context_texts, config)
                print(f"done ({time.time()-t_gen:.1f}s)")
                completions = [
                    full[len(ctx):] if full.startswith(ctx) else full
                    for ctx, full in zip(context_texts, raw_completions)
                ]
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
                completions = [
                    tokenizer.decode(outputs[i][input_ids_tensor[i].shape[0]:], skip_special_tokens=True)
                    for i in range(G)
                ]
        else:
            completions = [
                "```python\nimport torch\nimport torch.nn as nn\nclass ModelNew(nn.Module):\n    def forward(self, a): return a\n```"
            ] * G

        # Tokenize context + response for GRPO loss
        for i, (ctx_text, gen_text) in enumerate(zip(context_texts, completions)):
            if config.mock_mode:
                ctx_ids  = torch.tensor([1, 2, 3])
                resp_ids = torch.tensor([4, 5, 6])
            else:
                ctx_ids  = tokenizer(ctx_text, return_tensors="pt").input_ids[0]
                resp_ids = tokenizer(gen_text,  return_tensors="pt").input_ids[0]
            group_turns[i].append((ctx_ids.cpu(), resp_ids.cpu()))

        # Extract code and evaluate
        candidates = []
        for i, gen_text in enumerate(completions):
            model_new_py = _extract_python_block(gen_text)
            if i == 0:  # always print traj=0 every turn so we can trace the full repair loop
                print(f"  [CODE DUMP turn={turn_idx} traj=0]:\n{(model_new_py or gen_text)}")
            candidates.append(model_new_py if model_new_py else None)

        n_valid = sum(1 for c in candidates if c is not None)
        t_eval = time.time()
        print(f"  [{turn_label}] Evaluating {n_valid}/{G} valid kernels...", end=" ", flush=True)
        with ProcessPoolExecutor(max_workers=min(G, 16), mp_context=_MP_SPAWN_CTX) as pool:
            eval_results = list(pool.map(
                _worker_run_eval,
                [(c, prompt_text) for c in candidates],
            ))
        n_compiled = sum(1 for r in eval_results if r is not None and r.get("compiles", False))
        n_correct  = sum(1 for r in eval_results if r is not None and r.get("correct",  False))
        print(f"done ({time.time()-t_eval:.1f}s) | compiled={n_compiled}/{n_valid} correct={n_correct}/{G}")

        # Compile errors (all turns)
        for i, res in enumerate(eval_results):
            if res and not res.get("compiles", False) and res.get("compiler_error"):
                print(f"  [COMPILE ERROR traj={i}]: {res['compiler_error']}")

        # First wrong (compiled-but-incorrect) trajectory: print error + code so we can see
        # exactly what the kernel computed and what the sandbox told it.
        wrong_traj = next(
            (i for i in range(G)
             if eval_results[i] is not None
             and eval_results[i].get("compiles", False)
             and not eval_results[i].get("correct", False)),
            None,
        )
        if wrong_traj is not None:
            err = eval_results[wrong_traj].get("compiler_error") or "no error detail"
            print(f"  [WRONG traj={wrong_traj} turn={turn_idx+1}] {err}")
            if candidates[wrong_traj]:
                print(f"  [CODE DUMP turn={turn_idx} traj={wrong_traj}]:\n{candidates[wrong_traj]}")

        # Compute per-turn rewards
        for i, (eval_res, gen_text) in enumerate(zip(eval_results, completions)):
            gen_len = len(group_turns[i][turn_idx][1])
            length_penalty = -config.length_penalty_coef * gen_len

            if candidates[i] is None:
                overlong_penalty = -0.5 if gen_len >= int(config.max_new_tokens * 0.95) else 0.0
                r = config.reward_no_code + length_penalty + overlong_penalty
            elif eval_res is None or not eval_res.get("compiles", False):
                r = config.reward_compile_fail + length_penalty
            elif not eval_res["correct"]:
                wrong_frac = eval_res.get("wrong_frac")
                shape_ok = eval_res.get("shape_ok")
                if shape_ok is False:
                    r = config.reward_shape_mismatch + length_penalty
                elif wrong_frac is not None and wrong_frac < 0.30:
                    r = config.reward_nearly_correct + length_penalty
                elif wrong_frac is not None and wrong_frac < 0.90:
                    r = config.reward_partially_wrong + length_penalty
                else:
                    r = config.reward_mostly_wrong + length_penalty
                # Regression penalty (SCoRe): if previous turn was correct and this one isn't,
                # add an extra penalty so the model learns specifically not to break working code.
                if turn_idx > 0 and traj_evals[i][turn_idx - 1] is not None:
                    if traj_evals[i][turn_idx - 1].get("correct", False):
                        r += config.reward_regression_penalty
            else:
                r = config.reward_correct_base + calculate_reward(eval_res) + length_penalty
                if i < 3:
                    print(f"    ✅ Turn {turn_idx+1} Traj {i}: {eval_res['runtime_ms']:.3f}ms reward={r:.3f}")

            group_rewards[i].append(r)

        # ── DEBUG: reward breakdown this turn ───────────────────────────────
        turn_rewards = [group_rewards[i][turn_idx] for i in range(G)]
        reward_categories = {"no_code": 0, "compile_fail": 0, "shape_mismatch": 0,
                             "mostly_wrong": 0, "partially_wrong": 0, "nearly_correct": 0, "correct": 0}
        for i in range(G):
            if candidates[i] is None:
                reward_categories["no_code"] += 1
            elif eval_results[i] is None or not eval_results[i].get("compiles", False):
                reward_categories["compile_fail"] += 1
            elif not eval_results[i].get("correct", False):
                wf = eval_results[i].get("wrong_frac")
                so = eval_results[i].get("shape_ok")
                if so is False:
                    reward_categories["shape_mismatch"] += 1
                elif wf is not None and wf < 0.30:
                    reward_categories["nearly_correct"] += 1
                elif wf is not None and wf < 0.90:
                    reward_categories["partially_wrong"] += 1
                else:
                    reward_categories["mostly_wrong"] += 1
            else:
                reward_categories["correct"] += 1
        print(f"  [DEBUG] Turn {turn_idx+1} rewards: {reward_categories} "
              f"| min={min(turn_rewards):.2f} max={max(turn_rewards):.2f} "
              f"std={torch.tensor(turn_rewards).std().item():.3f}")
        # ── END DEBUG ────────────────────────────────────────────────────────

        # Store for next-turn context building
        for i in range(G):
            traj_responses[i].append(completions[i])
            traj_evals[i].append(eval_results[i])

    # ── DEBUG: final reward matrix ───────────────────────────────────────────
    print(f"  [DEBUG] Reward matrix [G={G} x T={T}]:")
    for i in range(G):
        print(f"    traj {i}: {[f'{r:.2f}' for r in group_rewards[i]]}")
    # ── END DEBUG ────────────────────────────────────────────────────────────

    return group_turns, group_rewards


# ---------------------------------------------------------------------------
# GRPO core
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 8192


def _get_token_log_probs(
    model,
    context_ids: torch.Tensor,
    response_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Forward pass for one turn. Returns per-token log probs [R].
    Truncates response from the right and context from the left so total
    sequence never exceeds MAX_SEQ_LEN (prevents OOM on long think+code responses).
    With reflection-only inter-turn context (Kevin's approach), contexts stay
    small (~1000-1300 tokens across all turns) so truncation rarely triggers.
    """
    max_resp = MAX_SEQ_LEN - 64
    if len(response_ids) > max_resp:
        response_ids = response_ids[:max_resp]
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


def _compute_grpo_loss_and_backward(
    model,
    group_turns: list[list[TurnData]],
    group_rewards: list[list[float]],
    old_log_probs: list[list[torch.Tensor]],
    config: GRPOConfig,
) -> float:
    """
    GRPO loss with Kevin's multi-turn discounted advantages.
    Calls backward() one sequence at a time to avoid OOM from holding
    G×T=32 computation graphs simultaneously.

    1. Discounted return:   R_i_t = Σ_{k=t}^{T-1} γ^{k-t} * r_i_k
    2. Per-turn group norm: A_i_t = (R_i_t - mean_t) / (std_t + ε)
    3. Token-level DAPO Clip-Higher
    4. Dr. GRPO per-sequence mean
    5. Entropy bonus

    Returns scalar loss value for logging (gradients already applied).
    """
    G = len(group_rewards)
    T = len(group_rewards[0])

    # Discounted returns [G, T]
    rewards_t = torch.tensor(group_rewards, dtype=torch.float32)
    disc_returns = torch.zeros(G, T, dtype=torch.float32)
    for t in range(T):
        for k in range(t, T):
            disc_returns[:, t] += (config.gamma ** (k - t)) * rewards_t[:, k]

    # Per-turn group normalization → advantages [G, T]
    mean_t = disc_returns.mean(dim=0, keepdim=True)
    std_t  = disc_returns.std(dim=0,  keepdim=True) + 1e-8
    advantages = (disc_returns - mean_t) / std_t

    # ── DEBUG: discounted returns and advantages ─────────────────────────────
    print(f"  [DEBUG] disc_returns per turn (mean): {disc_returns.mean(dim=0).tolist()}")
    print(f"  [DEBUG] disc_returns std per turn:    {disc_returns.std(dim=0).tolist()}")
    print(f"  [DEBUG] advantages range: min={advantages.min().item():.3f} max={advantages.max().item():.3f}")
    # ── END DEBUG ────────────────────────────────────────────────────────────

    # Count valid sequences upfront for loss normalization
    n_seqs = sum(
        1 for i in range(G)
        for (_, resp) in group_turns[i]
        if len(resp) > 0
    )
    if n_seqs == 0:
        return 0.0

    total_loss = 0.0

    for i in range(G):
        for turn_idx, (ctx, resp) in enumerate(group_turns[i]):
            if len(resp) == 0:
                continue

            A_i_t = advantages[i, turn_idx].item()

            new_lp = _get_token_log_probs(model, ctx, resp)
            old_lp = old_log_probs[i][turn_idx].to(new_lp.device)

            ratio   = torch.exp(new_lp - old_lp.detach())
            clipped = torch.clamp(ratio, 1.0 - config.cliprange_low, 1.0 + config.cliprange_high)

            seq_loss = -torch.min(ratio * float(A_i_t), clipped * float(A_i_t))
            loss_term = seq_loss.mean()

            if config.entropy_coef > 0:
                loss_term = loss_term - config.entropy_coef * (-new_lp.mean())

            # Divide by n_seqs so the sum of all backward() calls = proper mean loss.
            # Call backward immediately — frees this sequence's computation graph,
            # keeping peak activation memory to 1 sequence at a time.
            (loss_term / n_seqs).backward()
            total_loss += loss_term.item()

    return total_loss / n_seqs


def _run_evaluation(model, tokenizer, config: GRPOConfig, val_prompts: list[str]) -> dict:
    """Run an evaluation pass on the validation set."""
    print(f"\n--- Running Evaluation ({len(val_prompts)} prompts) ---")
    if config.mock_mode:
        return {"eval/pass_rate": 0.5, "eval/valid_rate": 0.8, "eval/avg_speedup": 1.1}

    model.eval()
    candidates_to_eval = []

    with torch.no_grad():
        for prompt_text in val_prompts:
            # Build prompt matching SFT training format via apply_chat_template
            user_msg = f"{FORMAT_EXAMPLE}Reference Program:\n```python\n{prompt_text}\n```"
            sys_content = get_system_prompt().replace("<|im_start|>system\n", "").replace("<|im_end|>\n", "").strip()
            messages = [
                {"role": "system", "content": sys_content},
                {"role": "user", "content": user_msg},
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

    # ── DEBUG: dataset fields and level distribution ─────────────────────────
    print(f"\n[DEBUG] Dataset fields: {list(rows[0].keys())}")
    has_level = "level_id" in rows[0]
    if has_level:
        from collections import Counter
        level_counts = Counter(r.get("level_id", "?") for r in rows)
        print(f"[DEBUG] level_id distribution: {dict(sorted(level_counts.items()))}")
    else:
        print(f"[DEBUG] No level_id field — curriculum disabled, random shuffle")
    # ── END DEBUG ─────────────────────────────────────────────────────────────

    # Curriculum: sort by level_id (easy first) if available and enabled
    if config.curriculum and has_level:
        rows.sort(key=lambda r: (r.get("level_id", 99), random.random()))
        print(f"Curriculum enabled: sorted {len(rows)} tasks by level_id (easy-first).")
    else:
        random.shuffle(rows)

    # Filter broken reference prompts using a fast AST check (no subprocess/torch needed).
    # Catches the most common failure: get_init_inputs() references a module-level name
    # (e.g. `bias`) that was never defined, causing every generated kernel to fail with
    # a misleading NameError that has nothing to do with the kernel code itself.
    import ast, hashlib, json as _json

    _cache_path = os.path.join(os.path.dirname(config.dataset_path), ".ref_valid_cache.json")
    try:
        with open(_cache_path) as _f:
            _valid_cache = _json.load(_f)
    except Exception:
        _valid_cache = {}

    def _ref_is_runnable(pytorch_code: str) -> bool:
        h = hashlib.md5(pytorch_code.encode()).hexdigest()
        if h in _valid_cache:
            return _valid_cache[h]
        try:
            tree = ast.parse(pytorch_code)
        except SyntaxError:
            _valid_cache[h] = False
            return False
        # Collect all names defined at module level (assignments, function defs, class defs, imports)
        defined = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defined.add(node.name)
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        defined.add(t.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    defined.add(alias.asname or alias.name.split(".")[0])
        # Check that get_inputs and get_init_inputs are defined
        if "get_inputs" not in defined or "get_init_inputs" not in defined:
            _valid_cache[h] = False
            return False
        # Check for NameErrors inside get_init_inputs by finding all Name loads and
        # checking they are defined at module level or are builtins
        builtins = set(dir(__builtins__)) if isinstance(__builtins__, dict) else set(dir(__builtins__))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef,)) and node.name == "get_init_inputs":
                for child in ast.walk(node):
                    if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                        if child.id not in defined and child.id not in builtins:
                            _valid_cache[h] = False
                            return False
        _valid_cache[h] = True
        return True

    valid_rows = [r for r in rows if _ref_is_runnable(r["pytorch_code"])]
    n_broken = len(rows) - len(valid_rows)
    if n_broken:
        print(f"[Dataset] Filtered {n_broken} broken reference prompts. {len(valid_rows)} remain.")
    try:
        with open(_cache_path, "w") as _f:
            _json.dump(_valid_cache, _f)
    except Exception:
        pass
    rows = valid_rows

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
            torch_dtype=torch.bfloat16,
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
            all_group_turns:   list[list[list[TurnData]]] = []       # [B][G][T]
            all_group_rewards: list[list[list[float]]] = []           # [B][G][T]

            n_degenerate = 0
            t0 = time.time()
            for p_idx, prompt_text in enumerate(batch):
                print(f"\n[Prompt {p_idx+1}/{len(batch)}] Generating {config.group_size} trajectories (batched/parallel)...")
                group_turns, group_rewards = _run_group_episodes(prompt_text, model, tokenizer, config)

                # Dynamic Sampling: skip degenerate groups (DAPO).
                # Use mean-reward-per-trajectory as the degeneracy signal.
                if config.dynamic_sampling:
                    traj_means = [sum(rews) / len(rews) for rews in group_rewards]
                    reward_std = torch.tensor(traj_means).std().item()
                    if reward_std <= 1e-4:
                        n_degenerate += 1
                        print(f"  [Dynamic Sampling] Degenerate group (std={reward_std:.6f}), skipping prompt.")
                        continue

                all_group_turns.append(group_turns)
                all_group_rewards.append(group_rewards)

                traj_means = [sum(rews) / len(rews) for rews in group_rewards]
                mean_r = sum(traj_means) / len(traj_means)
                best_r = max(traj_means)
                reward_std = torch.tensor(traj_means).std().item()
                print(f"  Group: mean={mean_r:.2f}, best={best_r:.2f}, std={reward_std:.3f}")

            elapsed = time.time() - t0
            step_bar.update(1)

            if not all_group_rewards:
                print(f"\n[Epoch {epoch+1} | Step {global_step+1}/{total_steps}] all prompts degenerate, skipping step.")
                global_step += 1
                continue

            batch_mean = sum(
                sum(sum(rews) / len(rews) for rews in group) / len(group)
                for group in all_group_rewards
            ) / len(all_group_rewards)
            step_bar.set_postfix(reward=f"{batch_mean:.3f}", step_time=f"{elapsed:.0f}s")
            print(f"\n[Epoch {epoch+1} | Step {global_step+1}/{total_steps}] reward={batch_mean:.3f}  step_time={elapsed:.0f}s")

            torch.cuda.empty_cache()

            # ── Sleep SGLang to free VRAM for the backward pass ─────────
            if not config.mock_mode and config.use_sglang:
                release_sglang_memory(config.sglang_port)

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
                    # backward() called per-sequence inside to avoid OOM from
                    # holding G×T computation graphs simultaneously
                    loss_val = _compute_grpo_loss_and_backward(
                        model, group_turns, group_rewards, group_old_lps, config
                    )
                    total_loss_val += loss_val

                # ── DEBUG: gradient norm before clipping ────────────────────
                total_norm = torch.nn.utils.clip_grad_norm_(trainable_params, config.max_grad_norm)
                print(f"  [DEBUG] grad_norm={total_norm:.4f} (clip={config.max_grad_norm}) loss={total_loss_val:.4f}")
                # ── END DEBUG ────────────────────────────────────────────────
                optimizer.step()

            # One scheduler step per batch step (not per grpo_epoch) so LR
            # schedule completes in the expected number of training steps.
            scheduler.step()

            # ── Wake SGLang and hot-reload updated LoRA ──────────────────
            if not config.mock_mode and config.use_sglang:
                resume_sglang_memory(config.sglang_port)
                sync_lora_to_sglang(model, config.sglang_port)

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

                # Not-Okay Ratio: fraction of trajectories where final-turn reward < 0
                # High = model still mostly failing; should drop as training progresses.
                all_final_rewards = [
                    group[g][-1]
                    for group in all_group_rewards
                    for g in range(len(group))
                ]
                not_okay_ratio = sum(1 for r in all_final_rewards if r < 0) / max(1, len(all_final_rewards))

                wandb.log({
                    "train/loss": total_loss_val,
                    "train/batch_mean_reward": batch_mean,
                    "train/mean_entropy": mean_entropy,
                    "train/degenerate_groups": n_degenerate,
                    "train/not_okay_ratio": not_okay_ratio,
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
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--grpo_epochs", type=int, default=2, help="Gradient updates per rollout (2 reduces staleness vs original 4)")
    parser.add_argument("--think_budget", type=int, default=2000, help="Max thinking tokens before forcing code output")
    parser.add_argument("--num_turns", type=int, default=4, help="Multi-turn refinement turns per trajectory (Kevin's recipe)")
    parser.add_argument("--gamma", type=float, default=0.4, help="Discount factor for multi-turn returns")
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
        think_budget=args.think_budget,
        num_turns=args.num_turns,
        gamma=args.gamma,
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
