"""
train_redi.py - REDI Stage 2: Train on verified distillation traces.

REINFORCE with binary rewards:
  - Positive traces (label=+1): increase likelihood
  - Negative traces (label=-1): decrease likelihood
  - Loss = -label * mean(token_log_probs)

No reference model, no critic, no value head. Continues training the
LoRA adapter from SFT — no new LoRA, no freezing.
"""

import json
import os
import random
import argparse

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from sys_prompt import get_system_prompt


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "model_id": "Qwen/Qwen3-14B",
    "adapter_path": "../sft/sft_qwen3_14b_lora",
    "traces_path": "data/redi_traces.jsonl",
    "output_dir": "checkpoints/kernelforge_redi",
    "learning_rate": 5e-6,
    "num_epochs": 2,
    "batch_size": 4,
    "max_grad_norm": 1.0,
    "max_seq_len": 3072,
    "save_steps": 100,
}

PREFILL = "```cpp\n#include <torch/extension.h>\n"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_traces(path: str) -> list[dict]:
    """Load REDI traces and filter out entries with no code."""
    traces = []
    with open(path) as f:
        for line in f:
            if line.strip():
                t = json.loads(line)
                # Must have cuda_code for training
                if t.get("cuda_code", "").strip():
                    traces.append(t)
    return traces


def balance_traces(traces: list[dict]) -> list[dict]:
    """
    Roughly balance positive and negative traces.
    Oversample the minority class to match the majority.
    """
    pos = [t for t in traces if t["label"] == 1]
    neg = [t for t in traces if t["label"] == -1]

    if not pos or not neg:
        return traces

    if len(pos) < len(neg):
        # Oversample positives
        factor = len(neg) // len(pos)
        pos = pos * factor + random.sample(pos, min(len(neg) % len(pos), len(pos)))
    elif len(neg) < len(pos):
        # Oversample negatives
        factor = len(pos) // len(neg)
        neg = neg * factor + random.sample(neg, min(len(pos) % len(neg), len(neg)))

    balanced = pos + neg
    random.shuffle(balanced)
    return balanced


def build_chat_text(pytorch_code: str, cuda_code: str, tokenizer) -> str:
    """
    Reconstruct the chat format used during generation.
    System + User prompt + Assistant response with the full Python ModelNew.
    """
    system_prompt = get_system_prompt()
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Write an optimized CUDA kernel to replace this PyTorch "
                "implementation.\n\n"
                f"Reference Program:\n```python\n{pytorch_code}\n```"
            ),
        },
        {
            "role": "assistant",
            "content": f"```python\n{cuda_code}\n```",
        },
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def compute_redi_loss(
    model,
    tokenizer,
    trace: dict,
    max_seq_len: int = 3072,
) -> torch.Tensor:
    """
    Compute REDI loss for one trace.
    loss = -weight * mean(token_log_probs of assistant response)
    Positive traces weighted by actual speedup reward (3x >> 1.01x).
    Negative traces weighted at 1.0.
    """
    device = next(model.parameters()).device
    label = trace["label"]  # +1 or -1
    if label == 1:
        weight = max(trace.get("reward", 1.0), 0.1)  # floor avoids zero grad on slow-but-correct
    else:
        weight = 1.0

    # Build full chat and tokenize
    full_text = build_chat_text(trace["pytorch_code"], trace["cuda_code"], tokenizer)
    tokens = tokenizer(full_text, return_tensors="pt", truncation=True,
                       max_length=max_seq_len)
    input_ids = tokens.input_ids[0].to(device)

    # Find where the assistant response starts
    # Tokenize everything EXCEPT the assistant message to find the boundary
    system_prompt = get_system_prompt()
    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Write an optimized CUDA kernel to replace this PyTorch "
                "implementation.\n\n"
                f"Reference Program:\n```python\n{trace['pytorch_code']}\n```"
            ),
        },
    ]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    prompt_len = len(tokenizer(prompt_text, truncation=True, max_length=max_seq_len).input_ids)

    if prompt_len >= len(input_ids):
        # Response got truncated away entirely
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Forward pass
    outputs = model(input_ids=input_ids.unsqueeze(0))
    logits = outputs.logits[0]  # [T, V]

    # Token-level log probs on the response tokens only
    resp_logits = logits[prompt_len - 1 : -1]  # [R, V] — predict next token
    resp_targets = input_ids[prompt_len:]       # [R]
    R = min(len(resp_logits), len(resp_targets))

    if R == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    resp_logits = resp_logits[:R]
    resp_targets = resp_targets[:R]

    log_probs = F.log_softmax(resp_logits, dim=-1)
    token_log_probs = log_probs[range(R), resp_targets]

    # REDI loss: -label * weight * mean(log_probs)
    # positive: weight=speedup reward (3x kernel gets 3x gradient vs 1x)
    # negative: weight=1.0 (uniform penalty)
    loss = -label * weight * token_log_probs.mean()

    return loss


def train(config: dict = None):
    if config is None:
        config = DEFAULT_CONFIG.copy()

    print(f"Loading traces from {config['traces_path']}...")
    raw_traces = load_traces(config["traces_path"])
    pos = sum(1 for t in raw_traces if t["label"] == 1)
    neg = sum(1 for t in raw_traces if t["label"] == -1)
    print(f"Raw traces: {len(raw_traces)} ({pos} positive, {neg} negative)")

    traces = balance_traces(raw_traces)
    print(f"Balanced traces: {len(traces)}")

    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model: {config['model_id']}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"Loading SFT adapter: {config['adapter_path']}...")
    model = PeftModel.from_pretrained(base_model, config['adapter_path'], is_trainable=True)
    
    model.base_model.model.model.gradient_checkpointing_enable()

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.2f}%)")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config["learning_rate"])

    os.makedirs(config["output_dir"], exist_ok=True)
    batch_size = config["batch_size"]
    max_seq_len = config["max_seq_len"]

    print(f"\n🚀 REDI Training")
    print(f"   epochs={config['num_epochs']}, batch_size={batch_size}, lr={config['learning_rate']}")

    global_step = 0

    for epoch in range(config["num_epochs"]):
        random.shuffle(traces)
        epoch_loss = 0.0
        n_batches = 0

        for batch_start in range(0, len(traces), batch_size):
            batch = traces[batch_start : batch_start + batch_size]

            optimizer.zero_grad()
            batch_loss_val = 0.0
            n_valid = 0

            for trace in batch:
                loss = compute_redi_loss(model, tokenizer, trace, max_seq_len)
                if loss.requires_grad:
                    (loss / len(batch)).backward()
                    batch_loss_val += loss.item()
                    n_valid += 1

            if n_valid > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, config["max_grad_norm"])
                optimizer.step()

            epoch_loss += batch_loss_val
            n_batches += 1
            global_step += 1

            if global_step % 20 == 0:
                avg = epoch_loss / max(n_batches, 1)
                print(f"  [Epoch {epoch+1} | Step {global_step}] avg_loss={avg:.4f}")

            if global_step % config["save_steps"] == 0:
                ckpt = f"{config['output_dir']}/step_{global_step}"
                model.save_pretrained(ckpt)
                tokenizer.save_pretrained(ckpt)
                print(f"  Checkpoint → {ckpt}")

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch+1} complete. avg_loss={avg_loss:.4f}")

    model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])
    print(f"\n✅ REDI training complete. Model saved to {config['output_dir']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="REDI Training")
    parser.add_argument("--traces", type=str, default="data/redi_traces.jsonl")
    parser.add_argument("--output_dir", type=str, default="checkpoints/kernelforge_redi")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--adapter", type=str, default="../sft/sft_qwen3_14b_lora")
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG.copy()
    cfg.update({
        "model_id": args.model,
        "adapter_path": args.adapter,
        "traces_path": args.traces,
        "output_dir": args.output_dir,
        "learning_rate": args.lr,
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
    })
    train(cfg)
