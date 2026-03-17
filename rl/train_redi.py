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


# ---------------------------------------------------------------------------
# SFT Format Compatibility
# ---------------------------------------------------------------------------
# The model was fine-tuned on raw text matching this exact format, not standard chat templates.

SYSTEM = """<|im_start|>system
You are an expert NVIDIA CUDA Systems Engineer.
Your objective is to write optimized CUDA C++ kernels to replace PyTorch operations.

# Constraints
- Write valid CUDA C++ with `#include <torch/extension.h>` and `#include <cuda_runtime.h>`.
- Write a `__global__ void` kernel with proper thread indexing.
- Write a C++ binding function returning `torch::Tensor` using PyTorch C++ API.
- Input tensors are `float32` by default. Use `float*` pointers and `data_ptr<float>()`.
- Do NOT use cuBLAS, cuDNN, or CUTLASS.

# Output Format
Output EXACTLY ONE ```cpp code block containing your kernel and binding function.

# Common Bugs to Avoid
- Use `fmaxf`/`fminf` in device code, NOT `std::max`/`std::min`.
- Max 1024 threads per block. For 2D blocks: blockDim.x * blockDim.y <= 1024.
- Declare `__shared__` arrays INSIDE the kernel function body.
- Use `torch::empty_like(input)` to preserve tensor shape and dtype.
<|im_end|>
"""

FORMAT_EXAMPLE = """Here is an example of the expected output format:

```cpp
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void add_kernel(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] + b[idx];
}

torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b) {
    auto out = torch::empty_like(a);
    int n = a.numel();
    add_kernel<<<(n + 255) / 256, 256>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}
```

Now write the kernel for the following operation:
"""

def _make_prompt_text(pytorch_code: str) -> str:
    user_msg = FORMAT_EXAMPLE + f"```python\n{pytorch_code}\n```"
    # Matches exactly what model saw: system + user + assistant start
    return SYSTEM + f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{PREFILL}"

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

    # Build full chat and tokenize exactly as training did
    prompt_text = _make_prompt_text(trace["pytorch_code"])
    # The full text is prompt + the rest of the generated C++ code + end token
    # We strip PREFILL from cuda_code because it's already in the prompt
    cuda_tail = trace["cuda_code"].replace(PREFILL, "", 1)
    full_text = prompt_text + cuda_tail + "\n```<|im_end|>\n"
    
    tokens = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=max_seq_len)
    input_ids = tokens.input_ids[0].to(device)

    # Find where the assistant response starts
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
    loss = -label * weight * token_log_probs.mean()

    return loss


import subprocess
import tempfile
from tqdm import tqdm
from transformers import StoppingCriteria, StoppingCriteriaList

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class StopTokenCriteria(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_seq in self.stop_token_ids:
            if input_ids.shape[-1] >= len(stop_seq):
                if torch.all(input_ids[0, -len(stop_seq):] == stop_seq):
                    return True
        return False

def evaluate_compile_rate(model, tokenizer, eval_prompts: list[dict], max_seq_len: int = 3072) -> float:
    """
    Generates CUDA code for the evaluation prompts and attempts to compile it using nvcc.
    Returns the Pass@1 compile rate (%).
    """
    if not eval_prompts:
        return 0.0

    print("\n[Eval] Running Compile Rate Pass@1 Evaluation...")
    model.eval()
    device = next(model.parameters()).device
    successes = 0
    
    stop_tokens = tokenizer("```", add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    stopping_criteria = StoppingCriteriaList([StopTokenCriteria([stop_tokens[0]])])

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, trace in enumerate(tqdm(eval_prompts, desc="Evaluating", leave=False)):
            prompt_text = _make_prompt_text(trace["pytorch_code"])
            inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=1500).to(device)
            
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=1500,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False, # greedy for Pass@1 eval
                    stopping_criteria=stopping_criteria
                )
            
            # Extract only the generated response
            gen_ids = output_ids[0][inputs.input_ids.shape[1]:]
            response_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            
            # The prompt already ended with PREFILL ("```cpp\n#include <torch/extension.h>\n"), 
            # so the model's generated text is the CONTINUATION of that C++ code.
            # We must prepend PREFILL back to make it a valid, complete C++ file.
            cuda_code = PREFILL + response_text
            
            # Extract only the actual C++ code if the model generated closing backticks
            if "```" in response_text:
                cuda_code = (PREFILL + response_text).split("```")[0].strip()
            
            if i == 0:
                print(f"\n[Eval Debug] Full Prompt:\n{prompt_text}\n")
                print(f"[Eval Debug] Successfully extracted C++ code (first 500 chars):\n{cuda_code[:500]}...\n")
                
            # Test Compile
            cu_file = os.path.join(tmpdir, f"test_{i}.cu")
            obj_file = os.path.join(tmpdir, f"test_{i}.o")
            with open(cu_file, "w") as f:
                f.write(cuda_code)
                
            # Dynamically get PyTorch includes so it handles .venv correctly
            from torch.utils.cpp_extension import include_paths
            import sysconfig
            inc_flags = [f"-I{p}" for p in include_paths()]
            python_inc = f"-I{sysconfig.get_path('include')}"
            
            try:
                cmd = ["nvcc", "-c", cu_file, "-o", obj_file] + inc_flags + [python_inc, "-std=c++17", "-O3", "-w"]
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
                if result.returncode == 0:
                    successes += 1
                elif successes == 0 and i == 0:
                    # Print the first error for debugging
                    print(f"\n[Eval Error Debug] nvcc failed:\n{result.stderr.decode('utf-8')[:500]}")
            except Exception as e:
                pass

    model.train()
    pass_rate = (successes / len(eval_prompts)) * 100
    print(f"[Eval] Compile Rate (Pass@1): {pass_rate:.1f}% ({successes}/{len(eval_prompts)})")
    return pass_rate

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

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
    
    # Load Evaluation Prompts
    eval_prompts = []
    if config.get("eval_path") and os.path.exists(config["eval_path"]):
        print(f"Loading eval prompts from {config['eval_path']}...")
        with open(config["eval_path"]) as f:
            for line in f:
                if line.strip():
                    eval_prompts.append(json.loads(line))
        # Just grab 1 random prompt for fast eval testing
        if len(eval_prompts) > 1:
            eval_prompts = random.sample(eval_prompts, 1)
        print(f"Loaded {len(eval_prompts)} eval prompts for Compile Rate testing.")

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

    # Initial Eval
    if eval_prompts:
        evaluate_compile_rate(model, tokenizer, eval_prompts)

    global_step = 0
    total_steps = (len(traces) // batch_size) * config["num_epochs"]

    with tqdm(total=total_steps, desc="Training") as pbar:
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
                
                avg_loss = epoch_loss / max(n_batches, 1)
                pbar.update(1)
                pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

                if global_step % config["save_steps"] == 0:
                    ckpt = f"{config['output_dir']}/step_{global_step}"
                    model.save_pretrained(ckpt)
                    tokenizer.save_pretrained(ckpt)
                    tqdm.write(f"  Checkpoint → {ckpt}")
                    
                    if eval_prompts:
                        evaluate_compile_rate(model, tokenizer, eval_prompts)

    if eval_prompts:
        evaluate_compile_rate(model, tokenizer, eval_prompts)

    model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])
    print(f"\n✅ REDI training complete. Model saved to {config['output_dir']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="REDI Training")
    parser.add_argument("--traces", type=str, default="data/redi_traces.jsonl")
    parser.add_argument("--eval_data", type=str, default="../sft/sft_training_pairs.jsonl")
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
        "eval_path": args.eval_data,
        "output_dir": args.output_dir,
        "learning_rate": args.lr,
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
    })
    train(cfg)
