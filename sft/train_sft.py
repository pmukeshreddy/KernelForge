"""
train_sft.py - SFT training on verified CUDA kernel pairs.

Pipeline:
  1. Split 325 pairs → 80% train / 10% val (early stopping) / 10% test (held out)
  2. Train Qwen3-14B with LoRA
  3. After training: run inference on held-out test set + 50 random KernelBench prompts
     → compile + verify with sandbox → report Pass@1
"""

import json
import os
import random
import sys
import re
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
from datasets import Dataset

_rl_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rl"))
if _rl_dir not in sys.path:
    sys.path.insert(0, _rl_dir)

# ── Prompt format (must match generate_sft_data.py exactly) ────────────────

SYSTEM = """\
<|im_start|>system
You are an expert NVIDIA CUDA Systems Engineer.
Your objective is to write an optimized CUDA kernel to replace a PyTorch operation,
delivered as a complete, self-contained Python file.

# Output Format
Output EXACTLY ONE ```python code block containing a complete model_new.py file that:
1. Embeds the CUDA C++ source as a string.
2. Compiles it with `torch.utils.cpp_extension.load_inline`.
3. Defines a `ModelNew(torch.nn.Module)` class whose `forward()` calls the kernel.

# Constraints
- CUDA kernel: `#include <torch/extension.h>`, `#include <cuda_runtime.h>`.
- Binding function must return `torch::Tensor` and include `PYBIND11_MODULE`.
- Input tensors are `float32`. Use `float*` and `.data_ptr<float>()`.
- Do NOT use cuBLAS, cuDNN, or CUTLASS.

# Common Bugs to Avoid
- Use `fmaxf`/`fminf` in device code, NOT `std::max`/`std::min`.
- Max 1024 threads per block.
- Declare `__shared__` arrays INSIDE the kernel body.
<|im_end|>
"""

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

Now write the complete model_new.py for the following operation:
"""


def make_prompt(pytorch_code: str) -> str:
    """Build the inference prompt (no assistant content — model generates it)."""
    user_msg = FORMAT_EXAMPLE + f"Reference Program:\n```python\n{pytorch_code}\n```"
    return (
        SYSTEM
        + f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        + "<|im_start|>assistant\n"
    )


def _extract_python_block(text: str) -> str:
    m = re.search(r'```python\s*(.*?)```', text, re.DOTALL)
    return m.group(1).strip() if m else ""


# ── Sandbox verification worker (same as GRPO) ──────────────────────────────

def _eval_worker(item):
    import os, sys, io
    if _rl_dir not in sys.path:
        sys.path.insert(0, _rl_dir)
    model_new_py, pytorch_code, label = item
    if not model_new_py or "__global__" not in model_new_py:
        return False, label, "no_kernel"
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = io.StringIO()
    try:
        from sandbox import evaluate
        sys.stdout, sys.stderr = old_out, old_err
        result = evaluate(model_new_py, pytorch_code)
        ok = bool(result and result.get("correct", False))
        err = (result or {}).get("compiler_error", "")
        return ok, label, err
    except Exception as e:
        sys.stdout, sys.stderr = old_out, old_err
        return False, label, str(e)


def run_eval(model, tokenizer, eval_items: list, workers: int = 16, tag: str = "eval",
             batch_size: int = 4):
    """
    Generate model_new.py for each (pytorch_code, label), compile+verify, report Pass@1.
    eval_items: list of (pytorch_code, label)
    """
    print(f"\n{'='*60}")
    print(f"Post-training eval: {tag} ({len(eval_items)} problems)")
    print(f"{'='*60}")

    # Left-pad so all sequences in a batch end at the same position
    tokenizer.padding_side = "left"
    model.eval()

    generated = []
    for i in range(0, len(eval_items), batch_size):
        batch = eval_items[i:i + batch_size]
        prompts = [make_prompt(pytorch_code) for pytorch_code, _ in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                           truncation=True, max_length=4096).to(model.device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        for j, (pytorch_code, label) in enumerate(batch):
            text = tokenizer.decode(out[j][prompt_len:], skip_special_tokens=True)
            model_new_py = _extract_python_block(text) or None
            generated.append((model_new_py, pytorch_code, label))
        print(f"  Generated {min(i+batch_size, len(eval_items))}/{len(eval_items)}", end="\r")

    # Parallel sandbox verification
    n_pass = n_fail = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_eval_worker, item): item for item in generated}
        for fut in futures:
            ok, label, err = fut.result()
            if ok:
                n_pass += 1
            else:
                n_fail += 1
                print(f"  FAIL [{label}]: {(err or '')[:120]}")

    total = n_pass + n_fail
    print(f"\n{tag} Pass@1: {n_pass}/{total} = {n_pass/max(1,total)*100:.1f}%")
    return n_pass / max(1, total)


# ── main ────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="Qwen/Qwen3-14B")
    parser.add_argument("--sft_data", default="./sft_training_pairs.jsonl")
    parser.add_argument("--rl_prompts", default="./rl_prompts.jsonl",
                        help="KernelBench prompts for post-training eval")
    parser.add_argument("--output_dir", default="./sft_qwen3_14b_lora")
    parser.add_argument("--eval_workers", type=int, default=16)
    parser.add_argument("--n_kernelbench_eval", type=int, default=50,
                        help="Number of KernelBench prompts to eval after training")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # ── Load and split data ────────────────────────────────────────────────
    print(f"Loading SFT pairs from {args.sft_data}")
    pairs = []
    with open(args.sft_data) as f:
        for line in f:
            p = json.loads(line)
            if p.get("text", "").strip():
                pairs.append(p)
    print(f"Total pairs: {len(pairs)}")

    random.shuffle(pairs)
    n_test = max(1, int(len(pairs) * 0.10))   # 10% held-out test
    n_val  = max(1, int(len(pairs) * 0.10))   # 10% val for early stopping
    test_pairs  = pairs[:n_test]
    val_pairs   = pairs[n_test:n_test + n_val]
    train_pairs = pairs[n_test + n_val:]

    print(f"Split → train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)} (held out)")

    train_dataset = Dataset.from_list(train_pairs)
    eval_dataset  = Dataset.from_list(val_pairs)

    # ── Load model + tokenizer ─────────────────────────────────────────────
    print(f"\nLoading tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Train ──────────────────────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir="./sft_output",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=1e-4,
        warmup_ratio=0.1,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        gradient_checkpointing=True,
        dataset_text_field="text",
        packing=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    print("\nStarting SFT training...")
    trainer.train()

    # Save LoRA adapter (do NOT merge — GRPO needs separate base + adapter)
    print(f"\nSaving LoRA adapter to {args.output_dir}")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # ── Post-training eval ─────────────────────────────────────────────────
    # 1. Held-out test set from SakanaAI (same distribution as training)
    test_items = [(p["pytorch_code"], f"sakana/{p.get('task_id','?')}") for p in test_pairs]
    run_eval(trainer.model, tokenizer, test_items,
             workers=args.eval_workers, tag="held-out SakanaAI test")

    # 2. KernelBench prompts (different distribution — unseen tasks)
    kb_items = []
    if os.path.exists(args.rl_prompts):
        with open(args.rl_prompts) as f:
            kb_prompts = [json.loads(l) for l in f]
        kb_sample = random.sample(kb_prompts, min(args.n_kernelbench_eval, len(kb_prompts)))
        kb_items = [(p["pytorch_code"], f"kb/{p.get('task_id','?')}") for p in kb_sample]
        run_eval(trainer.model, tokenizer, kb_items,
                 workers=args.eval_workers, tag="KernelBench (unseen)")
    else:
        print(f"\nSkipping KernelBench eval — {args.rl_prompts} not found")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
