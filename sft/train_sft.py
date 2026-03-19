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
from tqdm import tqdm
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
try:
    from trl import DataCollatorForCompletionOnlyLM
except ImportError:
    # Older TRL versions — inline implementation of response-only loss masking.
    from dataclasses import dataclass
    from transformers import PreTrainedTokenizerBase
    from typing import Any

    @dataclass
    class DataCollatorForCompletionOnlyLM:
        """Mask all tokens before the response_template so loss only flows through the assistant turn."""
        response_template: str
        tokenizer: Any

        def __call__(self, features):
            import torch
            from transformers.data.data_collator import DataCollatorForSeq2Seq
            input_ids = [torch.tensor(f["input_ids"]) for f in features]
            # Pad to max length in batch
            max_len = max(t.shape[0] for t in input_ids)
            padded_ids, labels = [], []
            for ids in input_ids:
                pad_len = max_len - ids.shape[0]
                padded = torch.cat([torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long), ids])
                label = padded.clone()
                # Find response template tokens
                tmpl_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)
                tmpl_len = len(tmpl_ids)
                # Mask everything up to and including the response template
                mask_end = 0
                for i in range(len(label) - tmpl_len + 1):
                    if label[i:i+tmpl_len].tolist() == tmpl_ids:
                        mask_end = i + tmpl_len
                label[:mask_end] = -100
                # Also mask padding tokens
                label[:pad_len] = -100
                padded_ids.append(padded)
                labels.append(label)
            return {
                "input_ids": torch.stack(padded_ids),
                "attention_mask": (torch.stack(padded_ids) != self.tokenizer.pad_token_id).long(),
                "labels": torch.stack(labels),
            }
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
- Binding function must return `torch::Tensor`. Do NOT include PYBIND11_MODULE — load_inline generates it automatically via functions=[].
- Input tensors are `float32`. Use `float*` and `.data_ptr<float>()`.
- Do NOT use cuBLAS, cuDNN, or CUTLASS.
- `load_inline` MUST always include `cpp_sources` as a string with the C++ function declaration(s).

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
    # Strip Qwen3 thinking tokens before extracting
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
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
             batch_size: int = 8):
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

    # Generation with tqdm
    generated = []
    n_batches = (len(eval_items) + batch_size - 1) // batch_size
    with tqdm(total=len(eval_items), desc="Generate", unit="problem") as bar:
        for i in range(0, len(eval_items), batch_size):
            batch = eval_items[i:i + batch_size]
            prompts = [make_prompt(pytorch_code) for pytorch_code, _ in batch]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                               truncation=True, max_length=4096).to(model.device)
            prompt_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=12288,
                    do_sample=True,
                    temperature=0.6,
                    pad_token_id=tokenizer.eos_token_id,
                )
            for j, (pytorch_code, label) in enumerate(batch):
                text = tokenizer.decode(out[j][prompt_len:], skip_special_tokens=True)
                model_new_py = _extract_python_block(text) or None
                generated.append((model_new_py, pytorch_code, label, text))
            bar.update(len(batch))

    # Parallel sandbox verification with live pass rate
    # Build lookup: label -> raw text for failure diagnosis
    raw_text_map = {label: text for _, _, label, text in generated}
    verify_input = [(m, p, label) for m, p, label, _ in generated]

    n_pass = n_fail = 0
    failures = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_eval_worker, item): item for item in verify_input}
        with tqdm(total=len(verify_input), desc="Verify", unit="kernel") as bar:
            for fut in as_completed(futures):
                ok, label, err = fut.result()
                if ok:
                    n_pass += 1
                else:
                    n_fail += 1
                    failures.append((label, err, raw_text_map.get(label, "")))
                total_so_far = n_pass + n_fail
                bar.set_postfix(
                    passed=n_pass,
                    failed=n_fail,
                    pass_rate=f"{n_pass/total_so_far*100:.0f}%"
                )
                bar.update(1)

    # Print full diagnostics — raw output + sandbox error for ALL failures
    if failures:
        print(f"\n{'='*60}")
        print(f"ALL {len(failures)} FAILURES — FULL OUTPUT")
        print(f"{'='*60}")
        for label, err, raw in failures:
            has_think = "<think>" in raw
            has_block = "```python" in raw
            print(f"\n{'─'*60}")
            print(f"[{label}]")
            print(f"  has_think={has_think}  has_python_block={has_block}")
            print(f"  sandbox_err: {(err or 'none')[:300]}")
            print(f"  raw_output:\n{raw}")

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
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--n_kernelbench_eval", type=int, default=50,
                        help="Number of KernelBench prompts to eval after training")
    parser.add_argument("--n_eval", type=int, default=0,
                        help="Limit held-out test to N samples (0 = all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_train", action="store_true",
                        help="Skip training, load existing LoRA from --output_dir and run eval only")
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
    if args.no_train:
        # Load base model + existing LoRA adapter
        from peft import PeftModel
        print(f"\nLoading tokenizer from {args.output_dir}")
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir, trust_remote_code=True, use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Loading base model: {args.model_id}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="kernels-community/flash-attn2",
        )
        print(f"Loading LoRA adapter from {args.output_dir}")
        model = PeftModel.from_pretrained(model, args.output_dir)
        eval_model = model
    else:
        print(f"\nLoading tokenizer: {args.model_id}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True, use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token

        print(f"Loading model: {args.model_id}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="kernels-community/flash-attn2",
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

        # Only compute loss on the assistant turn (think block + kernel).
        # The system prompt + format example + user message are identical boilerplate
        # across all examples — computing loss on them causes mode collapse.
        # DataCollatorForCompletionOnlyLM masks every token before the response
        # template, so gradients only flow through <think>...</think> + ```python...```.
        # packing must be False — sequence packing breaks per-example masking.
        RESPONSE_TEMPLATE = "<|im_start|>assistant\n"
        # Sanity check: verify the response template tokenizes to IDs that actually
        # appear in training examples. If this fails, loss masking is silently broken.
        tmpl_ids = tokenizer.encode(RESPONSE_TEMPLATE, add_special_tokens=False)
        sample_ids = tokenizer.encode(train_pairs[0]["text"], add_special_tokens=False)
        found = any(sample_ids[i:i+len(tmpl_ids)] == tmpl_ids
                    for i in range(len(sample_ids) - len(tmpl_ids) + 1))
        print(f"\nLoss masking sanity check:")
        print(f"  Template IDs : {tmpl_ids}")
        print(f"  Found in sample: {found}")
        if not found:
            raise RuntimeError(
                "Response template token IDs not found in training text. "
                "Loss masking will be silently broken. "
                f"Template '{RESPONSE_TEMPLATE}' -> {tmpl_ids}. "
                "Check that the text field uses the same tokenization as the template."
            )
        print("  OK — masking will work correctly.\n")

        response_collator = DataCollatorForCompletionOnlyLM(
            response_template=RESPONSE_TEMPLATE,
            tokenizer=tokenizer,
        )

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
            packing=False,
            report_to="none",
        )

        # Force 8192-token context — SFTTrainer reads model_max_length from tokenizer.
        tokenizer.model_max_length = 8192

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            data_collator=response_collator,
        )

        print("\nStarting SFT training...")
        trainer.train()

        print(f"\nSaving LoRA adapter to {args.output_dir}")
        trainer.model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        eval_model = trainer.model

    # ── Post-training eval ─────────────────────────────────────────────────
    # 1. Held-out test set from SakanaAI (same distribution as training)
    test_items = [(p["pytorch_code"], f"sakana/{p.get('task_id','?')}") for p in test_pairs]
    if args.n_eval > 0:
        test_items = test_items[:args.n_eval]
    run_eval(eval_model, tokenizer, test_items,
             workers=args.eval_workers, tag="held-out SakanaAI test",
             batch_size=args.eval_batch_size)

    # 2. KernelBench prompts (different distribution — unseen tasks)
    kb_items = []
    if os.path.exists(args.rl_prompts):
        with open(args.rl_prompts) as f:
            kb_prompts = [json.loads(l) for l in f]
        kb_sample = random.sample(kb_prompts, min(args.n_kernelbench_eval, len(kb_prompts)))
        kb_items = [(p["pytorch_code"], f"kb/{p.get('task_id','?')}") for p in kb_sample]
        run_eval(eval_model, tokenizer, kb_items,
                 workers=args.eval_workers, tag="KernelBench (unseen)",
                 batch_size=args.eval_batch_size)
    else:
        print(f"\nSkipping KernelBench eval — {args.rl_prompts} not found")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
