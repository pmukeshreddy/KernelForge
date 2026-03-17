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
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class StopTokenCriteria(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Check if the last sequence of tokens matches any of our stop token sequences
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
    
    # Define stop tokens for "```" to prevent hallucinating markdown into the C++ file
    stop_tokens = tokenizer("```", add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    stopping_criteria = StoppingCriteriaList([StopTokenCriteria([stop_tokens[0]])])

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, trace in enumerate(tqdm(eval_prompts, desc="Evaluating", leave=False)):
            system_prompt = get_system_prompt()
            messages = [
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
            
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # Force the model to start writing the C++ block using the global PREFILL
            prompt_text += PREFILL
            
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
            response_text = PREFILL + tokenizer.decode(gen_ids, skip_special_tokens=True)
            
            # The model was forced to start with PREFILL, so response_text IS the C++ block
            # Just strip off the trailing ``` if it generated it
            cuda_code = response_text.split("```")[0].strip()
                
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
