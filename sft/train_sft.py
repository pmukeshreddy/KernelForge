"""
train_sft.py - SFT training on CUDA-L1 kernel pairs.

Changes from v1:
  1. Qwen3-14B (was 8B)
  2. 5 epochs with early stopping (was 15 fixed)
  3. LR 1e-4 (LoRA standard for Qwen3, was 2e-5)
  4. packing=True (eliminates padding waste, 2-3x speedup)
  5. No DataCollatorForLanguageModeling (SFTTrainer handles it)
  6. Save LoRA adapter separately, DO NOT merge (GRPO needs LoRA toggle)
  7. target_modules="all-linear" (standard for Qwen3)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

def main():
    model_id = "Qwen/Qwen3-14B"
    output_adapter_dir = "./sft_qwen3_14b_lora"
    sft_data_file = "./sft_training_pairs.jsonl"
    
    print(f"Loading Tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading Model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="kernels-community/flash-attn2"
    )
    
    # LoRA configuration — "all-linear" catches all linear layers
    print("Setting up LoRA configuration")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load SFT training pairs (only entries with text field — no KernelBench)
    print(f"Loading SFT training data from {sft_data_file}")
    full_dataset = load_dataset("json", data_files=sft_data_file, split="train")
    full_dataset = full_dataset.filter(lambda x: bool(x.get("text", "").strip()))
    print(f"Loaded {len(full_dataset)} training examples")
    
    # 10% validation split for early stopping
    split = full_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    
    training_args = SFTConfig(
        output_dir="./sft_output",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
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
    
    print("Initializing SFTTrainer")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )
    
    print("Starting Training...")
    trainer.train()

    # Save LoRA adapter separately — DO NOT merge_and_unload()
    # GRPO stage loads base model + adapter, and toggles LoRA off for reference
    print(f"Saving LoRA adapter to {output_adapter_dir}")
    trainer.model.save_pretrained(output_adapter_dir)
    tokenizer.save_pretrained(output_adapter_dir)
    print("Done! Adapter saved (not merged).")

if __name__ == "__main__":
    main()
