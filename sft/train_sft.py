import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import json

def main():
    model_id = "Qwen/Qwen3-8B"
    output_model_dir = "./sft_qwen3_8b_cuda"
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
        attn_implementation="flash_attention_2"
    )
    
    # LoRA configuration for efficient fine-tuning
    print("Setting up LoRA configuration")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load the generated SFT training pairs
    print(f"Loading SFT training data from {sft_data_file}")
    train_dataset = load_dataset("json", data_files=sft_data_file, split="train")
    print(f"Loaded {len(train_dataset)} training examples")
    
    training_args = SFTConfig(
        output_dir="./sft_output",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        num_train_epochs=3,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        max_seq_length=4096,
        dataset_text_field="text",
    )
    
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    print("Initializing SFTTrainer")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=collator,
    )
    
    print("Starting Training...")
    trainer.train()
    
    # Save the final LoRA-merged model
    print(f"Saving Model to {output_model_dir}")
    trainer.save_model(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)
    print("Done!")

if __name__ == "__main__":
    main()
