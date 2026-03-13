import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, get_peft_model
from dataset_hf import prepare_cuda_agent_dataset

def main():
    model_id = "Qwen/Qwen3-8B"  # Or Qwen2.5-Coder-7B depending on release
    dataset_name = "BytedTsinghua-SIA/CUDA-Agent-Ops-6K" # The actual 6K ops from HF
    output_model_dir = "./sft_qwen3_8b_cuda"
    
    print(f"Loading Tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading Model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2" # Highly recommended for H100
    )
    
    # LoRA config for 8B model on a single node / H100
    print("Setting up LoRA configuration")
    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    
    # Load and format the dataset
    train_dataset = prepare_cuda_agent_dataset(dataset_name, split="train")
    
    training_args = SFTConfig(
        output_dir="./sft_output",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        logging_steps=10,
        num_train_epochs=3, # Usually 1-3 epochs is enough for SFT 
        save_strategy="epoch",
        bf16=True,
        report_to="tensorboard",
        max_grad_norm=1.0,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True
    )
    
    # Important: Tell the trainer to only compute loss on the response
    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    
    print("Initializing SFTTrainer")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=collator,
    )
    
    print("Starting Training...")
    trainer.train()
    
    print(f"Saving Model to {output_model_dir}")
    trainer.save_model(output_model_dir)

if __name__ == "__main__":
    main()
