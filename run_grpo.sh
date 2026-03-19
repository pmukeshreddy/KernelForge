#!/bin/bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd /content/KernelForge/rl
python train_grpo.py \
  --adapter mukeshreddy/kernelforge-qwen3-14b-lora-v2 \
  --dataset ../sft/sft_training_pairs.jsonl \
  --output_dir ./checkpoints/grpo_v1 \
  --group_size 8 \
  --batch_size 2 \
  --wandb_project kernelforge-rl \
  --use_sglang \
  --sglang_python /content/sglang_env/bin/python \
  --sglang_port 30000 \
  --sglang_tp 1
