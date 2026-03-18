#!/bin/bash
source /root/KernelForge/.venv/bin/activate
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd /root/KernelForge/rl
python train_grpo.py --adapter ../sft/sft_qwen3_14b_lora --dataset ../sft/sft_training_pairs.jsonl --output_dir ./checkpoints/grpo_v1 --group_size 8 --batch_size 2 --use_sglang --sglang_python /root/sglang_env/bin/python --wandb_project kernelforge-rl
