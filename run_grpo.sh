#!/bin/bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
# Auto-detect CUDA_HOME: try nvcc path, then common locations
if [ -z "$CUDA_HOME" ]; then
  if which nvcc &>/dev/null; then
    export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
  elif [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME=/usr/local/cuda
  elif [ -d "/usr/local/cuda-12.8" ]; then
    export CUDA_HOME=/usr/local/cuda-12.8
  fi
fi
export CUDA_PATH=${CUDA_HOME}
echo "CUDA_HOME=${CUDA_HOME}"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR/rl"
python train_grpo.py \
  --adapter mukeshreddy/kernelforge-qwen3-14b-lora-v2 \
  --dataset ../sft/sft_training_pairs.jsonl \
  --output_dir ./checkpoints/grpo_v1 \
  --group_size 8 \
  --batch_size 2 \
  --wandb_project kernelforge-rl \
  --use_sglang \
  --sglang_python ~/sglang_env/bin/python \
  --sglang_port 30000 \
  --sglang_tp 1
