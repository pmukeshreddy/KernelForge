#!/bin/bash
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Find nvcc — it's not in PATH on this server, search common locations
NVCC_PATH=$(find /usr/local/cuda* /usr/cuda* /opt/cuda* -name nvcc -type f 2>/dev/null | head -1)
if [ -z "$NVCC_PATH" ]; then
  NVCC_PATH=$(find /usr -name nvcc -type f 2>/dev/null | head -1)
fi

if [ -n "$NVCC_PATH" ]; then
  CUDA_BIN_DIR=$(dirname "$NVCC_PATH")
  export CUDA_HOME=$(dirname "$CUDA_BIN_DIR")
  export CUDA_PATH=${CUDA_HOME}
  export PATH="${CUDA_BIN_DIR}:${PATH}"
  echo "Found nvcc at: ${NVCC_PATH}"
  echo "CUDA_HOME=${CUDA_HOME}"
else
  echo "ERROR: nvcc not found. Install CUDA toolkit: sudo apt-get install cuda-toolkit-12-8"
  exit 1
fi
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
