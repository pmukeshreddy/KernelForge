#!/bin/bash
set -e

echo "=== KernelForge Phase 1: SFT Pipeline ==="

echo "1. Creating Python virtual environment..."
python3 -m venv .venv
source ./.venv/bin/activate

echo "2. Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "3. Installing flash-attn (Optional but recommended for H100)..."
pip install wheel setuptools
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu128torch2.10-cp310-cp310-linux_x86_64.whl || echo "Flash attention wheel installation failed, proceeding without it."

echo "4. Logging into HuggingFace (required for gated models like Qwen)..."
huggingface-cli login --token "${HF_TOKEN}" || echo "HF login failed, some models may not be accessible."

echo "5. Step 1: Generating SFT training data via rejection sampling..."
echo "   This generates CUDA kernels from base Qwen3-8B and filters for compilable ones."
python3 generate_sft_data.py

echo "6. Step 2: Training SFT on generated pairs..."
python3 train_sft.py

echo "7. Step 3: Verifying compilation ability on evaluation set..."
python3 verify_compilation.py

echo "Phase 1 SFT Pipeline finished successfully."
