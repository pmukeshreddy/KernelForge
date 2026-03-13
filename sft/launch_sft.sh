#!/bin/bash
set -e

echo "=== KernelForge Phase 1: SFT Pipeline ==="

echo "1. Creating Python virtual environment..."
python3 -m venv .venv
source ./.venv/bin/activate

echo "2. Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "3. Installing flash-attn..."
pip install wheel setuptools
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu128torch2.10-cp310-cp310-linux_x86_64.whl || echo "Flash attention wheel failed, proceeding without it."

echo "4. Logging into HuggingFace..."
huggingface-cli login --token "${HF_TOKEN}" || echo "HF login failed."

echo "5. Generating SFT training data (vLLM batch mode)..."
python3 generate_sft_data.py

echo "6. Training SFT on generated pairs..."
python3 train_sft.py

echo "7. Verifying compilation ability..."
python3 verify_compilation.py

echo "Phase 1 SFT Pipeline finished."
