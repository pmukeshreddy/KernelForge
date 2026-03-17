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
pip install flash-attn --no-build-isolation || echo "Flash attention build failed, proceeding without it."

echo "4. Logging into HuggingFace..."
huggingface-cli login --token "${HF_TOKEN}" || echo "HF login failed."

echo "5. Generating SFT training data (with nvcc compilation check)..."
python3 generate_sft_data.py

echo "6. Training SFT on generated pairs..."
python3 train_sft.py

echo "7. Verifying compilation ability..."
python3 verify_compilation.py

echo "Phase 1 SFT Pipeline finished."
