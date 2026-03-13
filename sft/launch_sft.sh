#!/bin/bash
set -e

# Setup environment for Phase 1 SFT Training

echo "1. Creating Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "2. Installing requirements..."
pip install -r requirements.txt

echo "3. Installing flash-attn from pre-built wheel (much faster)..."
# Note: You may need to update this URL to match your exact Python, PyTorch, and CUDA versions
# Find wheels at: https://github.com/Dao-AILab/flash-attention/releases
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.6/flash_attn-2.5.6+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl || echo "Wheel installation failed, proceeding without it."

echo "4. Logging into HuggingFace (required for gated models like Qwen)..."
# User needs to ensure HF_TOKEN is set in the environment, or run huggingface-cli login manually
if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN environment variable is not set. You might need to authenticate manually."
fi

# Run the SFT script using accelerate if configured, or just python directly
echo "5. Starting SFT Training..."
# If using accelerate: accelerate launch train_sft.py
python3 train_sft.py

echo "6. Training complete. Verifying compilation ability on evaluation set..."
python3 verify_compilation.py

echo "Phase 1 SFT Pipeline finished successfully."
