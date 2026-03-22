#!/bin/bash
# One-time server setup for KernelForge GRPO training.
# Run this after every fresh server/instance start:
#   bash setup_server.sh
set -e

echo "=== KernelForge Server Setup ==="

# 1. CUDA toolkit path
if [ -d /usr/local/cuda/bin ]; then
    export PATH=/usr/local/cuda/bin:$PATH
    echo "[OK] CUDA path: $(which nvcc)"
else
    echo "[INSTALL] Installing CUDA toolkit..."
    sudo apt update && sudo apt install -y cuda-toolkit-12-8
    export PATH=/usr/local/cuda/bin:$PATH
fi

# 2. ncu (Nsight Compute) profiler
if /usr/local/cuda/bin/ncu --version &>/dev/null; then
    echo "[OK] ncu: $(/usr/local/cuda/bin/ncu --version | head -1)"
else
    echo "[INSTALL] Installing Nsight Compute..."
    sudo apt update && sudo apt install -y cuda-nsight-compute-12-8
fi

# 3. GPU profiling permissions (survives reboot via modprobe.d)
echo "[SETUP] GPU profiler permissions..."
sudo sh -c 'echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" > /etc/modprobe.d/nvidia-profiler.conf'
# Apply immediately without driver reload
sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0 2>/dev/null || true

# 4. Passwordless sudo for ncu
if ! sudo -n /usr/local/cuda/bin/ncu --version &>/dev/null; then
    echo "[SETUP] Passwordless sudo for ncu..."
    sudo sh -c 'echo "'"$USER"' ALL=(ALL) NOPASSWD: /usr/local/cuda/bin/ncu" > /etc/sudoers.d/ncu'
    sudo chmod 440 /etc/sudoers.d/ncu
fi
echo "[OK] ncu sudo: passwordless"

# 5. Test profiler actually works
echo "[TEST] Running ncu profiler test..."
RESULT=$(sudo /usr/local/cuda/bin/ncu --target-processes all --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed --csv python3 -c "import torch; x=torch.randn(1000,device='cuda'); y=x*x" 2>&1)
if echo "$RESULT" | grep -q "ERR_NVGPUCTRPERM"; then
    echo "[WARN] Profiler permission still blocked. May need server reboot for modprobe.d to take effect."
elif echo "$RESULT" | grep -q "sm__throughput"; then
    echo "[OK] Profiler working!"
else
    echo "[WARN] Profiler test inconclusive. Output:"
    echo "$RESULT" | tail -5
fi

# 6. PyTorch memory config
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo ""
echo "=== Setup complete. Run training with: bash run_grpo.sh ==="
