"""
sandbox.py - Execution Sandbox for CUDA kernel evaluation.

Evaluates a generated CUDA kernel (KernelBench ModelNew format) against
a PyTorch reference (Model class). Returns compilation status, correctness,
and timing results.
"""
import os
import sys
import subprocess
import tempfile
import shutil
import json
from antihack import check_security


def evaluate(kernel_code: str, reference_code: str, timeout: int = 300,
             n_correctness: int = 5, n_warmup: int = 3, n_timed: int = 10) -> dict:
    """
    Evaluate a generated CUDA kernel against a reference PyTorch model.

    Returns dict with: compiles, compiler_error, correct, outputs_match,
    runtime_ms, baseline_runtime_ms
    """
    # 1. Static Security Analysis
    is_safe, sec_err = check_security(kernel_code)
    if not is_safe:
        return {
            "compiles": False,
            "compiler_error": sec_err,
            "correct": False,
            "outputs_match": [],
            "runtime_ms": None,
            "baseline_runtime_ms": None,
        }

    result = {
        "compiles": False,
        "compiler_error": None,
        "correct": False,
        "outputs_match": [],
        "runtime_ms": None,
        "baseline_runtime_ms": None,
    }

    tmpdir = tempfile.mkdtemp(prefix="kf_sandbox_")
    result_path = os.path.join(tmpdir, "result.json")

    try:
        # Write files
        with open(os.path.join(tmpdir, "model_ref.py"), "w") as f:
            f.write(reference_code)
        with open(os.path.join(tmpdir, "model_new.py"), "w") as f:
            f.write(kernel_code)

        eval_script = _build_eval_script(result_path, n_correctness, n_warmup, n_timed)
        eval_path = os.path.join(tmpdir, "eval_kernel.py")
        with open(eval_path, "w") as f:
            f.write(eval_script)

        # Run in subprocess
        env = os.environ.copy()
        env["PYTHONPATH"] = tmpdir
        env["TORCH_EXTENSIONS_DIR"] = os.path.join(tmpdir, "torch_extensions")
        if "TORCH_CUDA_ARCH_LIST" not in env:
            env["TORCH_CUDA_ARCH_LIST"] = "8.0"

        proc = subprocess.run(
            [sys.executable, eval_path],
            capture_output=True, text=True,
            timeout=timeout, cwd=tmpdir, env=env,
        )

        # Read result
        if os.path.exists(result_path):
            with open(result_path) as f:
                result = json.load(f)
        elif proc.returncode != 0:
            result["compiler_error"] = (proc.stderr or "")[-500:]

    except subprocess.TimeoutExpired:
        if os.path.exists(result_path):
            with open(result_path) as f:
                result = json.load(f)
        result["correct"] = False
        if not result.get("compiler_error"):
            result["compiler_error"] = f"Timed out after {timeout}s"
    except Exception as e:
        result["compiler_error"] = str(e)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return result


def _build_eval_script(result_path: str, n_correctness: int, n_warmup: int, n_timed: int) -> str:
    return f'''import sys, os, json, traceback
RESULT_PATH = {repr(result_path)}

def save(r):
    with open(RESULT_PATH, "w") as f:
        json.dump(r, f)

R = {{"compiles": False, "compiler_error": None, "correct": False,
     "outputs_match": [], "runtime_ms": None, "baseline_runtime_ms": None}}

import torch

# Import reference
try:
    import model_ref
    Model, get_inputs, get_init_inputs = model_ref.Model, model_ref.get_inputs, model_ref.get_init_inputs
except Exception as e:
    R["compiler_error"] = f"Ref import failed: {{e}}"
    save(R); sys.exit(1)

# Import kernel (triggers JIT compilation)
try:
    import model_new
    ModelNew = model_new.ModelNew
    R["compiles"] = True
except Exception as e:
    # PyTorch RuntimeError contains the C++ compiler log in the string representation.
    # We take the last 2000 characters to ensure we capture the actual semantic C++ errors 
    # instead of just "ninja: build stopped"
    err_str = str(e)
    if len(err_str) > 2000:
        err_str = "..." + err_str[-2000:]
    R["compiler_error"] = err_str
    save(R); sys.exit(0)

save(R)  # Save early so timeout knows compilation passed

import threading
try:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref_model = Model(*get_init_inputs()).to(dev).eval()
    new_model = ModelNew(*get_init_inputs()).to(dev).eval()
    
    # 2. Dynamic Security Check (Thread Count)
    # Exclude the main thread and any Pytorch internal daemon threads gracefully
    base_threads = threading.active_count()

    matches = []
    for i in range({n_correctness}):
        torch.manual_seed(42 + i)
        inputs = [x.to(dev) if isinstance(x, torch.Tensor) else x for x in get_inputs()]
        with torch.no_grad():
            ro, no = ref_model(*inputs), new_model(*inputs)
        if isinstance(ro, torch.Tensor): ro = (ro,)
        if isinstance(no, torch.Tensor): no = (no,)
        matches.append(all(torch.allclose(r.float(), n.float(), atol=1e-2, rtol=1e-2) for r, n in zip(ro, no)))

    R["outputs_match"] = matches
    R["correct"] = all(matches)
    
    # Verify no hidden threads spawned during evaluation
    if threading.active_count() > base_threads:
        R["correct"] = False
        R["compiler_error"] = f"SECURITY VIOLATION: Unauthorized threads spawned ({{threading.active_count()}} active)"
        save(R); sys.exit(0)
        
except Exception as e:
    R["compiler_error"] = f"Runtime: {{traceback.format_exc()[-1500:]}}"
    save(R); sys.exit(0)

# Timing (only if correct)
if R["correct"] and torch.cuda.is_available():
    try:
        inputs = [x.to(dev) if isinstance(x, torch.Tensor) else x for x in get_inputs()]
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        for m, key in [(ref_model, "baseline_runtime_ms"), (new_model, "runtime_ms")]:
            for _ in range({n_warmup}):
                with torch.no_grad(): m(*inputs)
            torch.cuda.synchronize()
            times = []
            for _ in range({n_timed}):
                s.record()
                with torch.no_grad(): m(*inputs)
                e.record(); torch.cuda.synchronize()
                times.append(s.elapsed_time(e))
            times.sort()
            R[key] = times[len(times)//2]
    except Exception as ex:
        R["compiler_error"] = f"Timing: {{ex}}"

save(R)
'''


if __name__ == "__main__":
    ref = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.relu(x)

def get_inputs():
    return [torch.randn(16, 16384)]

def get_init_inputs():
    return []
"""

    kernel = """
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = \"\"\"
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_kernel(float* output, const float* input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = fmaxf(0.0f, input[idx]);
}

torch::Tensor relu_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int n = input.numel();
    relu_kernel<<<(n+255)/256, 256>>>(output.data_ptr<float>(), input.data_ptr<float>(), n);
    return output;
}
\"\"\"

relu_ext = load_inline(
    name="relu_ext",
    cpp_sources="torch::Tensor relu_cuda(torch::Tensor input);",
    cuda_sources=cuda_source,
    functions=["relu_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return relu_ext.relu_cuda(x)
"""

    print("Testing sandbox...")
    r = evaluate(kernel, ref)
    print(json.dumps(r, indent=2))
