"""
sandbox.py - Execution Sandbox for CUDA kernel evaluation.

Evaluates a generated CUDA kernel (KernelBench ModelNew format) against
a PyTorch reference (Model class). Returns compilation status, correctness,
and timing results.
"""
import os
import sys
import random
import subprocess
import tempfile
import shutil
import json
from antihack import check_security


def evaluate(kernel_code: str, reference_code: str, timeout: int = 300,
             n_correctness: int = 10, n_warmup: int = 3, n_timed: int = 10) -> dict:
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
        
        # Each sandbox gets its own cache subdir to avoid parallel-eval race conditions
        # where two processes compile different kernels with the same name= simultaneously.
        cache_dir = os.path.join(tmpdir, ".kf_cache")
        os.makedirs(cache_dir, exist_ok=True)
        env["TORCH_EXTENSIONS_DIR"] = cache_dir
        if "TORCH_CUDA_ARCH_LIST" not in env:
            env["TORCH_CUDA_ARCH_LIST"] = "9.0"

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
            # Clean compiler output: extract only error lines, strip build flag noise
            raw_err = proc.stderr or ""
            error_lines = []
            for line in raw_err.split('\n'):
                line_stripped = line.strip()
                # Keep lines with actual errors, skip build command flags
                if any(kw in line_stripped for kw in ['error:', 'Error:', 'undefined', 'FAILED', 'fatal']):
                    # Strip leading path noise, keep just filename and error
                    if '.cu(' in line_stripped:
                        line_stripped = line_stripped[line_stripped.rfind('/', 0, line_stripped.find('.cu('))+1:]
                    error_lines.append(line_stripped)
            if error_lines:
                result["compiler_error"] = '\n'.join(error_lines[-10:])  # Last 10 error lines
            else:
                result["compiler_error"] = raw_err[-500:]  # Fallback

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
    return f'''import sys, os, json, random, traceback
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
    # load_inline throws a RuntimeError whose string contains the entire ninja
    # build log (thousands of chars of -isystem flags) followed by the actual
    # CUDA/C++ error. Extract only the meaningful lines.
    err_str = str(e)
    error_lines = []
    for line in err_str.split('\\n'):
        s = line.strip()
        if not s:
            continue
        # Skip pure build-flag lines
        if s.startswith('-isystem') or s.startswith('-I/') or s.startswith('-D__'):
            continue
        if any(kw in s for kw in ['error:', 'Error:', 'undefined', 'FAILED',
                                   'fatal error', 'note:', 'warning:', 'ninja:']):
            # Strip leading temp-path noise, keep filename(line): error
            if '.cu(' in s:
                s = s[s.rfind('/', 0, s.find('.cu(')) + 1:]
            error_lines.append(s)
    if error_lines:
        R["compiler_error"] = '\\n'.join(error_lines[-20:])
    else:
        # Fallback: last lines usually have "error: X" summary
        last = [l.strip() for l in err_str.strip().split('\\n')[-15:] if l.strip()]
        R["compiler_error"] = '\\n'.join(last)
    save(R); sys.exit(0)

save(R)  # Save early so timeout knows compilation passed

import threading
try:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref_model = Model(*get_init_inputs()).to(dev).eval()
    new_model = ModelNew(*get_init_inputs()).to(dev).eval()
    
    # Align parameters so both models use the same randomly initialized weights natively
    try:
        new_model.load_state_dict(ref_model.state_dict())
    except Exception as e:
        R["compiler_error"] = "Failed to align parameters: " + str(e)
        save(R); sys.exit(0)
    
    # 2. Dynamic Security Check (Thread Count)
    # Exclude the main thread and any Pytorch internal daemon threads gracefully
    base_threads = threading.active_count()

    matches = []
    correctness_detail = ""
    for i in range({n_correctness}):
        torch.manual_seed(random.randint(0, 100000))
        inputs = [x.to(dev) if isinstance(x, torch.Tensor) else x for x in get_inputs()]
        with torch.no_grad():
            ro, no = ref_model(*inputs), new_model(*inputs)
        if isinstance(ro, torch.Tensor): ro = (ro,)
        if isinstance(no, torch.Tensor): no = (no,)
        trial_ok = True
        for t_idx, (r, n) in enumerate(zip(ro, no)):
            rf, nf = r.float(), n.float()
            if rf.shape != nf.shape:
                trial_ok = False
                correctness_detail = f"Output tensor {{t_idx}}: SHAPE MISMATCH - expected shape {{list(rf.shape)}}, got {{list(nf.shape)}}."
                break
            if not torch.allclose(rf, nf, atol=1e-3, rtol=1e-3):
                trial_ok = False
                diff = (rf - nf).abs()
                max_err = diff.max().item()
                max_pos = diff.argmax().item()
                # Convert flat index to multi-dim for readability
                shape = rf.shape
                coords = []
                flat = max_pos
                for dim_size in reversed(shape):
                    coords.append(flat % dim_size)
                    flat //= dim_size
                coords = list(reversed(coords))
                exp_val = rf.flatten()[max_pos].item()
                got_val = nf.flatten()[max_pos].item()
                correctness_detail = f"Output tensor {{t_idx}}: max_abs_error={{max_err:.6f}} at position {{coords}}, expected={{exp_val:.6f}}, got={{got_val:.6f}}, shape={{list(shape)}}"
                break
        matches.append(trial_ok)
        if not trial_ok:
            break

    R["outputs_match"] = matches
    R["correct"] = all(matches)
    if not R["correct"] and correctness_detail:
        R["compiler_error"] = f"Correctness Failed: {{correctness_detail}}"
    
    # Verify no hidden threads spawned during evaluation
    if threading.active_count() > base_threads:
        R["correct"] = False
        R["compiler_error"] = f"SECURITY VIOLATION: Unauthorized threads spawned ({{threading.active_count()}} active)"
        save(R); sys.exit(0)
        
except Exception as e:
    # Extract useful info from traceback without tensor data dumps.
    # Tensor dumps appear at the END of the message; take the FIRST part instead.
    tb = traceback.format_exc()
    lines = tb.strip().split('\\n')
    # Get the exception class + message (last non-empty line) truncated
    exc_line = next((l.strip() for l in reversed(lines) if l.strip()), str(e))
    exc_line = exc_line[:300]
    # Get up to 3 File/line context lines
    file_lines = [l.strip() for l in lines if l.strip().startswith('File ')][-3:]
    R["compiler_error"] = "Runtime: " + "\\n".join([exc_line] + file_lines)
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
