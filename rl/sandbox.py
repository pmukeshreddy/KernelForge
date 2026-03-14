"""
sandbox.py - SP1 Execution Sandbox for CUDA kernel evaluation.

Takes a CUDA kernel (KernelBench format: ModelNew class with load_inline CUDA)
and evaluates it against a PyTorch reference (Model class).

Returns: compiles, compiler_error, correct, outputs_match, runtime_ms, baseline_runtime_ms
"""
import os
import sys
import subprocess
import tempfile
import shutil
import json
import signal


def evaluate(kernel_code: str, reference_code: str, timeout: int = 120,
             n_correctness: int = 5, n_warmup: int = 3, n_timed: int = 10) -> dict:
    """
    Evaluate a generated CUDA kernel against a reference PyTorch model.
    
    Args:
        kernel_code: Python code with ModelNew(nn.Module) class + inline CUDA
        reference_code: Python code with Model(nn.Module) class + get_inputs/get_init_inputs
        timeout: Max seconds for compilation + execution
        n_correctness: Number of random inputs for correctness check
        n_warmup: Warmup iterations before timing
        n_timed: Iterations for timing measurement
    
    Returns:
        dict with compiles, compiler_error, correct, outputs_match, runtime_ms, baseline_runtime_ms
    """
    result = {
        "compiles": False,
        "compiler_error": None,
        "correct": False,
        "outputs_match": [],
        "runtime_ms": None,
        "baseline_runtime_ms": None,
    }
    
    tmpdir = tempfile.mkdtemp(prefix="kernelforge_sandbox_")
    result_path = os.path.join(tmpdir, "result.json")
    
    try:
        # Write reference and kernel to temp files
        ref_path = os.path.join(tmpdir, "model_ref.py")
        kern_path = os.path.join(tmpdir, "model_new.py")
        eval_path = os.path.join(tmpdir, "eval_kernel.py")
        
        with open(ref_path, "w") as f:
            f.write(reference_code)
        
        with open(kern_path, "w") as f:
            f.write(kernel_code)
        
        # Write the evaluation script that runs inside subprocess
        eval_script = _build_eval_script(n_correctness, n_warmup, n_timed)
        with open(eval_path, "w") as f:
            f.write(eval_script)
        
        # Run evaluation in subprocess with timeout
        env = os.environ.copy()
        env["PYTHONPATH"] = tmpdir + ":" + env.get("PYTHONPATH", "")
        # Speed up JIT compilation by targeting specific GPU arch
        if "TORCH_CUDA_ARCH_LIST" not in env:
            env["TORCH_CUDA_ARCH_LIST"] = "8.0"
        
        # Use Popen with process group for clean kill on timeout
        proc = subprocess.Popen(
            [sys.executable, "-u", eval_path],  # -u for unbuffered output
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=tmpdir,
            env=env,
            preexec_fn=os.setsid,  # New process group
        )
        
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Kill entire process group (catches nvcc child processes too)
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            
            # Check if result.json was partially written
            if os.path.exists(result_path):
                try:
                    with open(result_path) as f:
                        sub_result = json.load(f)
                    result.update(sub_result)
                except Exception:
                    pass
            result["correct"] = False
            result["compiler_error"] = f"Execution timed out after {timeout}s"
        else:
            # Normal completion — parse results
            if os.path.exists(result_path):
                with open(result_path) as f:
                    sub_result = json.load(f)
                result.update(sub_result)
            elif proc.returncode != 0:
                result["compiles"] = False
                stderr = (stderr or "").strip()
                if stderr:
                    result["compiler_error"] = stderr[-500:]
                else:
                    result["compiler_error"] = f"Process exited with code {proc.returncode}"
    
    except Exception as e:
        result["compiler_error"] = str(e)
    finally:
        # Cleanup
        shutil.rmtree(tmpdir, ignore_errors=True)
    
    return result


def _build_eval_script(n_correctness: int, n_warmup: int, n_timed: int) -> str:
    """Build the evaluation script that runs in the subprocess."""
    return f'''
import sys
import os
import json
import time
import traceback

result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result.json")

def save_result(r):
    with open(result_path, "w") as f:
        json.dump(r, f)

result = {{
    "compiles": False,
    "compiler_error": None,
    "correct": False,
    "outputs_match": [],
    "runtime_ms": None,
    "baseline_runtime_ms": None,
}}

# Step 0: Import torch
print(f"[{{time.time():.0f}}] Importing torch...", flush=True)
import torch

# Step 1: Try to import reference model
print(f"[{{time.time():.0f}}] Importing reference model...", flush=True)
try:
    import model_ref
    Model = model_ref.Model
    get_inputs = model_ref.get_inputs
    get_init_inputs = model_ref.get_init_inputs
except Exception as e:
    result["compiler_error"] = f"Reference model import failed: {{e}}"
    save_result(result)
    sys.exit(1)

# Step 2: Try to import generated kernel (this triggers JIT compilation)
print(f"[{{time.time():.0f}}] Compiling generated kernel (JIT)...", flush=True)
try:
    import model_new
    ModelNew = model_new.ModelNew
    result["compiles"] = True
    print(f"[{{time.time():.0f}}] Compilation succeeded!", flush=True)
except Exception as e:
    result["compiles"] = False
    result["compiler_error"] = traceback.format_exc()[-500:]
    print(f"[{{time.time():.0f}}] Compilation failed.", flush=True)
    save_result(result)
    sys.exit(0)

# Save intermediate result (compiles=True) so timeout knows compilation passed
save_result(result)

# Step 3: Correctness check with multiple random inputs
print(f"[{{time.time():.0f}}] Running correctness check...", flush=True)
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init_inputs = get_init_inputs()
    
    ref_model = Model(*init_inputs).to(device).eval()
    new_model = ModelNew(*init_inputs).to(device).eval()
    
    outputs_match = []
    for i in range({n_correctness}):
        torch.manual_seed(42 + i)
        inputs = get_inputs()
        inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]
        
        with torch.no_grad():
            ref_out = ref_model(*inputs)
            new_out = new_model(*inputs)
        
        if isinstance(ref_out, torch.Tensor):
            ref_out = (ref_out,)
        if isinstance(new_out, torch.Tensor):
            new_out = (new_out,)
        
        match = all(
            torch.allclose(r.float(), n.float(), atol=1e-2, rtol=1e-2)
            for r, n in zip(ref_out, new_out)
        )
        outputs_match.append(match)
    
    result["outputs_match"] = outputs_match
    result["correct"] = all(outputs_match)
    print(f"[{{time.time():.0f}}] Correctness: {{result['correct']}} ({{sum(outputs_match)}}/{{len(outputs_match)}})", flush=True)

except Exception as e:
    result["correct"] = False
    result["compiler_error"] = f"Runtime error: {{traceback.format_exc()[-300:]}}"
    save_result(result)
    sys.exit(0)

# Step 4: Timing (only if correct)
if result["correct"]:
    print(f"[{{time.time():.0f}}] Running timing benchmark...", flush=True)
    try:
        torch.manual_seed(0)
        inputs = get_inputs()
        inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]
        
        # Time reference model
        for _ in range({n_warmup}):
            with torch.no_grad():
                ref_model(*inputs)
        torch.cuda.synchronize()
        
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        
        times = []
        for _ in range({n_timed}):
            start_evt.record()
            with torch.no_grad():
                ref_model(*inputs)
            end_evt.record()
            torch.cuda.synchronize()
            times.append(start_evt.elapsed_time(end_evt))
        
        times.sort()
        result["baseline_runtime_ms"] = times[len(times) // 2]
        
        # Time new model
        for _ in range({n_warmup}):
            with torch.no_grad():
                new_model(*inputs)
        torch.cuda.synchronize()
        
        times = []
        for _ in range({n_timed}):
            start_evt.record()
            with torch.no_grad():
                new_model(*inputs)
            end_evt.record()
            torch.cuda.synchronize()
            times.append(start_evt.elapsed_time(end_evt))
        
        times.sort()
        result["runtime_ms"] = times[len(times) // 2]
        print(f"[{{time.time():.0f}}] Timing done: baseline={{result['baseline_runtime_ms']:.3f}}ms, kernel={{result['runtime_ms']:.3f}}ms", flush=True)
        
    except Exception as e:
        result["compiler_error"] = f"Timing error: {{e}}"

save_result(result)
print(f"[{{time.time():.0f}}] Eval complete.", flush=True)
'''


if __name__ == "__main__":
    # Quick test
    ref = '''
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
'''
    
    kernel = '''
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_kernel(float* output, const float* input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

torch::Tensor relu_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int n = input.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(
        output.data_ptr<float>(), input.data_ptr<float>(), n
    );
    return output;
}
"""

cpp_source = "torch::Tensor relu_cuda(torch::Tensor input);"

relu_ext = load_inline(
    name="relu_ext",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["relu_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return relu_ext.relu_cuda(x)
'''
    
    print("Testing sandbox with valid ReLU kernel...")
    r = evaluate(kernel, ref)
    print(json.dumps(r, indent=2))
