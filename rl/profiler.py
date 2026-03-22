"""
profiler.py - SP4 Profiler Parser for RL

Wraps Nvidia Nsight Compute (ncu) to extract hardware performance metrics
(compute throughput, memory throughput, occupancy) and parse them into
actionable, natural language feedback for the RL agent.
"""
import os
import sys
import subprocess
import tempfile
import shutil
import csv
from typing import Dict, Any


def profile_kernel(kernel_code: str, reference_code: str, timeout: int = 120) -> str:
    """
    Profile a generated CUDA kernel using ncu and generate actionable feedback.
    
    Args:
        kernel_code: Code containing ModelNew and load_inline
        reference_code: Code containing Model and get_inputs
        timeout: Max seconds to run the profiler
        
    Returns:
        Readable string with bottleneck analysis and recommendations.
    """
    tmpdir = tempfile.mkdtemp(prefix="kf_profiler_")
    
    try:
        # Write files
        with open(os.path.join(tmpdir, "model_ref.py"), "w") as f:
            f.write(reference_code)
        with open(os.path.join(tmpdir, "model_new.py"), "w") as f:
            f.write(kernel_code)

        # Build execution script (just 1 iteration is enough for ncu)
        eval_script = _build_profiler_script()
        eval_path = os.path.join(tmpdir, "run_profiler.py")
        with open(eval_path, "w") as f:
            f.write(eval_script)

        # Run ncu targeting specific metrics via CSV output
        env = os.environ.copy()
        env["PYTHONPATH"] = tmpdir
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".kf_compilation_cache")
        os.makedirs(cache_dir, exist_ok=True)
        env["TORCH_EXTENSIONS_DIR"] = cache_dir
        if "TORCH_CUDA_ARCH_LIST" not in env:
            # Auto-detect GPU arch (e.g. 8.9 for RTX PRO 6000, 9.0 for H100)
            try:
                import torch as _torch
                if _torch.cuda.is_available():
                    major, minor = _torch.cuda.get_device_capability(0)
                    env["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
                else:
                    env["TORCH_CUDA_ARCH_LIST"] = "8.9"
            except Exception:
                env["TORCH_CUDA_ARCH_LIST"] = "8.9"

        # Explicitly profile the custom kernel, not PyTorch overhead
        metrics = [
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
            "sm__warps_active.avg.pct_of_peak_sustained_active"
        ]
        
        # Find ncu: prefer /usr/local/cuda/bin/ncu over system ncu
        ncu_bin = shutil.which("ncu")
        for candidate in ["/usr/local/cuda/bin/ncu", "/usr/local/cuda-12/bin/ncu"]:
            if os.path.isfile(candidate):
                ncu_bin = candidate
                break
        if ncu_bin is None:
            ncu_bin = "ncu"

        # ncu needs sudo for GPU performance counters
        ncu_cmd = [
            "sudo", ncu_bin,
            "--target-processes", "all",
            "--metrics", ",".join(metrics),
            "--csv",
            "--page", "raw",
            sys.executable, eval_path
        ]

        # Use Popen to allow process group kill if nvcc/ncu hangs
        try:
            proc = subprocess.Popen(
                ncu_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=tmpdir,
                env=env,
                preexec_fn=os.setsid, 
            )
        except FileNotFoundError:
            raise RuntimeError(
                "\n" + "="*60 + "\n"
                "FATAL: 'ncu' (Nsight Compute) is NOT INSTALLED.\n"
                "The optimization loop CANNOT function without hardware profiling.\n"
                "Install it with:\n"
                "  apt-get update && apt-get install -y nsight-compute\n"
                "Or:\n"
                "  apt-get install -y cuda-nsight-compute-12-*\n"
                + "="*60
            )
        
        stdout, stderr = "", ""
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            import signal
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()
            return "Profiler Error: Timed out during execution."
        
        if proc.returncode != 0:
            combined = (stderr.strip() + "\n" + stdout.strip()).strip()
            if "not found" in combined.lower() or "no such file or directory" in combined.lower():
                return "Profiler Error: 'ncu' command not found. Ensure Nsight Compute is installed."
            # Grab the end of combined output for debugging
            clean_err = combined[-500:] if combined else f"exit code {proc.returncode}"
            return f"Profiler Execution Failed:\n{clean_err}"

        # Parse CSV output for the target kernel
        # ncu CSV format is verbose. We look for rows with our metric names.
        extracted_metrics = _parse_ncu_csv(stdout)
        
        if not extracted_metrics:
            return "Profiler Error: Could not extract metric data from 'ncu' output."
            
        return _generate_feedback(extracted_metrics)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _parse_ncu_csv(csv_text: str) -> Dict[str, float]:
    """Parse output from ncu --csv to extract our 3 key metrics."""
    metrics = {}
    
    # ncu outputs warnings and infos before the CSV starts with "ID,Process ID,..."
    # We strip all lines until we see "ID,"
    lines = csv_text.split('\n')
    csv_start = -1
    for i, line in enumerate(lines):
        if line.startswith('"ID",') or line.startswith('ID,'):
            csv_start = i
            break
            
    if csv_start == -1:
        return {}
        
    reader = csv.DictReader(lines[csv_start:])
    for row in reader:
        # Format 1 (Newer NCU): Rows have "Metric Name" and "Metric Value"
        metric_col = next((k for k in row.keys() if k and "Metric Name" in k), None)
        val_col = next((k for k in row.keys() if k and "Metric Value" in k), None)
        
        if metric_col and val_col:
            name = str(row[metric_col]).strip()
            val_str = str(row[val_col]).strip().replace('%', '')
            try:
                if "sm__throughput" in name:
                    metrics["compute"] = float(val_str)
                elif "gpu__compute_memory" in name:
                    metrics["memory"] = float(val_str)
                elif "sm__warps_active" in name:
                    metrics["occupancy"] = float(val_str)
            except ValueError:
                pass
        
        # Format 2 (Older NCU 2022.x): Metrics are direct column headers
        else:
            for col_name, val in row.items():
                if not col_name: continue
                name = str(col_name).strip()
                val_str = str(val).strip().replace('%', '')
                try:
                    if "sm__throughput" in name:
                        metrics["compute"] = float(val_str)
                    elif "gpu__compute_memory" in name:
                        metrics["memory"] = float(val_str)
                    elif "sm__warps_active" in name:
                        metrics["occupancy"] = float(val_str)
                except ValueError:
                    pass
                    
    return metrics


def _generate_feedback(metrics: Dict[str, float]) -> str:
    """Bottleneck-aware profiler feedback with actionable optimization techniques."""
    compute = metrics.get("compute", 0.0)
    memory = metrics.get("memory", 0.0)
    occupancy = metrics.get("occupancy", 0.0)

    feedback = f"--- Hardware Profiler ---\n"
    feedback += f"Memory Throughput:  {memory:>5.1f}% of peak\n"
    feedback += f"Compute Throughput: {compute:>5.1f}% of peak\n"
    feedback += f"Warp Occupancy:     {occupancy:>5.1f}% of theoretical peak\n"

    # Diagnose bottleneck and suggest techniques based on actual metrics
    feedback += f"\n--- Bottleneck Analysis ---\n"

    if memory < 30 and compute < 30:
        # Both low → kernel is latency-bound (not doing enough work)
        feedback += (
            "Bottleneck: LATENCY-BOUND (both memory and compute underutilized).\n"
            "The kernel is not doing enough work per launch.\n"
            "Techniques: process multiple elements per thread (loop over elements), "
            "use float4/int4 vectorized loads/stores (128-bit transactions), "
            "fuse multiple operations into a single kernel."
        )
    elif memory > compute * 1.5:
        # Memory-bound
        feedback += "Bottleneck: MEMORY-BOUND.\n"
        if memory >= 80:
            feedback += (
                "Memory throughput is near peak — this kernel is already well-optimized "
                "for a memory-bound workload. Further speedup requires reducing the number "
                "of memory passes (fuse multiple operations into one kernel to avoid writing "
                "and re-reading intermediate results) or reducing total bytes moved "
                "(e.g., in-place operations, skip unnecessary copies)."
            )
        elif memory < 50:
            feedback += (
                "Memory bandwidth is underutilized.\n"
                "Techniques: use float4 vectorized loads/stores (reinterpret_cast<float4*>), "
                "ensure coalesced access patterns (consecutive threads access consecutive addresses), "
                "process multiple elements per thread to hide memory latency."
            )
        else:
            feedback += (
                "Memory bandwidth is the primary limiter.\n"
                "Techniques: use float4 vectorized loads/stores for 128-bit transactions, "
                "use __ldg() for read-only data (L2 cache hint), "
                "fuse sequential operations to avoid intermediate global memory writes, "
                "process multiple elements per thread to overlap compute and memory."
            )
    elif compute > memory * 1.5:
        # Compute-bound
        feedback += "Bottleneck: COMPUTE-BOUND.\n"
        if compute < 50:
            feedback += (
                "Compute units are underutilized.\n"
                "Techniques: increase parallelism (more threads/blocks), "
                "use warp-level primitives (__shfl_down_sync for reductions), "
                "unroll inner loops (#pragma unroll), "
                "reduce branch divergence within warps."
            )
        else:
            feedback += (
                "Compute is the primary limiter.\n"
                "Techniques: reduce arithmetic intensity (precompute constants, "
                "use fast intrinsics like __fmaf_rn, __expf, __rsqrtf), "
                "trade precision where acceptable (float vs double), "
                "use warp shuffle (__shfl_down_sync) instead of shared memory for reductions."
            )
    else:
        # Balanced
        feedback += (
            "Bottleneck: BALANCED (compute and memory roughly equal).\n"
            "Techniques: overlap compute and memory with software pipelining, "
            "use shared memory to stage data and reduce global accesses, "
            "increase elements per thread to improve instruction-level parallelism."
        )

    # Occupancy advice
    if occupancy < 40:
        feedback += (
            f"\n\nLow occupancy ({occupancy:.0f}%): too few active warps. "
            "Reduce registers per thread (fewer local variables, simpler logic), "
            "reduce shared memory per block, or use smaller block sizes (128 instead of 256)."
        )

    return feedback


def _build_profiler_script() -> str:
    """Script to run exactly 1 iteration of the custom kernel for ncu."""
    return '''
import sys, os, traceback, torch

try:
    import model_ref
    ModelNew = __import__("model_new").ModelNew
    
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelNew(*model_ref.get_init_inputs()).to(dev).eval()
    
    # Needs to be a bit hot for accurate profiling
    inputs = [x.to(dev) if isinstance(x, torch.Tensor) else x for x in model_ref.get_inputs()]
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            model(*inputs)
    torch.cuda.synchronize()
    
    # Profiler region
    torch.cuda.cudart().cudaProfilerStart()
    with torch.no_grad():
        model(*inputs)
    torch.cuda.cudart().cudaProfilerStop()
    
except Exception as e:
    print(f"Profiler target failed: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
'''
