"""
llm_feedback.py — LLM-based feedback for GRPO training.

Replaces BM25 RAG with a local GGUF model (via llama-cpp-python) that:
1. Diagnoses specific bugs in broken CUDA kernels (error turns)
2. Identifies specific bottlenecks in correct-but-slow kernels (optimization turns)

The model runs on CPU (n_gpu_layers=0) to avoid competing with training for GPU memory.
Load once at startup, call directly — no external server needed.

Usage:
    fb = LLMFeedback("/path/to/model.gguf")
    hint = fb.diagnose_error(task_description, broken_code, error_message)
    hint = fb.suggest_optimization(task_description, code, speedup, profiler_info)
"""

import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

# Optional import — graceful fallback if llama-cpp-python not installed
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False


# ── Prompt templates ─────────────────────────────────────────────────────────

_DIAGNOSE_SYSTEM = """\
You are a CUDA kernel debugging expert. You will be shown a broken CUDA kernel \
and the error it produced. Your job is to diagnose the root cause concisely.

Rules:
- Identify the SPECIFIC bug (wrong indexing, missing sync, type mismatch, etc.)
- Be concrete: reference line numbers or variable names from the code
- Keep your response to 2-4 sentences
- Do NOT provide corrected code — only diagnose the problem
- If the error is a compiler error, focus on the C++ syntax/API issue
- If the error is a correctness error, focus on the algorithmic/indexing bug"""

_DIAGNOSE_USER = """\
Task: Implement a CUDA kernel for the following PyTorch operation:
{task}

Generated kernel code:
```python
{code}
```

Error:
{error}

Diagnose the root cause of this error in 2-4 sentences."""

_OPTIMIZE_SYSTEM = """\
You are a CUDA performance optimization expert. You will be shown a correct but \
potentially slow CUDA kernel. Your job is to identify its #1 performance bottleneck \
and suggest ONE specific optimization.

Rules:
- Analyze the actual code structure (memory access pattern, thread utilization, etc.)
- Identify the SINGLE most impactful bottleneck
- Suggest ONE concrete optimization technique with a brief explanation
- Keep your response to 3-5 sentences
- Do NOT rewrite the kernel — just describe what to change and why
- Reference specific parts of the code (variable names, loop structures, etc.)"""

_OPTIMIZE_USER = """\
Task: Implement a CUDA kernel for the following PyTorch operation:
{task}

Current kernel code (correct, {speedup:.2f}x vs PyTorch):
```python
{code}
```
{profiler_section}
Identify the #1 performance bottleneck and suggest ONE specific optimization."""


class LLMFeedback:
    """LLM-based feedback using a local GGUF model via llama-cpp-python.

    Loads the model on CPU to avoid GPU memory conflicts with training.
    Thread-safe for concurrent calls from multiple trajectories.
    """

    def __init__(self, model_path: str, n_ctx: int = 4096, timeout: float = 120.0):
        """
        Args:
            model_path: Path to GGUF model file
            n_ctx: Context window size (4096 is enough for code + error + response)
            timeout: Max seconds per LLM call before giving up
        """
        self.timeout = timeout
        self._llm = None
        self._lock = threading.Lock()  # llama-cpp is not thread-safe

        if not model_path:
            print("[LLM Feedback] No model path — LLM feedback disabled")
            return

        if not LLAMA_CPP_AVAILABLE:
            print("[LLM Feedback] llama-cpp-python not installed. "
                  "Install with: pip install llama-cpp-python")
            return

        if not os.path.isfile(model_path):
            print(f"[LLM Feedback] Model file not found: {model_path}")
            return

        t0 = time.time()
        print(f"[LLM Feedback] Loading {os.path.basename(model_path)} on CPU...", flush=True)
        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=0,      # CPU only — GPU stays free for training
            n_threads=4,          # reasonable CPU parallelism
            verbose=False,
        )
        print(f"[LLM Feedback] Model loaded in {time.time()-t0:.1f}s", flush=True)

    @property
    def available(self) -> bool:
        return self._llm is not None

    def _call(self, system: str, user: str, max_tokens: int = 250,
              temperature: float = 0.3) -> str:
        """Make a single LLM call with timeout. Returns empty string on failure."""
        if not self.available:
            return ""

        result = [None]
        error = [None]

        def _generate():
            try:
                with self._lock:
                    resp = self._llm.create_chat_completion(
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=0.9,
                    )
                result[0] = resp["choices"][0]["message"]["content"].strip()
            except Exception as e:
                error[0] = str(e)

        thread = threading.Thread(target=_generate, daemon=True)
        thread.start()
        thread.join(timeout=self.timeout)

        if thread.is_alive():
            print(f"[LLM Feedback] Timeout after {self.timeout}s")
            return ""
        if error[0]:
            print(f"[LLM Feedback] Error: {error[0]}")
            return ""
        return result[0] or ""

    def diagnose_error(self, task: str, code: str, error: str) -> str:
        """Diagnose why a CUDA kernel failed.

        Args:
            task: The PyTorch operation description (prompt text)
            code: The generated Python/CUDA code that failed
            error: Compiler error or correctness error message

        Returns:
            2-4 sentence diagnosis, or empty string on failure
        """
        # Truncate inputs to fit context window
        task_trunc = task[:2000]
        code_trunc = code[:3000] if code else "(no code extracted)"
        error_trunc = error[:1000] if error else "Unknown error"

        user_msg = _DIAGNOSE_USER.format(
            task=task_trunc,
            code=code_trunc,
            error=error_trunc,
        )
        t0 = time.time()
        result = self._call(_DIAGNOSE_SYSTEM, user_msg, max_tokens=400)
        if result:
            print(f"[LLM Feedback] Diagnosis ({time.time()-t0:.1f}s): {result[:150]}...")
        return result

    def suggest_optimization(self, task: str, code: str,
                              speedup: float, profiler_info: str = "",
                              technique_hint: str = "",
                              temperature: float = 0.3) -> str:
        """Suggest optimization for a correct-but-slow kernel.

        Args:
            task: The PyTorch operation description
            code: The correct CUDA kernel code
            speedup: Current speedup vs PyTorch (e.g., 0.8 = 80% of PyTorch speed)
            profiler_info: Optional profiler output (bottleneck analysis)
            technique_hint: Optional specific technique to focus on (e.g. rule name)
            temperature: Sampling temperature (vary across trajectories for diversity)

        Returns:
            3-5 sentence optimization suggestion, or empty string on failure
        """
        code_trunc = code[:3000] if code else ""
        task_trunc = task[:2000]
        profiler_section = (
            f"\n\nProfiler analysis:\n{profiler_info[:500]}"
            if profiler_info else ""
        )

        # If a specific technique is given, append it to the prompt so the LLM
        # gives targeted advice instead of generic suggestions.
        technique_section = ""
        if technique_hint:
            technique_section = (
                f"\n\nFocus your suggestion on applying this technique: "
                f"**{technique_hint}**. Explain how it applies to this specific kernel."
            )

        user_msg = _OPTIMIZE_USER.format(
            task=task_trunc,
            code=code_trunc,
            speedup=speedup,
            profiler_section=profiler_section,
        ) + technique_section
        t0 = time.time()
        result = self._call(_OPTIMIZE_SYSTEM, user_msg, max_tokens=250,
                            temperature=temperature)
        if result:
            print(f"[LLM Feedback] Optimization hint ({time.time()-t0:.1f}s): {result[:150]}...")
        return result

    def diagnose_batch(self, items: list[dict], max_workers: int = 2) -> list[str]:
        """Diagnose multiple errors concurrently.

        Args:
            items: list of dicts with keys: task, code, error
            max_workers: concurrent threads (limited by CPU — 2 is usually optimal)

        Returns:
            list of diagnosis strings (empty string for failures)
        """
        if not self.available or not items:
            return [""] * len(items)

        # Sequential with the lock — llama-cpp isn't thread-safe for the same model
        # But we still use the batch API for a clean interface
        results = []
        for item in items:
            r = self.diagnose_error(
                task=item.get("task", ""),
                code=item.get("code", ""),
                error=item.get("error", ""),
            )
            results.append(r)
        return results


def _format_llm_hint(hint: str, hint_type: str = "diagnosis") -> str:
    """Format LLM feedback for injection into turn feedback."""
    if not hint:
        return ""
    label = "Bug Diagnosis" if hint_type == "diagnosis" else "Optimization Hint"
    return (
        f"\n\n--- {label} (from code analysis) ---\n"
        f"{hint}\n"
        f"--- End {label} ---"
    )


# ── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    model_path = sys.argv[1] if len(sys.argv) > 1 else ""
    if not model_path:
        print("Usage: python llm_feedback.py <path_to_gguf_model>")
        print("\nRunning without model (testing prompt construction only)...\n")

        # Test prompt construction
        user_msg = _DIAGNOSE_USER.format(
            task="torch.nn.functional.relu(x)",
            code='cuda_source = """\n__global__ void relu_kernel(float* x, int n) {\n    int idx = threadIdx.x;\n    if (idx < n) x[idx] = fmaxf(0.0f, x[idx]);\n}\n"""',
            error="RuntimeError: CUDA error: misaligned address",
        )
        print("=== Diagnosis Prompt ===")
        print(f"System ({len(_DIAGNOSE_SYSTEM)} chars):")
        print(_DIAGNOSE_SYSTEM[:200] + "...")
        print(f"\nUser ({len(user_msg)} chars):")
        print(user_msg[:500] + "...")

        print("\n=== Format test ===")
        print(_format_llm_hint("The kernel uses threadIdx.x without blockIdx.x, "
                                "so only the first block's threads are indexed. "
                                "Grid-stride or block-aware indexing is needed."))
        print("\nDone.")
        sys.exit(0)

    fb = LLMFeedback(model_path)
    if not fb.available:
        print("Model failed to load!")
        sys.exit(1)

    # Test error diagnosis
    print("\n=== Testing Error Diagnosis ===")
    hint = fb.diagnose_error(
        task="torch.nn.functional.relu(x) — element-wise ReLU activation",
        code=('import torch\nimport torch.nn as nn\n'
              'from torch.utils.cpp_extension import load_inline\n\n'
              'cuda_source = """\n'
              '#include <torch/extension.h>\n'
              '__global__ void relu_kernel(float* input, float* output, int n) {\n'
              '    int idx = threadIdx.x;\n'
              '    if (idx < n) output[idx] = fmaxf(0.0f, input[idx]);\n'
              '}\n'
              'torch::Tensor relu_cuda(torch::Tensor input) {\n'
              '    auto output = torch::empty_like(input);\n'
              '    int n = input.numel();\n'
              '    relu_kernel<<<1, 256>>>(input.data_ptr<float>(), output.data_ptr<float>(), n);\n'
              '    return output;\n'
              '}\n"""\n'),
        error="Outputs do not match reference. wrong_frac=0.95, max_abs_error=3.14159",
    )
    print(f"Diagnosis: {hint}")
    print(f"Formatted: {_format_llm_hint(hint)}")

    # Test optimization suggestion
    print("\n=== Testing Optimization Suggestion ===")
    hint2 = fb.suggest_optimization(
        task="torch.nn.functional.relu(x) — element-wise ReLU activation",
        code=('cuda_source = """\n'
              '__global__ void relu_kernel(const float* input, float* output, int n) {\n'
              '    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n'
              '    if (idx < n) output[idx] = fmaxf(0.0f, input[idx]);\n'
              '}\n"""\n'),
        speedup=0.85,
        profiler_info="Memory-bound kernel. Global memory throughput: 450 GB/s (56% of peak).",
    )
    print(f"Optimization: {hint2}")
    print(f"Formatted: {_format_llm_hint(hint2, 'optimization')}")
