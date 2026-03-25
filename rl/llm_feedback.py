"""
llm_feedback.py — LLM-based feedback for GRPO training.

Replaces BM25 RAG with a local GGUF model (via llama-cpp-python) that:
1. Diagnoses specific bugs in broken CUDA kernels (error turns)
2. Identifies specific bottlenecks in correct-but-slow kernels (optimization turns)

The model runs on CPU (n_gpu_layers=0) to avoid competing with training for GPU memory.
Loads a POOL of N model instances at startup for parallel inference.

Usage:
    fb = LLMFeedback("/path/to/model.gguf", n_workers=2)
    hint = fb.diagnose_error(task_description, broken_code, error_message)
    hints = fb.diagnose_batch([{"task": ..., "code": ..., "error": ...}, ...])
    hints = fb.optimize_batch([{"task": ..., "code": ..., "speedup": ..., ...}, ...])
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
- Start your response with the bug itself. NEVER begin with "Let me", "Looking at", or any preamble.
- Identify the SPECIFIC bug (wrong indexing, missing sync, type mismatch, etc.)
- Be concrete: reference variable names from the code
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
- Start with the bottleneck. NEVER begin with "Let me", "Looking at", or any preamble.
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


# ── Worker: one Llama instance + its own lock ────────────────────────────────

class _LLMWorker:
    """A single Llama model instance with its own lock for thread safety."""

    __slots__ = ("llm", "lock", "worker_id")

    def __init__(self, model_path: str, n_ctx: int, n_threads: int, worker_id: int):
        self.worker_id = worker_id
        self.lock = threading.Lock()
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=0,
            n_threads=n_threads,
            verbose=False,
        )

    def generate(self, system: str, user: str,
                 max_tokens: int, temperature: float) -> str:
        with self.lock:
            resp = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
            )
        return resp["choices"][0]["message"]["content"].strip()


class LLMFeedback:
    """LLM-based feedback using a pool of local GGUF model instances.

    Loads N copies of the model on CPU (each with its own threads) so
    multiple diagnoses / optimization hints can run in parallel.
    """

    def __init__(self, model_path: str, n_ctx: int = 4096,
                 timeout: float = 120.0, n_workers: int = 2):
        """
        Args:
            model_path: Path to GGUF model file
            n_ctx: Context window size
            timeout: Max seconds per LLM call before giving up
            n_workers: Number of parallel model instances (2-4 recommended)
        """
        self.timeout = timeout
        self._workers: list[_LLMWorker] = []
        self._pool: ThreadPoolExecutor | None = None
        self._robin = 0  # round-robin counter
        self._robin_lock = threading.Lock()

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

        # Split CPU threads across workers
        total_threads = max(4, os.cpu_count() or 4)
        threads_per_worker = max(2, total_threads // n_workers)

        t0 = time.time()
        print(f"[LLM Feedback] Loading {n_workers}x {os.path.basename(model_path)} "
              f"on CPU ({threads_per_worker} threads each)...", flush=True)
        for wid in range(n_workers):
            try:
                w = _LLMWorker(model_path, n_ctx, threads_per_worker, wid)
                self._workers.append(w)
            except Exception as e:
                print(f"[LLM Feedback] Worker {wid} failed to load: {e}")
                break
        if self._workers:
            self._pool = ThreadPoolExecutor(max_workers=len(self._workers))
            print(f"[LLM Feedback] {len(self._workers)} workers loaded in "
                  f"{time.time()-t0:.1f}s", flush=True)
        else:
            print("[LLM Feedback] No workers loaded — LLM feedback disabled")

    @property
    def available(self) -> bool:
        return len(self._workers) > 0

    def _pick_worker(self) -> _LLMWorker:
        """Round-robin worker selection."""
        with self._robin_lock:
            w = self._workers[self._robin % len(self._workers)]
            self._robin += 1
        return w

    def _call(self, system: str, user: str, max_tokens: int = 250,
              temperature: float = 0.3) -> str:
        """Make a single LLM call with timeout. Returns empty string on failure."""
        if not self.available:
            return ""

        worker = self._pick_worker()

        future = self._pool.submit(
            worker.generate, system, user, max_tokens, temperature
        )
        try:
            return future.result(timeout=self.timeout) or ""
        except FuturesTimeoutError:
            print(f"[LLM Feedback] Timeout after {self.timeout}s (worker {worker.worker_id})")
            return ""
        except Exception as e:
            print(f"[LLM Feedback] Error (worker {worker.worker_id}): {e}")
            return ""

    def _call_parallel(self, calls: list[dict]) -> list[str]:
        """Run multiple LLM calls in parallel across the worker pool.

        Args:
            calls: list of dicts with keys: system, user, max_tokens, temperature

        Returns:
            list of result strings (empty string for failures)
        """
        if not self.available or not calls:
            return [""] * len(calls)

        futures = []
        for c in calls:
            worker = self._pick_worker()
            f = self._pool.submit(
                worker.generate,
                c["system"], c["user"],
                c.get("max_tokens", 250),
                c.get("temperature", 0.3),
            )
            futures.append(f)

        results = []
        for f in futures:
            try:
                results.append(f.result(timeout=self.timeout) or "")
            except FuturesTimeoutError:
                print(f"[LLM Feedback] Timeout in parallel call")
                results.append("")
            except Exception as e:
                print(f"[LLM Feedback] Error in parallel call: {e}")
                results.append("")
        return results

    def diagnose_error(self, task: str, code: str, error: str) -> str:
        """Diagnose why a CUDA kernel failed.

        Args:
            task: The PyTorch operation description (prompt text)
            code: The generated Python/CUDA code that failed
            error: Compiler error or correctness error message

        Returns:
            2-4 sentence diagnosis, or empty string on failure
        """
        task_trunc = task[:2000]
        code_trunc = code[:3000] if code else "(no code extracted)"
        error_trunc = error[:1000] if error else "Unknown error"

        user_msg = _DIAGNOSE_USER.format(
            task=task_trunc,
            code=code_trunc,
            error=error_trunc,
        )
        t0 = time.time()
        result = self._call(_DIAGNOSE_SYSTEM, user_msg, max_tokens=512)
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

    # ── Batch APIs: run multiple calls in parallel ────────────────────────

    def diagnose_batch(self, items: list[dict]) -> list[str]:
        """Diagnose multiple errors in parallel across worker pool.

        Args:
            items: list of dicts with keys: task, code, error

        Returns:
            list of diagnosis strings (empty string for failures)
        """
        if not self.available or not items:
            return [""] * len(items)

        calls = []
        for item in items:
            task_trunc = item.get("task", "")[:2000]
            code_trunc = (item.get("code") or "(no code extracted)")[:3000]
            error_trunc = (item.get("error") or "Unknown error")[:1000]
            user_msg = _DIAGNOSE_USER.format(
                task=task_trunc, code=code_trunc, error=error_trunc,
            )
            calls.append({
                "system": _DIAGNOSE_SYSTEM,
                "user": user_msg,
                "max_tokens": 512,
                "temperature": 0.3,
            })

        t0 = time.time()
        results = self._call_parallel(calls)
        elapsed = time.time() - t0
        n_ok = sum(1 for r in results if r)
        print(f"[LLM Feedback] Batch diagnosis: {n_ok}/{len(items)} in {elapsed:.1f}s "
              f"({len(self._workers)} workers)", flush=True)
        return results

    def optimize_batch(self, items: list[dict]) -> list[str]:
        """Generate multiple optimization hints in parallel.

        Args:
            items: list of dicts with keys:
                task, code, speedup, profiler_info (opt),
                technique_hint (opt), temperature (opt)

        Returns:
            list of hint strings (empty string for failures)
        """
        if not self.available or not items:
            return [""] * len(items)

        calls = []
        for item in items:
            task_trunc = item.get("task", "")[:2000]
            code_trunc = (item.get("code") or "")[:3000]
            speedup = item.get("speedup", 0.0)
            profiler_info = item.get("profiler_info", "")
            technique_hint = item.get("technique_hint", "")
            temperature = item.get("temperature", 0.3)

            profiler_section = (
                f"\n\nProfiler analysis:\n{profiler_info[:500]}"
                if profiler_info else ""
            )
            technique_section = ""
            if technique_hint:
                technique_section = (
                    f"\n\nFocus your suggestion on applying this technique: "
                    f"**{technique_hint}**. Explain how it applies to this specific kernel."
                )
            user_msg = _OPTIMIZE_USER.format(
                task=task_trunc, code=code_trunc,
                speedup=speedup, profiler_section=profiler_section,
            ) + technique_section
            calls.append({
                "system": _OPTIMIZE_SYSTEM,
                "user": user_msg,
                "max_tokens": 250,
                "temperature": temperature,
            })

        t0 = time.time()
        results = self._call_parallel(calls)
        elapsed = time.time() - t0
        n_ok = sum(1 for r in results if r)
        print(f"[LLM Feedback] Batch optimize: {n_ok}/{len(items)} in {elapsed:.1f}s "
              f"({len(self._workers)} workers)", flush=True)
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

    fb = LLMFeedback(model_path, n_workers=2)
    if not fb.available:
        print("Model failed to load!")
        sys.exit(1)

    # Test single diagnosis
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

    # Test batch diagnosis (parallel)
    print("\n=== Testing Batch Diagnosis (parallel) ===")
    batch_results = fb.diagnose_batch([
        {"task": "relu", "code": "kernel code 1", "error": "wrong output"},
        {"task": "relu", "code": "kernel code 2", "error": "compile error"},
    ])
    for i, r in enumerate(batch_results):
        print(f"  [{i}]: {r[:100]}...")

    # Test optimization
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
