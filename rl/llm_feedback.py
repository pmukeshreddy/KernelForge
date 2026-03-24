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
CUDA kernel debugger. Reply with ONLY 1-2 sentences. No preamble. No code blocks. No bullet lists.

WRONG: "Let me analyze the code. Looking at line 54..."
WRONG: "```cpp\nfloat x = ...```"
RIGHT: "The variable `g_idx` overflows shared memory bounds because group*1024+tid exceeds 1024 when group>0."
RIGHT: "`gelu_kernel` is called but never defined — only a `__device__ gelu` function exists."

Name the buggy variable or expression and say why it's wrong. Nothing else."""

_DIAGNOSE_USER = """\
CUDA kernel for: {task}

{code}

Error: {error}"""

_OPTIMIZE_SYSTEM = """\
CUDA performance expert. Reply with ONLY 1-2 sentences. No preamble. No code blocks. No bullet lists.

WRONG: "Let me analyze the memory access pattern..."
RIGHT: "The inner loop in `conv_kernel` reads `weight[oc*IC + ic]` with stride IC, causing uncoalesced global memory access — tiling into shared memory would fix this."

Name the bottleneck variable/loop and say what optimization to apply. Nothing else."""

_OPTIMIZE_USER = """\
CUDA kernel ({speedup:.2f}x vs PyTorch) for: {task}

{code}
{profiler_section}
What is the #1 bottleneck?"""


import re as _re

# Preamble patterns the model loves to generate despite instructions
_PREAMBLE_PATTERNS = [
    _re.compile(r'^(?:The user is asking me to|Let me analyze|Looking at|I need to|Here\'s my analysis)[^.]*\.\s*', _re.IGNORECASE),
    _re.compile(r'^(?:Looking at the (?:CUDA |code|kernel|error)[^.]*\.\s*)+', _re.IGNORECASE),
    _re.compile(r'^(?:Let me (?:look|check|examine|analyze|find)[^.]*\.\s*)+', _re.IGNORECASE),
]

def _strip_preamble(text: str) -> str:
    """Remove LLM filler preamble, code blocks, and bullet lists."""
    # Remove code blocks
    text = _re.sub(r'```\w*\n?.*?```', '', text, flags=_re.DOTALL)
    # Remove numbered lists (1. 2. 3.)
    text = _re.sub(r'^\s*\d+\.\s+\*?\*?', '', text, flags=_re.MULTILINE)
    # Remove bullet points
    text = _re.sub(r'^\s*[-*]\s+', '', text, flags=_re.MULTILINE)
    # Remove bold markers
    text = _re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    # Strip known preamble patterns
    for pat in _PREAMBLE_PATTERNS:
        text = pat.sub('', text)
    # Collapse whitespace
    text = ' '.join(text.split())
    return text.strip()


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
              prefill: str = "") -> str:
        """Make a single LLM call with timeout. Returns empty string on failure.

        Args:
            prefill: Optional assistant prefill to force response format.
                     The model continues from this text.
        """
        if not self.available:
            return ""

        result = [None]
        error = [None]

        def _generate():
            try:
                with self._lock:
                    messages = [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ]
                    if prefill:
                        messages.append({"role": "assistant", "content": prefill})
                    resp = self._llm.create_chat_completion(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=0.3,  # low temp for consistent diagnosis
                        top_p=0.9,
                    )
                    raw = resp["choices"][0]["message"]["content"].strip()
                    # Prepend prefill since model continues from it
                    if prefill and not raw.startswith(prefill):
                        raw = prefill + raw
                    result[0] = _strip_preamble(raw)
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
        print(f"[LLM Feedback DEBUG] === DIAGNOSE INPUT ===", flush=True)
        print(f"[LLM Feedback DEBUG] System prompt ({len(_DIAGNOSE_SYSTEM)} chars):\n{_DIAGNOSE_SYSTEM}", flush=True)
        print(f"[LLM Feedback DEBUG] User msg ({len(user_msg)} chars):\n{user_msg[:1500]}{'...(truncated)' if len(user_msg) > 1500 else ''}", flush=True)
        print(f"[LLM Feedback DEBUG] === END INPUT ===", flush=True)
        t0 = time.time()
        result = self._call(_DIAGNOSE_SYSTEM, user_msg, max_tokens=100,
                             prefill="The bug is in `")
        elapsed = time.time() - t0
        if result:
            print(f"[LLM Feedback DEBUG] === DIAGNOSE OUTPUT ({elapsed:.1f}s) ===", flush=True)
            print(f"[LLM Feedback DEBUG] Full response ({len(result)} chars):\n{result}", flush=True)
            print(f"[LLM Feedback DEBUG] === END OUTPUT ===", flush=True)
        else:
            print(f"[LLM Feedback DEBUG] No response after {elapsed:.1f}s", flush=True)
        return result

    def suggest_optimization(self, task: str, code: str,
                              speedup: float, profiler_info: str = "") -> str:
        """Suggest optimization for a correct-but-slow kernel.

        Args:
            task: The PyTorch operation description
            code: The correct CUDA kernel code
            speedup: Current speedup vs PyTorch (e.g., 0.8 = 80% of PyTorch speed)
            profiler_info: Optional profiler output (bottleneck analysis)

        Returns:
            3-5 sentence optimization suggestion, or empty string on failure
        """
        code_trunc = code[:3000] if code else ""
        task_trunc = task[:2000]
        profiler_section = (
            f"\n\nProfiler analysis:\n{profiler_info[:500]}"
            if profiler_info else ""
        )

        user_msg = _OPTIMIZE_USER.format(
            task=task_trunc,
            code=code_trunc,
            speedup=speedup,
            profiler_section=profiler_section,
        )
        print(f"[LLM Feedback DEBUG] === OPTIMIZE INPUT ===", flush=True)
        print(f"[LLM Feedback DEBUG] System prompt ({len(_OPTIMIZE_SYSTEM)} chars):\n{_OPTIMIZE_SYSTEM}", flush=True)
        print(f"[LLM Feedback DEBUG] User msg ({len(user_msg)} chars):\n{user_msg[:1500]}{'...(truncated)' if len(user_msg) > 1500 else ''}", flush=True)
        print(f"[LLM Feedback DEBUG] === END INPUT ===", flush=True)
        t0 = time.time()
        result = self._call(_OPTIMIZE_SYSTEM, user_msg, max_tokens=100,
                             prefill="The bottleneck is ")
        elapsed = time.time() - t0
        if result:
            print(f"[LLM Feedback DEBUG] === OPTIMIZE OUTPUT ({elapsed:.1f}s) ===", flush=True)
            print(f"[LLM Feedback DEBUG] Full response ({len(result)} chars):\n{result}", flush=True)
            print(f"[LLM Feedback DEBUG] === END OUTPUT ===", flush=True)
        else:
            print(f"[LLM Feedback DEBUG] No response after {elapsed:.1f}s", flush=True)
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
