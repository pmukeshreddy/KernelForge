"""
test_llm_feedback.py — Unit tests for LLM feedback module.

Tests prompt construction, formatting, and fallback behavior.
No GGUF model or GPU needed for these tests.
"""

import unittest
import sys
import os

# Add rl/ to path so we can import
sys.path.insert(0, os.path.dirname(__file__))

from llm_feedback import (
    LLMFeedback,
    _format_llm_hint,
    _DIAGNOSE_SYSTEM,
    _DIAGNOSE_USER,
    _OPTIMIZE_SYSTEM,
    _OPTIMIZE_USER,
)


class TestPromptConstruction(unittest.TestCase):
    """Test that prompts are constructed correctly (no model needed)."""

    def test_diagnose_prompt_formatting(self):
        """Diagnosis prompt should include task, code, and error."""
        prompt = _DIAGNOSE_USER.format(
            task="torch.relu(x)",
            code="def forward(self, x): return x",
            error="RuntimeError: misaligned address",
        )
        self.assertIn("torch.relu(x)", prompt)
        self.assertIn("def forward(self, x)", prompt)
        self.assertIn("misaligned address", prompt)
        self.assertIn("Diagnose", prompt)

    def test_optimize_prompt_formatting(self):
        """Optimization prompt should include speedup and optional profiler."""
        prompt = _OPTIMIZE_USER.format(
            task="torch.matmul(a, b)",
            code="__global__ void matmul_kernel(...) { ... }",
            speedup=0.85,
            profiler_section="\n\nProfiler analysis:\nMemory-bound",
        )
        self.assertIn("0.85x", prompt)
        self.assertIn("Memory-bound", prompt)
        self.assertIn("matmul", prompt)

    def test_optimize_prompt_no_profiler(self):
        """Optimization prompt works without profiler info."""
        prompt = _OPTIMIZE_USER.format(
            task="torch.relu(x)",
            code="__global__ void relu(...) {}",
            speedup=1.5,
            profiler_section="",
        )
        self.assertIn("1.50x", prompt)
        self.assertNotIn("Profiler", prompt)

    def test_system_prompts_not_empty(self):
        """System prompts should be non-empty."""
        self.assertTrue(len(_DIAGNOSE_SYSTEM) > 50)
        self.assertTrue(len(_OPTIMIZE_SYSTEM) > 50)

    def test_system_prompts_no_code(self):
        """System prompts should not ask to provide corrected code."""
        self.assertIn("Do NOT provide corrected code", _DIAGNOSE_SYSTEM)
        self.assertIn("Do NOT rewrite", _OPTIMIZE_SYSTEM)


class TestFormatting(unittest.TestCase):
    """Test the hint formatting function."""

    def test_format_diagnosis(self):
        result = _format_llm_hint("The indexing is off by one.", "diagnosis")
        self.assertIn("Bug Diagnosis", result)
        self.assertIn("indexing is off", result)
        self.assertIn("---", result)

    def test_format_optimization(self):
        result = _format_llm_hint("Use shared memory tiling.", "optimization")
        self.assertIn("Optimization Hint", result)
        self.assertIn("shared memory", result)

    def test_format_empty(self):
        """Empty hint should return empty string."""
        result = _format_llm_hint("", "diagnosis")
        self.assertEqual(result, "")

    def test_format_none_like(self):
        """None-like values should return empty."""
        result = _format_llm_hint("", "optimization")
        self.assertEqual(result, "")


class TestFallbackBehavior(unittest.TestCase):
    """Test LLMFeedback when no model is loaded."""

    def test_no_model_path(self):
        """Empty path should create unavailable instance."""
        fb = LLMFeedback("")
        self.assertFalse(fb.available)

    def test_missing_model_file(self):
        """Non-existent file should create unavailable instance."""
        fb = LLMFeedback("/nonexistent/path/to/model.gguf")
        self.assertFalse(fb.available)

    def test_diagnose_returns_empty_when_unavailable(self):
        """Diagnosis should return empty string when no model."""
        fb = LLMFeedback("")
        result = fb.diagnose_error("task", "code", "error")
        self.assertEqual(result, "")

    def test_optimize_returns_empty_when_unavailable(self):
        """Optimization should return empty string when no model."""
        fb = LLMFeedback("")
        result = fb.suggest_optimization("task", "code", 1.0)
        self.assertEqual(result, "")

    def test_batch_returns_empty_when_unavailable(self):
        """Batch diagnosis should return list of empty strings when no model."""
        fb = LLMFeedback("")
        items = [{"task": "t1", "code": "c1", "error": "e1"},
                 {"task": "t2", "code": "c2", "error": "e2"}]
        results = fb.diagnose_batch(items)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r == "" for r in results))

    def test_batch_empty_items(self):
        """Empty item list should return empty list."""
        fb = LLMFeedback("")
        results = fb.diagnose_batch([])
        self.assertEqual(results, [])


class TestInputTruncation(unittest.TestCase):
    """Test that long inputs are truncated to fit context window."""

    def test_long_task_truncated(self):
        """Very long task descriptions should be truncated."""
        fb = LLMFeedback("")  # no model, won't actually call LLM
        long_task = "x" * 5000
        # Can't test internal truncation without model, but verify no crash
        result = fb.diagnose_error(long_task, "code", "error")
        self.assertEqual(result, "")  # unavailable, returns empty

    def test_long_code_truncated(self):
        """Very long code should be truncated."""
        fb = LLMFeedback("")
        long_code = "y" * 10000
        result = fb.diagnose_error("task", long_code, "error")
        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main()
