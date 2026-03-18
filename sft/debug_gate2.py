"""
debug_gate2.py - Show exactly why SakanaAI kernels fail Gate 2.

Loads N kernels from each level, runs build_load_inline_wrapper + evaluate
with all debug output visible, prints a failure breakdown.

Usage:
    cd /root/KernelForge/sft
    /root/KernelForge/.venv/bin/python3 debug_gate2.py --per_level 10
"""
import argparse
import os
import sys
import re

_rl_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rl"))
if _rl_dir not in sys.path:
    sys.path.insert(0, _rl_dir)

from datasets import load_dataset
from agent import build_load_inline_wrapper
from sandbox import evaluate


FAIL_CATEGORIES = {
    "no_binding":    0,
    "compile_fail":  0,
    "wrong_output":  0,
    "exception":     0,
    "pass":          0,
}


def run_one(cuda_code: str, pytorch_code: str, label: str, verbose: bool):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    if verbose:
        print("--- CUDA (first 400 chars) ---")
        print(cuda_code[:400])
        print("---")

    # Step 1: wrapper
    wrapper = build_load_inline_wrapper(cuda_code, pytorch_code)
    if not wrapper:
        print("  FAIL: no binding found")
        FAIL_CATEGORIES["no_binding"] += 1
        return

    # Step 2: evaluate
    try:
        result = evaluate(wrapper, pytorch_code)
    except Exception as e:
        print(f"  FAIL: exception in evaluate: {e}")
        FAIL_CATEGORIES["exception"] += 1
        return

    if result is None:
        print("  FAIL: compile failed (evaluate returned None)")
        FAIL_CATEGORIES["compile_fail"] += 1
        return

    if result.get("correct", False):
        speedup = result.get("speedup", 1.0)
        print(f"  PASS  speedup={speedup:.2f}x")
        FAIL_CATEGORIES["pass"] += 1
    else:
        err = result.get("compiler_error") or "wrong output"
        # Classify
        if "error:" in err.lower() or "nvcc" in err.lower() or "undefined" in err.lower():
            FAIL_CATEGORIES["compile_fail"] += 1
            cat = "compile_fail"
        else:
            FAIL_CATEGORIES["wrong_output"] += 1
            cat = "wrong_output"
        print(f"  FAIL [{cat}]: {err[:300]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_level", type=int, default=10,
                        help="Kernels to test per level (default 10)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print CUDA code snippet for each kernel")
    parser.add_argument("--level", default="",
                        help="Only test this level (level_1, level_2, level_3)")
    args = parser.parse_args()

    levels = [args.level] if args.level else ["level_1", "level_2", "level_3"]
    total = 0

    for level in levels:
        print(f"\n{'#'*60}")
        print(f"  {level}")
        print(f"{'#'*60}")
        try:
            ds = load_dataset("SakanaAI/AI-CUDA-Engineer-Archive", split=level)
        except Exception as e:
            print(f"Could not load {level}: {e}")
            continue

        count = 0
        for row in ds:
            if count >= args.per_level:
                break
            if not row.get("Correct", False):
                continue

            pytorch_code = (
                row.get("PyTorch_Code_Module", "")
                or row.get("PyTorch_Code_Functional", "")
            ).strip()
            cuda_code = row.get("CUDA_Code", "").strip()

            if not pytorch_code or not cuda_code:
                continue
            if "__global__" not in cuda_code or "torch::Tensor" not in cuda_code:
                continue

            label = f"{level} task_id={row.get('task_id', '?')}"
            run_one(cuda_code, pytorch_code, label, args.verbose)
            count += 1
            total += 1

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for cat, n in FAIL_CATEGORIES.items():
        pct = n / max(1, total) * 100
        print(f"  {cat:<16} {n:3d}  ({pct:.0f}%)")
    print(f"  {'total':<16} {total}")


if __name__ == "__main__":
    main()
