"""
debug_gate2.py - Show exactly why SakanaAI kernels fail Gate 2.

Runs in parallel (same pool.map pattern as GRPO), prints per-failure
category breakdown + sample error messages for each category.

Usage:
    cd /root/KernelForge/sft
    /root/KernelForge/.venv/bin/python3 debug_gate2.py --per_level 30
    /root/KernelForge/.venv/bin/python3 debug_gate2.py --per_level 30 --level level_1
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

_rl_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rl"))
if _rl_dir not in sys.path:
    sys.path.insert(0, _rl_dir)

from datasets import load_dataset


def _debug_worker(item):
    """
    Returns (label, category, detail) where category is one of:
      pass | no_binding | compile_fail | wrong_output | exception
    """
    import os, sys, io
    if _rl_dir not in sys.path:
        sys.path.insert(0, _rl_dir)

    cuda_code, pytorch_code, label = item

    # Capture [WRAPPER DEBUG] prints instead of suppressing them
    buf = io.StringIO()
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = buf

    try:
        from agent import build_load_inline_wrapper
        from sandbox import evaluate

        wrapper = build_load_inline_wrapper(cuda_code, pytorch_code)
        sys.stdout, sys.stderr = old_stdout, old_stderr
        debug_out = buf.getvalue()

        if not wrapper:
            return label, "no_binding", debug_out.strip()[-400:]

        result = evaluate(wrapper, pytorch_code)

        if result is None:
            return label, "compile_fail", "evaluate returned None\n---WRAPPER DEBUG---\n" + debug_out.strip()[-400:]

        if result.get("correct", False):
            speedup = result.get("speedup", 1.0)
            return label, "pass", f"speedup={speedup:.2f}x"

        err = result.get("compiler_error") or "wrong output"
        detail = err[:300] + "\n---WRAPPER DEBUG---\n" + debug_out.strip()[-400:]
        if any(k in err.lower() for k in ("error:", "nvcc", "undefined", "fatal")):
            return label, "compile_fail", detail
        return label, "wrong_output", detail

    except Exception as e:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        return label, "exception", str(e)[:300]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_level", type=int, default=10)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--level", default="",
                        help="Only test this level (level_1, level_2, level_3)")
    args = parser.parse_args()

    levels = [args.level] if args.level else ["level_1", "level_2", "level_3"]

    # Collect items
    items = []
    for level in levels:
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
            label = f"{level}/task_{row.get('task_id','?')}"
            items.append((cuda_code, pytorch_code, label))
            count += 1

    print(f"Running {len(items)} kernels on {args.workers} workers...")

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(tqdm(
            pool.map(_debug_worker, items, chunksize=1),
            total=len(items), unit="kernel",
        ))

    # Aggregate
    cats = {"pass": [], "no_binding": [], "compile_fail": [], "wrong_output": [], "exception": []}
    for label, cat, detail in results:
        cats[cat].append((label, detail))

    total = len(results)
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for cat, entries in cats.items():
        pct = len(entries) / max(1, total) * 100
        print(f"  {cat:<16} {len(entries):3d}  ({pct:.0f}%)")
    print(f"  {'total':<16} {total}")

    # Print sample failures for each non-pass category
    for cat in ("no_binding", "compile_fail", "wrong_output", "exception"):
        entries = cats[cat]
        if not entries:
            continue
        print(f"\n{'─'*60}")
        print(f"  {cat.upper()} — sample errors (up to 5)")
        print(f"{'─'*60}")
        for label, detail in entries[:5]:
            print(f"\n  [{label}]")
            print(f"  {detail[:400]}")


if __name__ == "__main__":
    main()
