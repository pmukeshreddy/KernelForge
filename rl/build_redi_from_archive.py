"""
build_redi_from_archive.py - Build REDI training data from SakanaAI/AI-CUDA-Engineer-Archive.

Instead of making expensive API calls, this script downloads the pre-existing
archive of Claude-generated CUDA kernels, filters for correct+fast ones, wraps
them in the ModelNew/load_inline format, re-verifies in the sandbox, and writes
redi_traces.jsonl ready for train_redi.py.

Pipeline:
  1. Download SakanaAI/AI-CUDA-Engineer-Archive from HuggingFace
  2. Filter: Correct == True and CUDA_Speedup_Native >= 1.0
  3. Deduplicate: keep best (highest speedup) kernel per Task_ID
  4. Wrap CUDA_Code through build_load_inline_wrapper
  5. Re-verify correctness + timing in sandbox (don't trust archived speedups)
  6. Write verified traces to output JSONL

Output format matches redi_traces.jsonl used by train_redi.py:
  - pytorch_code, cuda_code, label, reward, error, step
"""

import json
import os
import re
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore

from agent import build_load_inline_wrapper, _fix_cuda_api
from reward import calculate_reward
from sandbox import evaluate


# Only use level_1 and level_2 for REDI (level_3 goes to GRPO)
REDI_LEVELS = {"level_1", "level_2"}

# Semaphore to limit concurrent sandbox/GPU processes
_gpu_sem = None


def _verify_one(row: dict, idx: int, total: int) -> dict | None:
    """
    Wrap and sandbox-verify one archive row.
    Returns a trace dict on success, None if it should be skipped.
    """
    pytorch_code = row["pytorch_code"]
    cuda_code_raw = row["cuda_code_raw"]   # raw CUDA C++ with PYBIND11_MODULE
    task_id = row.get("task_id", "?")

    # Strip PYBIND11_MODULE block — load_inline generates its own bindings
    cuda_code_raw = re.sub(
        r'PYBIND11_MODULE\s*\(.*?\)\s*\{[^}]*\}', '', cuda_code_raw, flags=re.DOTALL
    ).strip()

    # Fix common CUDA API mismatches
    cuda_code_raw = _fix_cuda_api(cuda_code_raw)

    # Wrap raw CUDA C++ into full ModelNew/load_inline Python
    cuda_code = build_load_inline_wrapper(cuda_code_raw, pytorch_code)
    if not cuda_code:
        print(f"[{idx}/{total}] {task_id}: ⚠️  no torch::Tensor binding found — skipping", flush=True)
        return None

    # Re-verify in sandbox
    print(f"[{idx}/{total}] {task_id}: compiling...", flush=True)
    with _gpu_sem:
        eval_result = evaluate(cuda_code, pytorch_code)

    if not eval_result["correct"]:
        error = eval_result.get("compiler_error", "incorrect output")
        if "out of memory" in error.lower():
            print(f"[{idx}/{total}] {task_id}: ⚠️  OOM — skipping", flush=True)
        else:
            print(f"[{idx}/{total}] {task_id}: ❌ {error[:80]}", flush=True)
        return None

    reward = calculate_reward(eval_result)
    runtime_ms = eval_result["runtime_ms"]
    baseline_ms = eval_result.get("baseline_runtime_ms", 0.0)
    print(f"[{idx}/{total}] {task_id}: ✅ {runtime_ms:.3f}ms vs {baseline_ms:.3f}ms baseline — {reward:.2f}x", flush=True)

    return {
        "pytorch_code": pytorch_code,
        "cuda_code": cuda_code,
        "label": 1,
        "reward": reward,
        "error": "",
        "step": 0,
    }


def build_traces(
    output_path: str,
    min_speedup: float = 1.0,
    max_rows: int | None = None,
    workers: int = 2,
):
    global _gpu_sem
    _gpu_sem = Semaphore(workers)

    print("Loading SakanaAI/AI-CUDA-Engineer-Archive from HuggingFace...", flush=True)
    from datasets import load_dataset, concatenate_datasets
    splits = [load_dataset("SakanaAI/AI-CUDA-Engineer-Archive", split=lvl) for lvl in REDI_LEVELS]
    ds = concatenate_datasets(splits)
    print(f"Loaded {len(ds)} total rows (level_1 + level_2)", flush=True)

    # Step 1: Filter correct + fast
    filtered = [
        row for row in ds
        if row.get("Correct") is True
        and (row.get("CUDA_Speedup_Native") or 0.0) >= min_speedup
    ]
    print(f"After filter (Correct=True, speedup>={min_speedup}, level 1+2): {len(filtered)} rows", flush=True)

    # Step 2: Deduplicate — best speedup per Task_ID
    best: dict[str, dict] = {}
    for row in filtered:
        tid = row.get("Task_ID", "")
        spd = row.get("CUDA_Speedup_Native") or 0.0
        if tid not in best or spd > (best[tid].get("CUDA_Speedup_Native") or 0.0):
            best[tid] = row
    rows = list(best.values())
    print(f"After dedup (best per task): {len(rows)} rows", flush=True)

    if max_rows:
        rows = rows[:max_rows]
        print(f"Capped to {len(rows)} rows", flush=True)

    # Inspect: what does CUDA_Code actually look like across rows?
    python_count = sum(1 for r in rows if r.get("CUDA_Code", "").lstrip().startswith("import"))
    cpp_count = len(rows) - python_count
    print(f"CUDA_Code format: {python_count} full Python files, {cpp_count} raw C++ files", flush=True)
    if python_count > 0:
        # Show a sample Python one
        for r in rows:
            if r.get("CUDA_Code", "").lstrip().startswith("import"):
                print("\n--- Sample Python CUDA_Code (first 20 lines) ---")
                for i, line in enumerate(r["CUDA_Code"].split("\n")[:20], 1):
                    print(f"{i:3}: {line}")
                print("---\n", flush=True)
                break

    # Normalize to internal format
    work = [
        {
            "pytorch_code": r.get("PyTorch_Code_Module", ""),
            "cuda_code_raw": r.get("CUDA_Code", ""),
            "task_id": r.get("Task_ID", ""),
        }
        for r in rows
        if r.get("PyTorch_Code_Module", "").strip() and r.get("CUDA_Code", "").strip()
    ]
    total = len(work)
    print(f"\nVerifying {total} candidates in sandbox ({workers} parallel GPU workers)...\n", flush=True)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    passed = 0
    failed = 0

    with open(output_path, "w") as out_f:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_verify_one, row, idx + 1, total): idx
                for idx, row in enumerate(work)
            }
            for future in as_completed(futures):
                trace = future.result()
                if trace is not None:
                    out_f.write(json.dumps(trace) + "\n")
                    out_f.flush()
                    passed += 1
                else:
                    failed += 1

    print(f"\n{'='*50}")
    print(f"Archive REDI build complete")
    print(f"  Passed (verified correct+fast): {passed}")
    print(f"  Failed/skipped:                 {failed}")
    print(f"  Saved to: {output_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build REDI traces from SakanaAI archive")
    parser.add_argument("--output", type=str, default="data/redi_traces.jsonl")
    parser.add_argument("--min_speedup", type=float, default=1.0,
                        help="Minimum CUDA_Speedup_Native to include before sandbox re-check")
    parser.add_argument("--max_rows", type=int, default=None,
                        help="Cap rows for testing (e.g. --max_rows 20)")
    parser.add_argument("--workers", type=int, default=2,
                        help="Concurrent sandbox GPU workers")
    args = parser.parse_args()

    build_traces(
        output_path=args.output,
        min_speedup=args.min_speedup,
        max_rows=args.max_rows,
        workers=args.workers,
    )
