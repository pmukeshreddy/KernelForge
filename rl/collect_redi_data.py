"""
collect_redi_data.py - REDI Stage 2: Collect verified distillation traces.

Reads existing SFT data (pytorch_code + cuda_kernel pairs from SakanaAI),
wraps the C++ into load_inline Python, evaluates in sandbox, and labels
as positive or negative.

Output: redi_traces.jsonl with fields:
  - pytorch_code: reference PyTorch implementation
  - cuda_code: the full Python ModelNew wrapper sent to sandbox
  - label: +1 (correct + speedup) or -1 (fail)
  - reward: float speedup if success, 0.0 if fail
  - error: string if failed, empty if success
  - level_id: which difficulty level (level_1, level_2, level_3)

Reuses: sandbox.evaluate(), reward.calculate_reward(),
        agent.build_load_inline_wrapper(), agent._fix_cuda_api()
"""

import json
import os
import sys
import argparse
from typing import Optional

import re

from agent import build_load_inline_wrapper, _fix_cuda_api
from reward import calculate_reward
from sandbox import evaluate
from concurrent.futures import ThreadPoolExecutor


def _sanitize(text: str) -> str:
    """Replace Unicode smart quotes and non-ASCII chars that break ASCII locales."""
    return (text
        .replace('\u2018', "'").replace('\u2019', "'")
        .replace('\u201c', '"').replace('\u201d', '"')
        .replace('\u2013', '-').replace('\u2014', '--')
        .encode('ascii', 'replace').decode('ascii'))


def _strip_pybind(cuda_code: str) -> str:
    """Remove PYBIND11_MODULE(...) blocks from CUDA code.
    
    SakanaAI kernels include their own PYBIND11_MODULE, but load_inline
    generates one automatically. Having two causes duplicate symbol linker errors.
    """
    # Remove PYBIND11_MODULE block with balanced braces
    result = []
    i = 0
    lines = cuda_code.split('\n')
    skip = False
    brace_depth = 0
    for line in lines:
        if not skip and 'PYBIND11_MODULE' in line:
            skip = True
            brace_depth = 0
        if skip:
            brace_depth += line.count('{') - line.count('}')
            if brace_depth <= 0 and '{' in cuda_code:
                skip = False
            continue
        result.append(line)
    return '\n'.join(result)


# ---------------------------------------------------------------------------
# Process a single entry
# ---------------------------------------------------------------------------

def _process_entry(
    idx: int,
    pytorch_code: str,
    cuda_kernel: str,
    level_id: str,
    total: int,
) -> dict:
    """Process a single pytorch_code + cuda_kernel pair through the sandbox."""
    print(f"[{idx+1}/{total}] ({level_id}) Wrapping and evaluating...", flush=True)
    pytorch_code = _sanitize(pytorch_code)
    cuda_kernel = _sanitize(cuda_kernel)

    # Strip PYBIND11_MODULE — load_inline generates its own
    cuda_kernel = _strip_pybind(cuda_kernel)

    trace = {
        "pytorch_code": pytorch_code,
        "cuda_code": "",
        "label": -1,
        "reward": 0.0,
        "error": "",
        "level_id": level_id,
    }

    # 1. Wrap raw C++ into full Python ModelNew file
    wrapper = build_load_inline_wrapper(cuda_kernel, pytorch_code)
    if not wrapper:
        trace["error"] = "build_load_inline_wrapper returned None (no binding function found)"
        print(f"[{idx+1}/{total}] ❌ No binding function found")
        return trace

    trace["cuda_code"] = wrapper

    # 2. Evaluate in sandbox
    eval_result = evaluate(wrapper, pytorch_code)

    if not eval_result["correct"]:
        error = eval_result.get("compiler_error", "Outputs do not match")

        # OOM is infrastructure failure — skip
        if "out of memory" in error.lower() or "cudaErrorMemoryAllocation" in error:
            trace["error"] = "OOM (infra)"
            print(f"[{idx+1}/{total}] ⚠️  OOM — skipping")
            return trace

        trace["error"] = error[:500]
        print(f"[{idx+1}/{total}] ❌ {error[:80]}")
        return trace

    # 3. Success — calculate reward
    reward = calculate_reward(eval_result)
    trace["label"] = 1
    trace["reward"] = reward

    runtime_ms = eval_result["runtime_ms"]
    baseline_ms = eval_result.get("baseline_runtime_ms", 0.0)
    print(f"[{idx+1}/{total}] ✅ {runtime_ms:.3f}ms vs {baseline_ms:.3f}ms baseline, reward={reward:.2f}")
    return trace


def collect_traces(
    dataset_path: str,
    output_path: str,
    level_filter: Optional[str] = None,
    max_prompts: Optional[int] = None,
    num_workers: int = 2,
):
    """
    Read existing SFT pairs, evaluate in sandbox, save labeled traces.
    """
    # Load entries with both pytorch_code and cuda_kernel
    entries = []
    with open(dataset_path) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                pytorch_code = entry.get("pytorch_code", "")
                cuda_kernel = entry.get("cuda_kernel", "")
                level_id = entry.get("level_id", "level_1")

                if not pytorch_code.strip() or not cuda_kernel.strip():
                    continue
                if level_filter and level_id != level_filter:
                    continue

                entries.append({
                    "pytorch_code": pytorch_code,
                    "cuda_kernel": cuda_kernel,
                    "level_id": level_id,
                })

    levels = level_filter or "all levels"
    print(f"Loaded {len(entries)} entries for REDI ({levels})")

    if max_prompts:
        entries = entries[:max_prompts]

    total = len(entries)
    print(f"Processing {total} entries ({num_workers} parallel workers)")

    stats = {"positive": 0, "negative": 0, "errors": 0}

    # Parallel processing
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = []
        for idx, entry in enumerate(entries):
            futures.append(
                pool.submit(
                    _process_entry,
                    idx,
                    entry["pytorch_code"],
                    entry["cuda_kernel"],
                    entry["level_id"],
                    total,
                )
            )

        # Write traces as they complete
        with open(output_path, "w") as f:
            for future in futures:
                trace = future.result()
                if trace["label"] == 1:
                    stats["positive"] += 1
                elif trace["error"]:
                    stats["errors"] += 1
                else:
                    stats["negative"] += 1
                f.write(json.dumps(trace) + "\n")

    # Final stats
    print(f"\n{'='*50}")
    print(f"REDI Collection Complete")
    print(f"  Positive traces: {stats['positive']}")
    print(f"  Negative traces: {stats['negative']}")
    print(f"  Errors (OOM/wrapper): {stats['errors']}")
    print(f"  Total:           {total}")
    print(f"  Saved to:        {output_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="REDI Trace Collection (offline)")
    parser.add_argument("--dataset", type=str, default="../sft/sft_training_pairs.jsonl",
                        help="Input JSONL with pytorch_code + cuda_kernel fields")
    parser.add_argument("--output", type=str, default="data/redi_traces.jsonl")
    parser.add_argument("--level", type=str, default=None,
                        choices=["level_1", "level_2", "level_3"],
                        help="Filter to a specific level (default: all)")
    parser.add_argument("--max_prompts", type=int, default=None,
                        help="Limit number of entries (for testing)")
    parser.add_argument("--workers", type=int, default=2,
                        help="Number of parallel workers")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    collect_traces(
        dataset_path=args.dataset,
        output_path=args.output,
        level_filter=args.level,
        max_prompts=args.max_prompts,
        num_workers=args.workers,
    )
