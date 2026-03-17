"""Debug script: show what wrapper code is generated for the first entry and full error."""
import json
import sys
import re
from agent import build_load_inline_wrapper, _fix_cuda_api
from sandbox import evaluate
from collect_redi_data import _strip_pybind

dataset = sys.argv[1] if len(sys.argv) > 1 else "../sft/sft_training_pairs.jsonl"

# Read first level_2 entry
with open(dataset) as f:
    for line in f:
        entry = json.loads(line)
        if entry.get("level_id") == "level_2" and entry.get("cuda_kernel", "").strip():
            break

pytorch_code = entry["pytorch_code"]
cuda_kernel = entry["cuda_kernel"]

print("=" * 60)
print("PYTORCH CODE + INIT INPUTS:")
print("=" * 60)
print(pytorch_code)


print("\n" + "=" * 60)
print("CUDA KERNEL (first 800 chars):")
print("=" * 60)
print(cuda_kernel[:800])

print("\n" + "=" * 60)
print("STRIPPING PYBIND11_MODULE...")
print("=" * 60)
cuda_kernel = _strip_pybind(cuda_kernel)
print(f"After strip (last 200 chars): ...{cuda_kernel[-200:]}")

print("\n" + "=" * 60)
print("BUILDING WRAPPER...")
print("=" * 60)
wrapper = build_load_inline_wrapper(cuda_kernel, pytorch_code)

if wrapper is None:
    print("ERROR: build_load_inline_wrapper returned None!")
    sys.exit(1)

print("WRAPPER CODE:")
print("=" * 60)
print(wrapper)

print("\n" + "=" * 60)
print("EVALUATING IN SANDBOX...")
print("=" * 60)
result = evaluate(wrapper, pytorch_code)
print(json.dumps(result, indent=2))
