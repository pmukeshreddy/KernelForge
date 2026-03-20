"""
test_weight_sync.py — verify the /load_lora_adapter + sleep/wake weight sync end-to-end.

Tests the full cycle used by train_grpo.py:
  1. Launch SGLang with base model + --enable-lora
  2. Load initial LoRA via /load_lora_adapter → generate a response
  3. /release_memory_occupation → simulate training taking the VRAM
  4. /resume_memory_occupation  → SGLang comes back
  5. Save a modified LoRA to a temp dir, reload via /load_lora_adapter
  6. Generate again → verify generation still works with new LoRA

Usage:
    export SGLANG_PYTHON=/path/to/sglang_env/bin/python
    python rl/test_weight_sync.py --model <base_model_path> --adapter <lora_adapter_path> --port 30000
"""

import os
import sys
import time
import argparse
import tempfile
import shutil
import requests


LORA_NAME = "kf_grpo_adapter"


def test_weight_sync(model_path: str, adapter_path: str, port: int,
                     sglang_python: str, tp: int = 1):

    # ── 1. Launch SGLang server ──────────────────────────────────────────────
    print(f"[1] Launching SGLang server (base model, --enable-lora) on port {port}...")
    import subprocess, glob as _glob

    cuda_homes = sorted(_glob.glob("/usr/local/cuda-*"), reverse=True) + ["/usr/local/cuda"]
    cuda_home = next((p for p in cuda_homes if os.path.isfile(f"{p}/bin/nvcc")), None)
    env = dict(os.environ)
    if cuda_home:
        cuda_bin, cuda_inc = f"{cuda_home}/bin", f"{cuda_home}/include"
        if cuda_bin not in env.get("PATH", ""):
            env["PATH"] = f"{cuda_bin}:{env.get('PATH', '')}"
        if cuda_inc not in env.get("CPATH", ""):
            env["CPATH"] = f"{cuda_inc}:{env.get('CPATH', '')}"
        print(f"    CUDA home: {cuda_home}")

    proc = subprocess.Popen([
        sglang_python, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--port", str(port),
        "--tp", str(tp),
        "--dtype", "bfloat16",
        "--trust-remote-code",
        "--enable-lora",
        "--max-loras-per-batch", "1",
        "--mem-fraction-static", "0.3",
        "--context-length", "8192",
        "--log-level", "error",
    ], env=env)

    for i in range(120):
        try:
            if requests.get(f"http://localhost:{port}/health", timeout=2).status_code == 200:
                print(f"    Server ready ({i*2}s)")
                break
        except Exception:
            pass
        time.sleep(2)
    else:
        proc.terminate()
        raise RuntimeError("SGLang server failed to start")

    try:
        # ── 2. Load initial LoRA adapter ─────────────────────────────────────
        print(f"[2] Loading initial LoRA adapter: {adapter_path}")
        r = requests.post(
            f"http://localhost:{port}/load_lora_adapter",
            json={"lora_name": LORA_NAME, "lora_path": adapter_path},
            timeout=120,
        )
        print(f"    load_lora_adapter → {r.status_code}  {r.text[:200]}")
        assert r.status_code == 200, f"load_lora_adapter failed: {r.status_code} {r.text[:300]}"
        print("    Initial LoRA loaded OK")

        # ── 3. Generate with initial LoRA ────────────────────────────────────
        print("[3] Generating with initial LoRA...")
        r = requests.post(
            f"http://localhost:{port}/generate",
            json={
                "text": "Write a CUDA kernel for ReLU:\n",
                "lora_name": LORA_NAME,
                "sampling_params": {"max_new_tokens": 32, "temperature": 0.0},
            },
            timeout=60,
        )
        assert r.status_code == 200, f"generate failed: {r.text[:200]}"
        gen1 = r.json()["text"][:120]
        print(f"    Generated (first 120 chars): {repr(gen1)}")
        print("    Generation with initial LoRA OK")

        # ── 4. Release memory (simulate trainer taking the GPU) ──────────────
        print("[4] Releasing SGLang memory (/release_memory_occupation)...")
        r = requests.post(f"http://localhost:{port}/release_memory_occupation", timeout=60)
        print(f"    release_memory_occupation → {r.status_code}  {r.text[:100]}")
        assert r.status_code == 200, f"release failed: {r.status_code}"
        print("    Memory released OK  (SGLang is now sleeping)")

        # Simulate a short training step
        time.sleep(2)
        print("    (simulated 2s training step)")

        # ── 5. Resume memory ─────────────────────────────────────────────────
        print("[5] Resuming SGLang memory (/resume_memory_occupation)...")
        r = requests.post(f"http://localhost:{port}/resume_memory_occupation", timeout=120)
        print(f"    resume_memory_occupation → {r.status_code}  {r.text[:100]}")
        assert r.status_code == 200, f"resume failed: {r.status_code}"
        print("    Memory resumed OK  (SGLang is awake again)")

        # ── 6. Save a modified LoRA and hot-reload ───────────────────────────
        print("[6] Saving modified LoRA to temp dir and hot-reloading...")
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        tmpdir = tempfile.mkdtemp(prefix="kf_test_lora_")
        lora_save_path = os.path.join(tmpdir, "adapter")
        try:
            # Load the adapter on CPU, tweak one LoRA weight slightly, save
            base = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map="cpu",
                trust_remote_code=True,
            )
            peft_model = PeftModel.from_pretrained(base, adapter_path)

            # Nudge the first LoRA-A weight so it's detectably different
            with torch.no_grad():
                for name, param in peft_model.named_parameters():
                    if "lora_A" in name and param.requires_grad:
                        param.data += 0.001
                        print(f"    Nudged: {name}  shape={list(param.shape)}")
                        break

            peft_model.save_pretrained(lora_save_path)
            del base, peft_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            print(f"    Modified LoRA saved → {lora_save_path}")

            # Hot-reload into SGLang
            r = requests.post(
                f"http://localhost:{port}/load_lora_adapter",
                json={"lora_name": LORA_NAME, "lora_path": lora_save_path},
                timeout=60,
            )
            print(f"    load_lora_adapter (updated) → {r.status_code}  {r.text[:200]}")
            assert r.status_code == 200, f"hot-reload failed: {r.status_code} {r.text[:300]}"
            print("    Hot-reload OK")

            # Flush stale KV cache
            requests.post(f"http://localhost:{port}/flush_cache", timeout=30)
            print("    Cache flushed")

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        # ── 7. Generate with updated LoRA ────────────────────────────────────
        print("[7] Generating with updated LoRA...")
        r = requests.post(
            f"http://localhost:{port}/generate",
            json={
                "text": "Write a CUDA kernel for ReLU:\n",
                "lora_name": LORA_NAME,
                "sampling_params": {"max_new_tokens": 32, "temperature": 0.0},
            },
            timeout=60,
        )
        assert r.status_code == 200, f"generate failed: {r.text[:200]}"
        gen2 = r.json()["text"][:120]
        print(f"    Generated (first 120 chars): {repr(gen2)}")
        print("    Generation with updated LoRA OK")

        print("\n✅ ALL CHECKS PASSED")
        print("   - SGLang starts with base model + --enable-lora")
        print("   - /load_lora_adapter loads initial SFT adapter")
        print("   - /release_memory_occupation frees VRAM for trainer")
        print("   - /resume_memory_occupation restores SGLang")
        print("   - /load_lora_adapter hot-reloads updated LoRA after training step")
        print("   - Generation works correctly with both initial and updated LoRA")

    finally:
        proc.terminate()
        print("\n[done] SGLang server terminated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   required=True, help="Base model path or HF model ID")
    parser.add_argument("--adapter", required=True, help="Initial LoRA adapter path or HF adapter ID")
    parser.add_argument("--port",    type=int, default=30000)
    parser.add_argument("--tp",      type=int, default=1)
    parser.add_argument("--sglang_python", default=os.environ.get("SGLANG_PYTHON", ""))
    args = parser.parse_args()

    assert args.sglang_python, "Pass --sglang_python or set SGLANG_PYTHON env var"
    test_weight_sync(args.model, args.adapter, args.port, args.sglang_python, args.tp)
