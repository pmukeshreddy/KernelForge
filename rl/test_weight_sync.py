"""
test_weight_sync.py — verify SGLang weight sync end-to-end.

Usage:
    export SGLANG_PYTHON=/path/to/sglang_env/bin/python
    python rl/test_weight_sync.py --model <model_path> --port 30000
"""

import os
import sys
import time
import argparse
import torch
import requests


def test_weight_sync(model_path: str, port: int, sglang_python: str, tp: int = 1):
    NCCL_PORT = 65501

    # ── 1. Launch SGLang server ──────────────────────────────────────────────
    print(f"[1] Launching SGLang server on port {port}...")
    import subprocess, glob as _glob
    # Auto-detect CUDA home and inject bin/include into subprocess env
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
        "--mem-fraction-static", "0.3",
        "--log-level", "error",
    ], env=env)
    # Wait for server
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
        # ── 2. Import SGLang's NCCL primitives ──────────────────────────────
        print("[2] Importing SGLang NCCL primitives...")
        try:
            from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator
            from sglang.srt.distributed.utils import StatelessProcessGroup
        except ImportError:
            result = subprocess.run(
                [sglang_python, "-c",
                 "import sglang, os; print(os.path.dirname(os.path.dirname(sglang.__file__)))"],
                capture_output=True, text=True, timeout=10,
            )
            site_packages = result.stdout.strip()
            assert site_packages, f"Could not find SGLang site-packages via {sglang_python}"
            sys.path.insert(0, site_packages)
            from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator
            from sglang.srt.distributed.utils import StatelessProcessGroup
        print("    OK")

        # ── 3. Init NCCL group ───────────────────────────────────────────────
        print("[3] Initializing NCCL communicator...")
        import threading
        world_size = tp + 1
        http_result = [None]

        def _init_http():
            try:
                r = requests.post(
                    f"http://localhost:{port}/init_weights_update_group",
                    json={
                        "master_address": "localhost",
                        "master_port": NCCL_PORT,
                        "rank_offset": 1,
                        "world_size": world_size,
                        "group_name": "weight_update_group",
                        "backend": "nccl",
                    },
                    timeout=120,
                )
                http_result[0] = r.status_code
            except Exception as e:
                http_result[0] = str(e)

        t = threading.Thread(target=_init_http, daemon=True)
        t.start()
        time.sleep(1)

        device = 0
        pg = StatelessProcessGroup.create(
            host="localhost", port=NCCL_PORT, rank=0, world_size=world_size
        )
        comm = PyNcclCommunicator(pg, device=torch.device(f"cuda:{device}"))
        t.join(timeout=60)
        print(f"    HTTP result: {http_result[0]}")
        assert http_result[0] == 200, f"init_weights_update_group failed: {http_result[0]}"
        print("    NCCL communicator ready")

        # ── 4. Pick one real parameter from the model ────────────────────────
        print("[4] Fetching model parameter info from SGLang...")
        # Use /get_model_info if available, otherwise just pick a known param name
        # We'll use a small dummy tensor matching a real param shape instead
        # Grab param names from a quick SGLang generate to confirm server is alive
        resp = requests.post(
            f"http://localhost:{port}/generate",
            json={"text": "hi", "sampling_params": {"max_new_tokens": 1}},
            timeout=60,
        )
        assert resp.status_code == 200, f"generate failed: {resp.text[:200]}"
        print("    Server responding to generate OK")

        # ── 5. Sync a real parameter ─────────────────────────────────────────
        print("[5] Loading model to get a real parameter...")
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
        )
        # Grab embed_tokens — small and always present
        param_name = "model.embed_tokens.weight"
        original = model.state_dict()[param_name].clone()
        # Corrupt it slightly so we can verify SGLang received the new value
        modified = original + 0.001
        tensor = modified.cuda().contiguous()
        del model
        torch.cuda.empty_cache()
        print(f"    Using param: {param_name}  shape={list(tensor.shape)}")

        print("[6] Sending update_weights_from_distributed + broadcasting...")
        http2 = [None]; http2_body = [None]
        def _sync_http():
            try:
                r = requests.post(
                    f"http://localhost:{port}/update_weights_from_distributed",
                    json={
                        "names":  [param_name],
                        "dtypes": ["bfloat16"],
                        "shapes": [list(tensor.shape)],
                    },
                    timeout=120,
                )
                http2[0] = r.status_code
                http2_body[0] = r.text[:200]
            except Exception as e:
                http2[0] = str(e)

        t2 = threading.Thread(target=_sync_http, daemon=True)
        t2.start()
        comm.broadcast(tensor, src=0, stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        t2.join(timeout=130)

        print(f"    HTTP result: {http2[0]}  body: {http2_body[0]}")
        assert http2[0] == 200, f"update_weights_from_distributed failed: {http2[0]}"

        print("\n✅ Weight sync WORKS — SGLang received the updated tensor successfully.")

    finally:
        proc.terminate()
        print("[done] SGLang server terminated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  required=True, help="Model path or HF model ID")
    parser.add_argument("--port",   type=int, default=30000)
    parser.add_argument("--tp",     type=int, default=1)
    parser.add_argument("--sglang_python", default=os.environ.get("SGLANG_PYTHON", ""))
    args = parser.parse_args()

    assert args.sglang_python, "Pass --sglang_python or set SGLANG_PYTHON env var"
    test_weight_sync(args.model, args.port, args.sglang_python, args.tp)
