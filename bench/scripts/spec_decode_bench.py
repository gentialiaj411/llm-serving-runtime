from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wait_health(port: int) -> None:
    deadline = time.time() + 45.0
    while time.time() < deadline:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=1.0).status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.25)
    raise RuntimeError("worker did not become healthy")


def run_case(speculative: bool, args: argparse.Namespace) -> dict[str, float]:
    port = free_port()
    env = os.environ.copy()
    env["PHASE2_BACKEND"] = "transformers"
    env["PHASE2_SPECULATIVE"] = "1" if speculative else "0"
    env["PHASE2_SPEC_K"] = str(args.spec_k)
    env["HF_MODEL_ID"] = args.model
    env["HF_DRAFT_MODEL_ID"] = args.draft_model
    env["HF_TORCH_DTYPE"] = args.dtype
    env["HF_DEVICE"] = args.device
    cmd = [sys.executable, "-m", "uvicorn", "runtime.phase2.worker_server:app", "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True, env=env)
    tokens = 0
    try:
        wait_health(port)
        prompt = " ".join(f"tok{i % 64}" for i in range(64))
        t0 = time.perf_counter()
        with httpx.Client(timeout=120.0) as client:
            for i in range(args.requests):
                body = {"request_id": f"spec-bench-{int(speculative)}-{i}", "prompt": prompt, "max_tokens": 64, "temperature": 0.0}
                with client.stream("POST", f"http://127.0.0.1:{port}/generate_stream", json=body) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        event = json.loads(line)
                        if event.get("type") == "token":
                            tokens += 1
        elapsed = max(1e-9, time.perf_counter() - t0)
        metrics = httpx.get(f"http://127.0.0.1:{port}/metrics", timeout=5.0).json()
        return {"tokens_per_sec": tokens / elapsed, "acceptance_rate": float(metrics.get("speculative_acceptance_rate", 0.0))}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--seed", type=int, default=5070)
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--draft-model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--spec-k", type=int, default=4)
    parser.add_argument("--output", default="bench/results/spec-decode-comparison.json")
    args = parser.parse_args()
    baseline = run_case(False, args)
    speculative = run_case(True, args)
    result = {"baseline_tokens_per_sec": baseline["tokens_per_sec"], "speculative_tokens_per_sec": speculative["tokens_per_sec"], "speedup_ratio": speculative["tokens_per_sec"] / baseline["tokens_per_sec"], "acceptance_rate": speculative["acceptance_rate"], "seed": args.seed}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote: {output}")


if __name__ == "__main__":
    main()
