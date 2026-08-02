"""Run Phase 5 Qwen2 repair scenarios against local phase2 worker+coordinator (Windows).

Writes: bench/results/orcaforge-qwen2-repair.{csv,manifest.json}

Reproduce:
  .venv311\\Scripts\\python.exe scripts/run_orcaforge_qwen2_repair.py
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = os.getenv("RUN_ID", "orcaforge-qwen2-repair")
MODEL = os.getenv("MODEL", "Qwen/Qwen2-1.5B-Instruct")
SCENARIOS = os.getenv("SCENARIOS", "bench/scenarios/head_to_head_qwen2_repair.yaml")
PYTHON = os.getenv("PYTHON_BIN", sys.executable)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_health(url: str, proc: subprocess.Popen[str], timeout_s: float = 300.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if httpx.get(url, timeout=2.0).status_code < 500:
                return
        except Exception:
            pass
        if proc.poll() is not None:
            err = proc.stderr.read() if proc.stderr else ""
            raise RuntimeError(f"process exited early: {err}")
        time.sleep(0.5)
    raise RuntimeError(f"health timeout: {url}")


def main() -> None:
    worker_port = _free_port()
    coord_port = _free_port()
    env = os.environ.copy()
    env.update(
        {
            "PHASE2_BACKEND": "transformers",
            "HF_MODEL_ID": MODEL,
            "HF_TORCH_DTYPE": env.get("HF_TORCH_DTYPE", "float16"),
            "HF_DEVICE": env.get("HF_DEVICE", "cuda"),
            "PHASE2_KV_BACKEND": env.get("PHASE2_KV_BACKEND", "dynamic"),
            "PHASE2_PREFIX_CACHE": env.get("PHASE2_PREFIX_CACHE", "0"),
            "PHASE2_BATCH_DECODE_STEPS": "1",
            "PHASE2_DECODE_STEP_MS": "1",
            "KV_TOTAL_BLOCKS": env.get("KV_TOTAL_BLOCKS", "4096"),
            "KV_BLOCK_SIZE_TOKENS": env.get("KV_BLOCK_SIZE_TOKENS", "16"),
            "WORKER_URLS": f"http://127.0.0.1:{worker_port}",
            "COORDINATOR_HEALTH_UNHEALTHY_THRESHOLD": "5",
            "COORDINATOR_DEFAULT_DEADLINE_MS": "900000",
            "COORDINATOR_ADMISSION_MAX": "128",
            "COORDINATOR_ADMISSION_MIN": "128",
            "HARNESS_HTTP_TIMEOUT_S": env.get("HARNESS_HTTP_TIMEOUT_S", "900"),
        }
    )

    worker = subprocess.Popen(
        [
            PYTHON,
            "-m",
            "uvicorn",
            "runtime.phase2.worker_server:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(worker_port),
            "--log-level",
            "warning",
        ],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_health(f"http://127.0.0.1:{worker_port}/healthz", worker, timeout_s=300.0)
        coord = subprocess.Popen(
            [
                PYTHON,
                "-m",
                "uvicorn",
                "runtime.phase2.coordinator_server:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(coord_port),
                "--log-level",
                "warning",
            ],
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            _wait_health(f"http://127.0.0.1:{coord_port}/healthz", coord, timeout_s=120.0)
            cmd = [
                PYTHON,
                "bench/harness/run.py",
                "--system",
                "phase2",
                "--base-url",
                f"http://127.0.0.1:{coord_port}/v1/chat/completions",
                "--model",
                MODEL,
                "--vllm-version",
                "0.21.0",
                "--scenarios",
                SCENARIOS,
                "--run-id",
                RUN_ID,
                "--gpu-type",
                "rtx5070-laptop",
                "--gpu-hour-usd",
                "2.50",
                "--inference-mode",
                "real_model_inference",
                "--determinism-check",
                "strict",
                "--enable-gpu-sampling",
                "--warmup-requests",
                "1",
            ]
            print("running:", " ".join(cmd), flush=True)
            subprocess.check_call(cmd, cwd=str(ROOT), env=env)
        finally:
            coord.terminate()
            try:
                coord.wait(timeout=30)
            except subprocess.TimeoutExpired:
                coord.kill()
                coord.wait(timeout=10)
    finally:
        worker.terminate()
        try:
            worker.wait(timeout=60)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.wait(timeout=10)


if __name__ == "__main__":
    main()
