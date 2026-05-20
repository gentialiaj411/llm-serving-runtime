from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _format_cmd(cmd: list[str]) -> str:
    return " ".join(cmd)


def _wait_for_health(url: str, timeout_s: float = 60.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(1.0)
    raise RuntimeError(f"Timed out waiting for {url}")


def _tail(path: Path, lines: int = 40) -> str:
    try:
        data = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(data[-lines:])


def build_harness_cmd(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "bench/harness/run.py",
        "--system",
        args.system,
        "--inference-mode",
        args.inference_mode,
        "--determinism-check",
        args.determinism_check,
        "--base-url",
        args.base_url,
        "--model",
        args.model,
        "--vllm-version",
        args.vllm_version,
        "--scenarios",
        args.scenarios,
        "--run-id",
        args.run_id,
        "--gpu-type",
        args.gpu_type,
        "--gpu-count",
        str(args.gpu_count),
        "--gpu-hour-usd",
        str(args.gpu_hour_usd),
        "--enable-gpu-sampling",
        "--warmup-requests",
        str(args.warmup_requests),
    ]


def build_server_cmd(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-m",
        "uvicorn",
        "frontend.phase1_server:app",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--log-level",
        args.server_log_level,
    ]


def main() -> int:
    p = argparse.ArgumentParser(description="Run the repo's reproducible final benchmark.")
    p.add_argument("--dry-run", action="store_true", help="Print and validate the command without starting the server.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1/chat/completions")
    p.add_argument("--system", default="runtime")
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--gpu-type", default="RTX5070Ti")
    p.add_argument("--gpu-count", type=int, default=1)
    p.add_argument("--gpu-hour-usd", type=float, default=0.9)
    p.add_argument("--vllm-version", default="0.8.5")
    p.add_argument("--scenarios", default="bench/scenarios/baseline.yaml")
    p.add_argument("--run-id", default="runtime-final-local")
    p.add_argument("--inference-mode", default="auto", choices=["auto", "stub_token_generation", "real_model_inference", "unknown"])
    p.add_argument("--determinism-check", default="strict", choices=["strict", "warn", "skip"])
    p.add_argument("--warmup-requests", type=int, default=1)
    p.add_argument("--server-log-level", default="warning")
    args = p.parse_args()

    harness_cmd = build_harness_cmd(args)
    server_cmd = build_server_cmd(args)

    print(f"server: {_format_cmd(server_cmd)}")
    print(f"benchmark: {_format_cmd(harness_cmd)}")

    if args.dry_run:
        with tempfile.TemporaryDirectory(prefix="final-benchmark-dry-run-") as tmpdir:
            return subprocess.run(
                harness_cmd + ["--dry-run", "--output-dir", tmpdir],
                cwd=ROOT,
                check=True,
            ).returncode

    with tempfile.NamedTemporaryFile(prefix="phase1-final-", suffix=".log", delete=False) as log_file:
        log_path = Path(log_file.name)
    log_handle = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        server_cmd,
        cwd=ROOT,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_for_health(f"http://{args.host}:{args.port}/healthz")
        return subprocess.run(harness_cmd, cwd=ROOT, check=True).returncode
    except Exception:
        print(f"server log tail ({log_path}):")
        tail = _tail(log_path)
        if tail:
            print(tail)
        raise
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        log_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
