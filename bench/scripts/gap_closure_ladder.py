from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

from bench.harness.run import run_scenario


ROOT = Path(__file__).resolve().parents[2]
JSON_PATH = ROOT / "bench" / "results" / "gap_closure_ladder.json"
MD_PATH = ROOT / "bench" / "results" / "gap_closure_ladder.md"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_health(url: str, timeout_s: float = 180.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if httpx.get(url, timeout=3.0).status_code < 500:
                return
        except Exception:
            pass
        time.sleep(1.0)
    raise RuntimeError(f"timed out waiting for {url}")


def _start_services(args: argparse.Namespace) -> tuple[subprocess.Popen[str], subprocess.Popen[str], str]:
    worker_port = _free_port()
    coordinator_port = _free_port()
    env = os.environ.copy()
    env.update(
        {
            "PHASE2_BACKEND": args.backend,
            "PHASE2_KV_BACKEND": args.kv_backend,
            "HF_MODEL_ID": args.model,
            "HF_DEVICE": args.device,
            "HF_TORCH_DTYPE": args.dtype,
            "PHASE2_DECODE_STEP_MS": str(args.decode_step_ms),
            "WORKER_URLS": f"http://127.0.0.1:{worker_port}",
        }
    )
    worker = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "runtime.phase2.worker_server:app", "--host", "127.0.0.1", "--port", str(worker_port), "--log-level", "warning"],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    coordinator = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "runtime.phase2.coordinator_server:app", "--host", "127.0.0.1", "--port", str(coordinator_port), "--log-level", "warning"],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        _wait_health(f"http://127.0.0.1:{worker_port}/healthz")
        _wait_health(f"http://127.0.0.1:{coordinator_port}/healthz")
    except Exception:
        for proc in (coordinator, worker):
            proc.terminate()
        raise
    return worker, coordinator, f"http://127.0.0.1:{coordinator_port}/v1/chat/completions"


def _load_rows() -> list[dict]:
    if not JSON_PATH.exists():
        return []
    payload = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else payload.get("rows", [])


def _write_artifacts(rows: list[dict]) -> None:
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Gap-Closure Ladder",
        "",
        "| Phase | Change | c1 tok/s | c16 tok/s | TTFT p50 (ms) | TTFT p95 (ms) | vLLM ratio c16 | Backend | Timestamp |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['phase']} | {row['change_summary']} | {row['tok_s_c1']:.2f} | {row['tok_s_c16']:.2f} | "
            f"{row['ttft_p50_ms']:.2f} | {row['ttft_p95_ms']:.2f} | {row['vllm_ratio_c16']:.4f} | "
            f"{row['kv_backend']} | {row['timestamp']} |"
        )
    MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def _run(args: argparse.Namespace) -> dict:
    worker, coordinator, base_url = _start_services(args)
    try:
        scenarios = [
            {"id": "short_short", "prompt_tokens": 128, "max_output_tokens": 64},
            {"id": "shared_prefix", "prompt_tokens": 512, "max_output_tokens": 32},
        ]
        results: dict[tuple[str, int], dict] = {}
        for scenario in scenarios:
            for concurrency in ([1, 16] if scenario["id"] == "short_short" else [8]):
                results[(scenario["id"], concurrency)] = await run_scenario(
                    base_url, args.model, scenario, concurrency, warmup_requests=args.warmup_requests
                )
        c1 = results[("short_short", 1)]
        c16 = results[("short_short", 16)]
        vllm_c16 = args.vllm_c16_tok_s
        row = {
            "phase": args.phase,
            "change_summary": args.change_summary,
            "tok_s_c1": float(c1["tokens_per_sec_output"]),
            "tok_s_c16": float(c16["tokens_per_sec_output"]),
            "ttft_p50_ms": float(c16["ttft_ms_p50"]),
            "ttft_p95_ms": float(c16["ttft_ms_p95"]),
            "vllm_ratio_c16": float(c16["tokens_per_sec_output"]) / vllm_c16 if vllm_c16 else 0.0,
            "kv_backend": args.kv_backend,
            "model": args.model,
            "gpu_type": args.gpu_type,
            "shared_prefix": results[("shared_prefix", 8)],
            "commit_sha": "working-tree",
            "timestamp": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        }
        rows = _load_rows()
        rows.append(row)
        _write_artifacts(rows)
        print(json.dumps(row, indent=2))
        return row
    finally:
        for proc in (coordinator, worker):
            proc.terminate()
        for proc in (coordinator, worker):
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one matched gap-closure ladder rung.")
    parser.add_argument("--phase", default="0")
    parser.add_argument("--change-summary", default="baseline")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--backend", default="transformers")
    parser.add_argument("--kv-backend", default=os.getenv("PHASE2_KV_BACKEND", "dynamic"))
    parser.add_argument("--device", default=os.getenv("HF_DEVICE", "cuda"))
    parser.add_argument("--dtype", default=os.getenv("HF_TORCH_DTYPE", "float16"))
    parser.add_argument("--gpu-type", default="rtx5070-laptop")
    parser.add_argument("--decode-step-ms", type=int, default=0)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--vllm-c16-tok-s", type=float, default=735.27)
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
