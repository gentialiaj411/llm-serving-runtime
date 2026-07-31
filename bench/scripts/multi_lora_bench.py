"""Benchmark multi-LoRA hot-swap vs base-only on a cross-adapter request mix.

Outputs: bench/results/multi_lora.json, bench/results/multi_lora.md

Reproduce:
  .venv311\\Scripts\\pip.exe install peft>=0.13.0
  .venv311\\Scripts\\python.exe bench/scripts/multi_lora_bench.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ADAPTERS = ["base", "adapter_a", "adapter_b", "adapter_c"]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _start_worker(lora_enabled: bool, model_id: str) -> tuple[subprocess.Popen[str], int]:
    port = _free_port()
    env = os.environ.copy()
    env.pop("HF_MODEL_ID", None)
    env.update(
        {
            "PHASE2_BACKEND": "transformers",
            "HF_MODEL_ID": model_id,
            "HF_TORCH_DTYPE": env.get("HF_TORCH_DTYPE", "float16"),
            "HF_DEVICE": env.get("HF_DEVICE", "cuda"),
            "PHASE2_LORA": "1" if lora_enabled else "0",
            "LORA_ADAPTER_NAMES": ",".join(ADAPTERS),
            "PHASE2_BATCH_DECODE_STEPS": "1",
            "PHASE2_DECODE_STEP_MS": "1",
        }
    )
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "runtime.phase2.worker_server:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    health = f"http://127.0.0.1:{port}/healthz"
    deadline = time.time() + 180.0
    while time.time() < deadline:
        try:
            if httpx.get(health, timeout=2.0).status_code == 200:
                return proc, port
        except Exception:
            pass
        if proc.poll() is not None:
            raise RuntimeError(f"worker exited: {proc.stderr.read() if proc.stderr else ''}")
        time.sleep(0.5)
    proc.terminate()
    proc.wait(timeout=10)
    raise RuntimeError("worker not healthy")


async def _one_stream(
    client: httpx.AsyncClient,
    base_url: str,
    request_id: str,
    adapter: str,
    max_tokens: int,
) -> dict[str, Any]:
    output_tokens = 0
    status = "error"
    async with client.stream(
        "POST",
        f"{base_url}/generate_stream",
        json={
            "request_id": request_id,
            "prompt": f"adapter {adapter} prompt words here",
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "adapter": adapter,
        },
        timeout=600.0,
    ) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line:
                continue
            event = json.loads(line)
            kind = event.get("type")
            if kind == "token":
                output_tokens += 1
            elif kind == "done":
                status = "completed"
                break
            elif kind in {"cancelled", "timed_out", "error"}:
                status = str(kind)
                break
    return {"adapter": adapter, "status": status, "output_tokens": output_tokens}


async def _run_mix(base_url: str, request_count: int, max_tokens: int, concurrency: int) -> dict[str, Any]:
    sem = asyncio.Semaphore(concurrency)
    start = time.perf_counter()

    async with httpx.AsyncClient(timeout=None) as client:

        async def _guarded(i: int) -> dict[str, Any]:
            async with sem:
                adapter = ADAPTERS[i % len(ADAPTERS)]
                return await _one_stream(
                    client,
                    base_url,
                    f"lora-bench-{i}-{time.time_ns()}",
                    adapter,
                    max_tokens,
                )

        results = await asyncio.gather(*[_guarded(i) for i in range(request_count)])

    elapsed = max(1e-6, time.perf_counter() - start)
    completed = sum(1 for r in results if r["status"] == "completed")
    output_tokens = sum(int(r["output_tokens"]) for r in results)
    return {
        "request_count": request_count,
        "concurrency": concurrency,
        "adapter_mix": ADAPTERS,
        "completed_requests": completed,
        "success_rate": completed / max(1, request_count),
        "output_tokens": output_tokens,
        "output_tokens_per_sec": output_tokens / elapsed,
        "duration_s": elapsed,
    }


def _run_mode(lora_enabled: bool, model_id: str, request_count: int, max_tokens: int, concurrency: int) -> dict[str, Any]:
    proc, port = _start_worker(lora_enabled, model_id)
    base_url = f"http://127.0.0.1:{port}"
    try:
        metrics = asyncio.run(_run_mix(base_url, request_count, max_tokens, concurrency))
        worker_metrics = httpx.get(f"{base_url}/metrics", timeout=10.0).json()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)

    return {
        "lora_enabled": lora_enabled,
        **metrics,
        "lora_adapter_swaps_total": worker_metrics.get("lora_adapter_swaps_total", 0),
        "lora_adapters": worker_metrics.get("lora_adapters", []),
    }


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    base = payload["modes"]["base_only"]
    multi = payload["modes"]["multi_lora"]
    delta = 100.0 * (multi["output_tokens_per_sec"] / max(1e-6, base["output_tokens_per_sec"]) - 1.0)
    lines = [
        "# Multi-LoRA benchmark",
        "",
        f"- Model: `{payload['model_id']}`",
        f"- Requests: `{payload['request_count']}` round-robin across `{payload['adapter_mix']}`",
        "",
        "| Mode | tokens/sec | success | adapter swaps |",
        "|------|------------|---------|---------------|",
        f"| Base only (`PHASE2_LORA=0`) | {base['output_tokens_per_sec']:.2f} | {base['success_rate']:.2f} | n/a |",
        f"| Multi-LoRA (`PHASE2_LORA=1`) | {multi['output_tokens_per_sec']:.2f} | {multi['success_rate']:.2f} | {multi.get('lora_adapter_swaps_total', 0)} |",
        "",
        f"- Throughput delta (multi vs base-only): **{delta:.1f}%**",
        "",
        "Note: synthetic PEFT adapters unless `LORA_ADAPTER_PATHS_JSON` provides real checkpoints.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen2-1.5B-Instruct")
    parser.add_argument("--requests", type=int, default=24)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    base_only = _run_mode(False, args.model_id, args.requests, args.max_new_tokens, args.concurrency)
    multi_lora = _run_mode(True, args.model_id, args.requests, args.max_new_tokens, args.concurrency)
    delta = 100.0 * (
        multi_lora["output_tokens_per_sec"] / max(1e-6, base_only["output_tokens_per_sec"]) - 1.0
    )

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model_id,
        "request_count": args.requests,
        "max_new_tokens": args.max_new_tokens,
        "concurrency": args.concurrency,
        "adapter_mix": ADAPTERS,
        "measurement": "measured_on_gpu",
        "modes": {"base_only": base_only, "multi_lora": multi_lora},
        "throughput_delta_percent": delta,
    }

    out = ROOT / "bench/results/multi_lora.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_md(ROOT / "bench/results/multi_lora.md", payload)
    print(f"wrote: {out}")
    print(f"throughput delta: {delta:.1f}%")


if __name__ == "__main__":
    main()
