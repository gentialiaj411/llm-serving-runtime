"""Live verify: paged KV + prefix cache shared-suffix hits (Approach A).

Writes: bench/results/paged_prefix_verify.json

Reproduce:
  .venv311\\Scripts\\python.exe bench/scripts/paged_prefix_verify.py
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


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _build_shared_prompt(model_id: str, target_tokens: int) -> str:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    chunk = (
        "You are a helpful assistant for Orcaforge paged-prefix verification. "
        "Answer concisely and follow instructions. "
    )
    text = chunk
    while True:
        encoded = tokenizer(text, return_tensors="pt")
        n = int(encoded["input_ids"].shape[1])
        if n >= target_tokens:
            break
        text += chunk
    encoded = tokenizer(text, return_tensors="pt")
    ids = encoded["input_ids"][0, :target_tokens].tolist()
    return tokenizer.decode(ids, skip_special_tokens=True)


def _start_worker(model_id: str) -> tuple[subprocess.Popen[str], int]:
    port = _free_port()
    env = os.environ.copy()
    env.pop("HF_MODEL_ID", None)
    env.update(
        {
            "PHASE2_BACKEND": "transformers",
            "HF_MODEL_ID": model_id,
            "HF_TORCH_DTYPE": env.get("HF_TORCH_DTYPE", "float16"),
            "HF_DEVICE": env.get("HF_DEVICE", "cuda"),
            "PHASE2_KV_BACKEND": "paged",
            "PHASE2_PREFIX_CACHE": "1",
            "PHASE2_MAX_ACTIVE": "8",
            "PHASE2_BATCH_DECODE_STEPS": "1",
            "PHASE2_DECODE_STEP_MS": "1",
            "KV_TOTAL_BLOCKS": env.get("KV_TOTAL_BLOCKS", "4096"),
            "KV_BLOCK_SIZE_TOKENS": env.get("KV_BLOCK_SIZE_TOKENS", "16"),
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
    deadline = time.time() + 300.0
    while time.time() < deadline:
        try:
            if httpx.get(health, timeout=2.0).status_code == 200:
                return proc, port
        except Exception:
            pass
        if proc.poll() is not None:
            err = proc.stderr.read() if proc.stderr else ""
            raise RuntimeError(f"worker exited: {err}")
        time.sleep(0.5)
    proc.terminate()
    proc.wait(timeout=10)
    raise RuntimeError("worker not healthy")


async def _one_stream(
    client: httpx.AsyncClient,
    base_url: str,
    request_id: str,
    prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    output_tokens = 0
    status = "error"
    error = None
    async with client.stream(
        "POST",
        f"{base_url}/generate_stream",
        json={
            "request_id": request_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
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
                error = event.get("error")
                break
    return {
        "request_id": request_id,
        "status": status,
        "output_tokens": output_tokens,
        "error": error,
    }


async def _run_workload(
    base_url: str,
    system_prompt: str,
    request_count: int,
    max_tokens: int,
    concurrency: int,
) -> dict[str, Any]:
    sem = asyncio.Semaphore(concurrency)
    start = time.perf_counter()

    async with httpx.AsyncClient(timeout=None) as client:

        async def _guarded(i: int) -> dict[str, Any]:
            async with sem:
                suffix = f" User question {i}: explain topic {i % 17} in one sentence."
                return await _one_stream(
                    client,
                    base_url,
                    f"paged-pfx-{i}-{time.time_ns()}",
                    system_prompt + suffix,
                    max_tokens,
                )

        results = await asyncio.gather(*[_guarded(i) for i in range(request_count)])

    elapsed = max(1e-6, time.perf_counter() - start)
    completed = sum(1 for r in results if r["status"] == "completed")
    errors = [r for r in results if r["status"] != "completed"]
    output_tokens = sum(int(r["output_tokens"]) for r in results)
    return {
        "request_count": request_count,
        "concurrency": concurrency,
        "completed_requests": completed,
        "success_rate": completed / max(1, request_count),
        "output_tokens": output_tokens,
        "output_tokens_per_sec": output_tokens / elapsed,
        "duration_s": elapsed,
        "sample_errors": [
            {"status": e["status"], "error": e.get("error")} for e in errors[:5]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen2-1.5B-Instruct")
    parser.add_argument("--shared-prefix-tokens", type=int, default=256)
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--output-json", default="bench/results/paged_prefix_verify.json")
    parser.add_argument("--min-success", type=float, default=0.99)
    parser.add_argument("--min-hits", type=int, default=1)
    args = parser.parse_args()

    system_prompt = _build_shared_prompt(args.model_id, args.shared_prefix_tokens)
    proc, port = _start_worker(args.model_id)
    base_url = f"http://127.0.0.1:{port}"
    try:
        metrics = asyncio.run(
            _run_workload(
                base_url,
                system_prompt,
                args.requests,
                args.max_new_tokens,
                args.concurrency,
            )
        )
        worker_metrics = httpx.get(f"{base_url}/metrics", timeout=10.0).json()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)

    hits = int(worker_metrics.get("prefix_cache_hits", 0) or 0)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_type": "paged_prefix_verify",
        "model_id": args.model_id,
        "kv_backend": "paged",
        "prefix_cache": True,
        "shared_prefix_tokens": args.shared_prefix_tokens,
        "request_count": args.requests,
        "max_new_tokens": args.max_new_tokens,
        "concurrency": args.concurrency,
        "measurement": "measured_on_gpu",
        "success_rate": metrics["success_rate"],
        "output_tokens_per_sec": metrics["output_tokens_per_sec"],
        "prefix_cache_hits": hits,
        "prefix_cache_misses": worker_metrics.get("prefix_cache_misses", 0),
        "prefix_cache_hit_rate": worker_metrics.get("prefix_cache_hit_rate", 0.0),
        "prefix_cache_inserts": worker_metrics.get("prefix_cache_inserts", 0),
        "sample_errors": metrics.get("sample_errors", []),
        "pass": bool(metrics["success_rate"] >= args.min_success and hits >= args.min_hits),
        "gates": {"min_success": args.min_success, "min_hits": args.min_hits},
    }
    out = ROOT / args.output_json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("pass", "success_rate", "prefix_cache_hits", "output_tokens_per_sec")}, indent=2))
    print(f"wrote: {out}")
    if not payload["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
