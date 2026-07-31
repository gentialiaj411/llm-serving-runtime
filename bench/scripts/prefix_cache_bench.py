"""Benchmark prefix KV cache on shared-prefix workload (Qwen2-1.5B).

Outputs: bench/results/prefix_cache.json, bench/results/prefix_cache.md

Reproduce:
  .venv311\\Scripts\\python.exe bench/scripts/prefix_cache_bench.py
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
        "You are a helpful assistant for Orcaforge prefix-cache benchmarks. "
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


def _start_worker(prefix_cache: bool, model_id: str) -> tuple[subprocess.Popen[str], int]:
    port = _free_port()
    env = os.environ.copy()
    env.pop("HF_MODEL_ID", None)
    env.update(
        {
            "PHASE2_BACKEND": "transformers",
            "HF_MODEL_ID": model_id,
            "HF_TORCH_DTYPE": env.get("HF_TORCH_DTYPE", "float16"),
            "HF_DEVICE": env.get("HF_DEVICE", "cuda"),
            "PHASE2_PREFIX_CACHE": "1" if prefix_cache else "0",
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
    deadline = time.time() + 180.0
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
                break
    return {"request_id": request_id, "status": status, "output_tokens": output_tokens}


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
                    f"prefix-bench-{i}-{time.time_ns()}",
                    system_prompt + suffix,
                    max_tokens,
                )

        results = await asyncio.gather(*[_guarded(i) for i in range(request_count)])

    elapsed = max(1e-6, time.perf_counter() - start)
    completed = sum(1 for r in results if r["status"] == "completed")
    output_tokens = sum(int(r["output_tokens"]) for r in results)
    return {
        "request_count": request_count,
        "concurrency": concurrency,
        "completed_requests": completed,
        "success_rate": completed / max(1, request_count),
        "output_tokens": output_tokens,
        "output_tokens_per_sec": output_tokens / elapsed,
        "duration_s": elapsed,
    }


def _run_mode(
    prefix_cache: bool,
    system_prompt: str,
    model_id: str,
    request_count: int,
    max_tokens: int,
    concurrency: int,
) -> dict[str, Any]:
    proc, port = _start_worker(prefix_cache, model_id)
    base_url = f"http://127.0.0.1:{port}"
    try:
        metrics = asyncio.run(
            _run_workload(base_url, system_prompt, request_count, max_tokens, concurrency)
        )
        worker_metrics = httpx.get(f"{base_url}/metrics", timeout=10.0).json()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)

    return {
        "prefix_cache_enabled": prefix_cache,
        **metrics,
        "prefix_cache_hits": worker_metrics.get("prefix_cache_hits", 0),
        "prefix_cache_misses": worker_metrics.get("prefix_cache_misses", 0),
        "prefix_cache_hit_rate": worker_metrics.get("prefix_cache_hit_rate", 0.0),
        "prefix_cache_evictions": worker_metrics.get("prefix_cache_evictions", 0),
        "prefix_cache_entries": worker_metrics.get("prefix_cache_entries", 0),
    }


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    base = payload["modes"]["baseline"]
    cached = payload["modes"]["prefix_cache"]
    improvement = 100.0 * (
        cached["output_tokens_per_sec"] / max(1e-6, base["output_tokens_per_sec"]) - 1.0
    )
    lines = [
        "# Prefix KV cache benchmark",
        "",
        f"- Model: `{payload['model_id']}`",
        f"- Shared prefix tokens: `{payload['shared_prefix_tokens']}`",
        f"- Requests: `{payload['request_count']}`, concurrency `{payload['concurrency']}`, decode `{payload['max_new_tokens']}` tokens",
        "",
        "| Mode | tokens/sec | success | hit rate | evictions |",
        "|------|------------|---------|----------|-----------|",
        f"| Baseline (cache off) | {base['output_tokens_per_sec']:.2f} | {base['success_rate']:.2f} | n/a | n/a |",
        f"| Prefix cache on | {cached['output_tokens_per_sec']:.2f} | {cached['success_rate']:.2f} | {cached['prefix_cache_hit_rate']:.2f} | {cached['prefix_cache_evictions']} |",
        "",
        f"- Throughput improvement: **{improvement:.1f}%**",
        "",
        "Artifact: `bench/results/prefix_cache.json`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen2-1.5B-Instruct")
    parser.add_argument("--shared-prefix-tokens", type=int, default=512)
    parser.add_argument("--requests", type=int, default=24)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--output-json", default="bench/results/prefix_cache.json")
    parser.add_argument("--output-md", default="bench/results/prefix_cache.md")
    args = parser.parse_args()

    system_prompt = _build_shared_prompt(args.model_id, args.shared_prefix_tokens)
    baseline = _run_mode(False, system_prompt, args.model_id, args.requests, args.max_new_tokens, args.concurrency)
    cached = _run_mode(True, system_prompt, args.model_id, args.requests, args.max_new_tokens, args.concurrency)

    improvement = 100.0 * (
        cached["output_tokens_per_sec"] / max(1e-6, baseline["output_tokens_per_sec"]) - 1.0
    )
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model_id,
        "shared_prefix_tokens": args.shared_prefix_tokens,
        "request_count": args.requests,
        "max_new_tokens": args.max_new_tokens,
        "concurrency": args.concurrency,
        "measurement": "measured_on_gpu",
        "modes": {"baseline": baseline, "prefix_cache": cached},
        "throughput_improvement_percent": improvement,
    }

    out = ROOT / args.output_json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_md(ROOT / args.output_md, payload)
    print(f"wrote: {out}")
    print(f"throughput improvement: {improvement:.1f}%")


if __name__ == "__main__":
    main()
