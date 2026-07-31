from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _pctl(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * min(1.0, max(0.0, q))
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    w = pos - lo
    return float(xs[lo] * (1.0 - w) + xs[hi] * w)


def _query_gpu_mem_mb() -> int:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    first = out.splitlines()[0].strip()
    return int(first)


def _start_worker(quant_mode: str, model_id: str, awq_model_id: str) -> tuple[subprocess.Popen[str], int]:
    port = _free_port()
    env = os.environ.copy()
    env["PHASE2_BACKEND"] = "transformers"
    env["PHASE2_QUANT"] = quant_mode
    env["HF_MODEL_ID"] = model_id
    env["HF_AWQ_MODEL_ID"] = awq_model_id
    env["HF_TORCH_DTYPE"] = env.get("HF_TORCH_DTYPE", "float16")
    env["HF_DEVICE"] = env.get("HF_DEVICE", "cuda")
    env["PYTHONUNBUFFERED"] = "1"
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
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
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
        time.sleep(0.5)
    proc.terminate()
    proc.wait(timeout=10)
    raise RuntimeError(f"worker did not become healthy for quant_mode={quant_mode}")


async def _one_request(client: httpx.AsyncClient, base_url: str, request_id: str, max_tokens: int) -> dict[str, Any]:
    start = time.perf_counter()
    output_tokens = 0
    status = "error"
    async with client.stream(
        "POST",
        f"{base_url}/generate_stream",
        json={"request_id": request_id, "prompt": "alpha beta gamma", "max_tokens": max_tokens, "temperature": 0.0},
        timeout=300.0,
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
            elif kind in {"cancelled", "timed_out", "error", "duplicate"}:
                status = str(kind)
                break
    latency_ms = (time.perf_counter() - start) * 1000.0
    return {"status": status, "latency_ms": latency_ms, "output_tokens": output_tokens}


async def _run_concurrency(base_url: str, concurrency: int, total_requests: int, max_tokens: int) -> dict[str, Any]:
    req_ids = [f"qbench-c{concurrency}-r{i}-{time.time_ns()}" for i in range(total_requests)]
    batches = [req_ids[i : i + concurrency] for i in range(0, len(req_ids), concurrency)]
    results: list[dict[str, Any]] = []
    peak_mem_mb = _query_gpu_mem_mb()
    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=None) as client:
        for group in batches:
            out = await asyncio.gather(*[_one_request(client, base_url, rid, max_tokens) for rid in group])
            results.extend(out)
            peak_mem_mb = max(peak_mem_mb, _query_gpu_mem_mb())
    elapsed = max(time.perf_counter() - start, 1e-6)
    done = [r for r in results if r["status"] == "completed"]
    lat = [float(r["latency_ms"]) for r in done]
    total_out = sum(int(r["output_tokens"]) for r in done)
    return {
        "concurrency": concurrency,
        "requests": total_requests,
        "completed": len(done),
        "success_rate": len(done) / max(1, total_requests),
        "throughput_output_tokens_per_sec": total_out / elapsed,
        "latency_p50_ms": _pctl(lat, 0.50),
        "latency_p99_ms": _pctl(lat, 0.99),
        "peak_vram_gb": peak_mem_mb / 1024.0,
    }


async def _run_mode(
    quant_mode: str,
    model_id: str,
    awq_model_id: str,
    concurrencies: list[int],
    requests_per_concurrency: int,
    max_tokens: int,
) -> dict[str, Any]:
    proc, port = _start_worker(quant_mode, model_id, awq_model_id)
    try:
        base_url = f"http://127.0.0.1:{port}"
        async with httpx.AsyncClient(timeout=300.0) as client:
            warm = await client.post(
                f"{base_url}/generate",
                json={"request_id": f"warm-{time.time_ns()}", "prompt": "warmup", "max_tokens": 2, "temperature": 0.0},
            )
            warm.raise_for_status()
        rows = []
        for c in concurrencies:
            rows.append(await _run_concurrency(base_url, c, requests_per_concurrency, max_tokens))
        return {
            "quant_mode": quant_mode,
            "model_id": awq_model_id if quant_mode == "int4" else model_id,
            "per_concurrency": rows,
            "peak_vram_gb": max(r["peak_vram_gb"] for r in rows),
        }
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


async def _quality_check(model_id: str, awq_model_id: str) -> dict[str, Any]:
    prompt = "Summarize continuous batching benefits in one paragraph."
    max_tokens = 64
    fp16_proc, fp16_port = _start_worker("none", model_id, awq_model_id)
    awq_proc, awq_port = _start_worker("int4", model_id, awq_model_id)
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            r1 = await client.post(
                f"http://127.0.0.1:{fp16_port}/generate",
                json={"request_id": f"q-fp16-{time.time_ns()}", "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0},
            )
            r2 = await client.post(
                f"http://127.0.0.1:{awq_port}/generate",
                json={"request_id": f"q-awq-{time.time_ns()}", "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0},
            )
            r1.raise_for_status()
            r2.raise_for_status()
            t1 = str(r1.json().get("text", "")).split()
            t2 = str(r2.json().get("text", "")).split()
            n = min(max_tokens, len(t1), len(t2))
            if n == 0:
                agreement = 0.0
            else:
                same = sum(1 for i in range(n) if t1[i] == t2[i])
                agreement = same / n
            return {
                "prompt": prompt,
                "tokens_compared": n,
                "token_level_agreement_rate": agreement,
            }
    finally:
        fp16_proc.terminate()
        awq_proc.terminate()
        try:
            fp16_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            fp16_proc.kill()
            fp16_proc.wait(timeout=10)
        try:
            awq_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            awq_proc.kill()
            awq_proc.wait(timeout=10)


async def _main_async(args: argparse.Namespace) -> dict[str, Any]:
    fp16 = await _run_mode("none", args.model_id, args.awq_model_id, args.concurrency, args.requests, args.max_tokens)
    awq = await _run_mode("int4", args.model_id, args.awq_model_id, args.concurrency, args.requests, args.max_tokens)
    quality = await _quality_check(args.model_id, args.awq_model_id)
    return {
        "artifact_type": "quantization_comparison",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fp16_baseline": fp16,
        "awq_int4": awq,
        "model_quality_check": quality,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Qwen/Qwen2-1.5B-Instruct")
    p.add_argument("--awq-model-id", default="Qwen/Qwen2-1.5B-Instruct-AWQ")
    p.add_argument("--concurrency", nargs="+", type=int, default=[1, 4, 16])
    p.add_argument("--requests", type=int, default=64)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--output", default="bench/results/quantization_comparison.json")
    args = p.parse_args()
    artifact = asyncio.run(_main_async(args))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()

