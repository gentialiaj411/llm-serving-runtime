from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
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


def _start_worker(model_id: str, block_size_tokens: int, total_blocks: int, decode_step_ms: int) -> tuple[subprocess.Popen[str], int]:
    port = _free_port()
    env = os.environ.copy()
    env["PHASE2_BACKEND"] = "transformers"
    env["HF_MODEL_ID"] = model_id
    env["HF_TORCH_DTYPE"] = env.get("HF_TORCH_DTYPE", "float32")
    env["KV_BLOCK_SIZE_TOKENS"] = str(block_size_tokens)
    env["KV_TOTAL_BLOCKS"] = str(total_blocks)
    env["PHASE2_BATCH_DECODE_STEPS"] = "1"
    env["PHASE2_DECODE_STEP_MS"] = str(decode_step_ms)
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "runtime.phase2.worker_server:app", "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        env=env,
    )
    deadline = time.time() + 45.0
    health = f"http://127.0.0.1:{port}/healthz"
    while time.time() < deadline:
        try:
            if httpx.get(health, timeout=1.0).status_code == 200:
                return proc, port
        except Exception:
            pass
        time.sleep(0.25)
    proc.terminate()
    proc.wait(timeout=10)
    raise RuntimeError("worker did not become healthy")


async def _one_stream(client: httpx.AsyncClient, base_url: str, request_id: str, max_tokens: int) -> dict[str, Any]:
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
    return {"request_id": request_id, "status": status, "output_tokens": output_tokens}


async def _run_round(base_url: str, concurrency: int, total_requests: int, max_tokens: int) -> dict[str, Any]:
    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=None) as client:
        req_ids = [f"bench-c{concurrency}-r{i}-{time.time_ns()}" for i in range(total_requests)]
        chunks = [req_ids[i : i + concurrency] for i in range(0, len(req_ids), concurrency)]
        results: list[dict[str, Any]] = []
        for chunk in chunks:
            out = await asyncio.gather(*[_one_stream(client, base_url, rid, max_tokens) for rid in chunk])
            results.extend(out)
    elapsed = max(1e-6, time.perf_counter() - start)
    completed = sum(1 for r in results if r["status"] == "completed")
    output_tokens = sum(int(r["output_tokens"]) for r in results)
    errors = [r for r in results if r["status"] != "completed"]
    return {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "completed_requests": completed,
        "failed_requests": len(errors),
        "success_rate": completed / max(1, total_requests),
        "output_tokens": output_tokens,
        "output_tokens_per_sec": output_tokens / elapsed,
        "elapsed_sec": elapsed,
        "non_completed_statuses": sorted({str(e["status"]) for e in errors}),
    }


async def _run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    proc, port = _start_worker(args.model, args.kv_block_size_tokens, args.kv_total_blocks, args.decode_step_ms)
    artifact: dict[str, Any] = {
        "artifact_type": "continuous_batching_kv_live_smoke",
        "model_name": args.model,
        "backend": "transformers",
        "status": "failed",
    }
    try:
        base_url = f"http://127.0.0.1:{port}"
        async with httpx.AsyncClient(timeout=300.0) as client:
            warmup = await client.post(
                f"{base_url}/generate",
                json={"request_id": f"warmup-{time.time_ns()}", "prompt": "warm up", "max_tokens": 2, "temperature": 0.0},
            )
            warmup.raise_for_status()
            warm_payload = warmup.json()
            if warm_payload.get("error"):
                artifact["error"] = str(warm_payload.get("error"))
                return artifact

            rounds = []
            for c in args.concurrency:
                rounds.append(await _run_round(base_url, c, args.requests_per_concurrency, args.max_tokens))

            cancel_rid = f"cancel-smoke-{time.time_ns()}"
            cancel_seen = False
            async with client.stream(
                "POST",
                f"{base_url}/generate_stream",
                json={"request_id": cancel_rid, "prompt": "cancel smoke", "max_tokens": args.max_tokens * 4, "temperature": 0.0},
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    event = json.loads(line)
                    if event.get("type") == "token":
                        await client.post(f"{base_url}/cancel/{cancel_rid}")
                    if event.get("type") == "cancelled":
                        cancel_seen = True
                        break

            metrics = (await client.get(f"{base_url}/metrics")).json()
            c16 = next((r for r in rounds if r["concurrency"] == 16), None)
            accepted = (
                c16 is not None
                and c16["success_rate"] == 1.0
                and c16["completed_requests"] >= 16
                and int(metrics.get("peak_active_requests", 0)) >= 16
                and int(metrics.get("continuous_batches_total", 0)) > 0
                and int(metrics.get("max_batch_size", 0)) >= 2
                and int(metrics.get("active_allocations", -1)) == 0
                and cancel_seen
            )
            artifact.update(
                {
                    "concurrency_results": rounds,
                    "concurrency_16": c16,
                    "allocator_block_size_tokens": args.kv_block_size_tokens,
                    "allocations_total": metrics.get("allocations_total", 0),
                    "frees_total": metrics.get("frees_total", 0),
                    "allocation_failures_total": metrics.get("allocation_failures_total", 0),
                    "peak_kv_bytes": metrics.get("peak_kv_bytes", 0),
                    "peak_active_requests": metrics.get("peak_active_requests", 0),
                    "max_batch_size": metrics.get("max_batch_size", 0),
                    "continuous_batches_total": metrics.get("continuous_batches_total", 0),
                    "active_requests_current": metrics.get("active_requests", 0),
                    "active_allocations_current": metrics.get("active_allocations", 0),
                    "stream_completed_total": metrics.get("stream_completed_total", 0),
                    "stream_cancelled_total": metrics.get("stream_cancelled_total", 0),
                    "stream_timed_out_total": metrics.get("stream_timed_out_total", 0),
                    "request_errors_total": metrics.get("request_errors_total", 0),
                    "cancellation_cleanup_ok": cancel_seen and int(metrics.get("active_allocations", 0)) == 0,
                    "decode_step_ms": args.decode_step_ms,
                    "status": "ok" if accepted else "failed",
                    "note": "CPU-friendly real-transformer streaming concurrency smoke; not a vLLM comparison or production-scale throughput claim.",
                }
            )
            return artifact
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4, 16])
    parser.add_argument("--requests-per-concurrency", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--kv-block-size-tokens", type=int, default=16)
    parser.add_argument("--kv-total-blocks", type=int, default=4096)
    parser.add_argument("--decode-step-ms", type=int, default=8)
    parser.add_argument("--output", default="bench/results/continuous_batching_kv_live.json")
    args = parser.parse_args()
    artifact = asyncio.run(_run_benchmark(args))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"wrote: {output}")


if __name__ == "__main__":
    main()

