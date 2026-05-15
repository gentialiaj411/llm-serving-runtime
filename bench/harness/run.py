from __future__ import annotations

import argparse
import asyncio
import csv
import datetime as dt
import hashlib
import json
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx
import yaml

CSV_COLUMNS = [
    "timestamp_utc","run_id","system_under_test","model_id","gpu_type","gpu_count","precision",
    "scenario_id","concurrency","request_count","prompt_tokens_p50","prompt_tokens_mean","prompt_tokens_p95",
    "gen_tokens_mean","duration_s","tokens_per_sec_output","tokens_per_sec_total",
    "ttft_ms_p50","ttft_ms_p95","ttft_ms_p99",
    "inter_token_latency_p50","inter_token_latency_p95","inter_token_latency_p99",
    "latency_ms_p50","latency_ms_p95","latency_ms_p99",
    "gpu_util_pct_p50","gpu_util_pct_p95","gpu_mem_used_mb_p50","gpu_mem_used_mb_p95",
    "success_rate","http_error_rate","timeout_rate","other_error_rate","est_dollars_per_million_output_tokens"
]


def pctl(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * min(1.0, max(0.0, q))
    lower = int(pos)
    upper = min(lower + 1, len(xs) - 1)
    weight = pos - lower
    return float(xs[lower] * (1.0 - weight) + xs[upper] * weight)


class RequestFailure(Exception):
    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


def make_prompt(tokens: int, turns: int = 1) -> str:
    if turns <= 1:
        return " ".join(f"tok{i % 100}" for i in range(tokens))
    per_turn = max(1, tokens // turns)
    messages = []
    for t in range(turns):
        role = "User" if t % 2 == 0 else "Assistant"
        content = " ".join(f"tok{(t * per_turn + i) % 100}" for i in range(per_turn))
        messages.append(f"{role}: {content}")
    return "\n".join(messages)


def get_gpu_snapshot() -> tuple[float, float] | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=2,
        ).strip()
        if not out:
            return None
        first = out.splitlines()[0]
        util_s, mem_s = [x.strip() for x in first.split(",")[:2]]
        return float(util_s), float(mem_s)
    except Exception:
        return None


async def sample_gpu(stop_event: asyncio.Event, sink: list[tuple[float, float]], interval_s: float) -> None:
    while not stop_event.is_set():
        snap = get_gpu_snapshot()
        if snap is not None:
            sink.append(snap)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_s)
        except asyncio.TimeoutError:
            pass


async def wait_for_health(base_url: str, timeout_s: float = 180.0) -> None:
    health_candidates = [
        base_url.replace("/v1/chat/completions", "/healthz"),
        base_url.replace("/v1/chat/completions", "/health"),
    ]
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=5.0) as client:
        while time.perf_counter() - t0 < timeout_s:
            for health_url in health_candidates:
                try:
                    r = await client.get(health_url)
                    if r.status_code < 500:
                        return
                except Exception:
                    pass
            await asyncio.sleep(1.0)
    raise RuntimeError(f"Timed out waiting for health endpoints: {', '.join(health_candidates)}")


def launch_vllm(args: argparse.Namespace) -> subprocess.Popen[str] | None:
    if not args.launch_vllm:
        return None

    if shutil.which("python") is None:
        raise RuntimeError("python executable not found for vLLM launch")

    cmd = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model,
        "--host",
        args.vllm_host,
        "--port",
        str(args.vllm_port),
        "--dtype",
        "float16",
        "--max-model-len",
        str(args.vllm_max_model_len),
    ]
    if args.vllm_tensor_parallel_size > 1:
        cmd.extend(["--tensor-parallel-size", str(args.vllm_tensor_parallel_size)])

    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)


def _extract_stream_delta(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    choice = choices[0]
    delta = choice.get("delta") or {}
    if isinstance(delta, dict):
        content = delta.get("content")
        if content is not None:
            return str(content)
    message = choice.get("message") or {}
    if isinstance(message, dict):
        content = message.get("content")
        if content is not None:
            return str(content)
    return ""


async def one_request_streaming(
    client: httpx.AsyncClient, url: str, model: str, prompt_tokens: int, max_out: int, turns: int = 1
) -> dict[str, Any]:
    prompt = make_prompt(prompt_tokens, turns=turns)
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_out,
        "temperature": 0.0,
        "stream": True,
    }
    chunks: list[str] = []
    arrivals_ms: list[float] = []
    usage: dict[str, Any] = {}
    t0 = time.perf_counter()
    try:
        async with client.stream("POST", url, json=body) as resp:
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise RequestFailure("http", str(exc)) from exc

            async for line in resp.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    line = line[5:].strip()
                if line == "[DONE]":
                    break

                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RequestFailure("other", f"Non-JSON streaming chunk: {line[:80]}") from exc

                if isinstance(payload, dict) and payload.get("usage"):
                    usage = payload["usage"]

                delta = _extract_stream_delta(payload)
                if delta:
                    chunks.append(delta)
                    arrivals_ms.append((time.perf_counter() - t0) * 1000.0)
    except httpx.TimeoutException as exc:
        raise RequestFailure("timeout", str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        raise RequestFailure("http", str(exc)) from exc
    except RequestFailure:
        raise
    except Exception as exc:
        raise RequestFailure("other", str(exc)) from exc

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if not arrivals_ms:
        raise RequestFailure("other", "Streaming response contained no assistant token chunks")

    inter_token_ms = [
        arrivals_ms[i] - arrivals_ms[i - 1]
        for i in range(1, len(arrivals_ms))
    ]
    out = "".join(chunks)
    prompt_tok = int(usage.get("prompt_tokens", prompt_tokens)) if usage else prompt_tokens
    out_tok = int(usage.get("completion_tokens", len(chunks))) if usage else len(chunks)
    return {
        "latency_ms": elapsed_ms,
        "ttft_ms": arrivals_ms[0],
        "inter_token_ms": inter_token_ms,
        "prompt_tokens": prompt_tok,
        "out_tokens": out_tok,
        "output": out,
    }


async def run_scenario(base_url: str, model: str, scenario: dict, c: int, warmup_requests: int = 1) -> dict:
    req_count = max(20, c * 6)
    latencies: list[float] = []
    ttfts: list[float] = []
    itoks: list[float] = []
    ptoks: list[int] = []
    outtoks: list[int] = []
    timeout_failures = 0
    http_failures = 0
    other_failures = 0
    turns = int(scenario.get("turns", 1))

    sem = asyncio.Semaphore(c)
    async with httpx.AsyncClient(timeout=120.0) as client:
        for _ in range(max(0, warmup_requests)):
            try:
                await one_request_streaming(
                    client,
                    base_url,
                    model,
                    int(scenario["prompt_tokens"]),
                    int(scenario["max_output_tokens"]),
                    turns=turns,
                )
            except RequestFailure:
                pass

        async def run_one() -> None:
            nonlocal timeout_failures, http_failures, other_failures
            async with sem:
                try:
                    r = await one_request_streaming(
                        client, base_url, model, int(scenario["prompt_tokens"]), int(scenario["max_output_tokens"]), turns=turns
                    )
                    latencies.append(r["latency_ms"])
                    ttfts.append(r["ttft_ms"])
                    itoks.extend(r["inter_token_ms"])
                    ptoks.append(r["prompt_tokens"])
                    outtoks.append(r["out_tokens"])
                except RequestFailure as exc:
                    if exc.kind == "timeout":
                        timeout_failures += 1
                    elif exc.kind == "http":
                        http_failures += 1
                    else:
                        other_failures += 1

        t0 = time.perf_counter()
        await asyncio.gather(*[run_one() for _ in range(req_count)])
        duration_s = max(1e-9, time.perf_counter() - t0)

    failures = timeout_failures + http_failures + other_failures
    success = req_count - failures
    total_out = sum(outtoks)
    total_prompt = sum(ptoks)
    return {
        "scenario_id": scenario["id"],
        "concurrency": c,
        "request_count": req_count,
        "duration_s": duration_s,
        "prompt_tokens_p50": pctl([float(x) for x in ptoks], 0.50),
        "prompt_tokens_mean": statistics.fmean(ptoks) if ptoks else 0.0,
        "prompt_tokens_p95": pctl([float(x) for x in ptoks], 0.95),
        "gen_tokens_mean": statistics.fmean(outtoks) if outtoks else 0.0,
        "tokens_per_sec_output": total_out / duration_s,
        "tokens_per_sec_total": (total_prompt + total_out) / duration_s,
        "ttft_ms_p50": pctl(ttfts, 0.50),
        "ttft_ms_p95": pctl(ttfts, 0.95),
        "ttft_ms_p99": pctl(ttfts, 0.99),
        "inter_token_latency_p50": pctl(itoks, 0.50),
        "inter_token_latency_p95": pctl(itoks, 0.95),
        "inter_token_latency_p99": pctl(itoks, 0.99),
        "latency_ms_p50": pctl(latencies, 0.50),
        "latency_ms_p95": pctl(latencies, 0.95),
        "latency_ms_p99": pctl(latencies, 0.99),
        "success_rate": (success / req_count) if req_count else 0.0,
        "http_error_rate": (http_failures / req_count) if req_count else 0.0,
        "timeout_rate": (timeout_failures / req_count) if req_count else 0.0,
        "other_error_rate": (other_failures / req_count) if req_count else 0.0,
        "total_output_tokens": total_out,
    }


def determinism(outputs: list[str]) -> dict:
    h1 = hashlib.sha256(outputs[0].encode("utf-8")).hexdigest()
    h2 = hashlib.sha256(outputs[1].encode("utf-8")).hexdigest()
    return {"passed": h1 == h2, "hash_a": h1, "hash_b": h2}


def row_cost_per_million(row: dict, gpu_hour_usd: float, gpu_count: int) -> float:
    total_output = float(row.get("total_output_tokens", 0) or 0)
    if total_output <= 0:
        return 0.0
    row_cost_usd = gpu_hour_usd * gpu_count * (float(row["duration_s"]) / 3600.0)
    return row_cost_usd / (total_output / 1_000_000.0)


async def main_async(args: argparse.Namespace) -> None:
    vllm_proc = None
    gpu_samples: list[tuple[float, float]] = []
    sampler_stop = asyncio.Event()
    sampler_task = None
    try:
        vllm_proc = launch_vllm(args)
        if vllm_proc is not None:
            await wait_for_health(args.base_url)

        scenarios = yaml.safe_load(Path(args.scenarios).read_text(encoding="utf-8"))["scenarios"]
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")

        rows = []
        if args.enable_gpu_sampling:
            sampler_task = asyncio.create_task(sample_gpu(sampler_stop, gpu_samples, args.gpu_sample_interval_s))

        if args.dry_run:
            for s in scenarios:
                for c in s["concurrency"]:
                    rows.append({
                        "scenario_id": s["id"], "concurrency": c, "request_count": max(20, c * 6),
                        "duration_s": 1.0, "prompt_tokens_p50": float(s.get("prompt_tokens", 0)), "prompt_tokens_mean": float(s.get("prompt_tokens", 0)),
                        "prompt_tokens_p95": float(s.get("prompt_tokens", 0)), "gen_tokens_mean": float(s.get("max_output_tokens", 0)),
                        "tokens_per_sec_output": 0.0, "tokens_per_sec_total": 0.0,
                        "ttft_ms_p50": 0.0, "ttft_ms_p95": 0.0, "ttft_ms_p99": 0.0,
                        "inter_token_latency_p50": 0.0, "inter_token_latency_p95": 0.0, "inter_token_latency_p99": 0.0,
                        "latency_ms_p50": 0.0, "latency_ms_p95": 0.0, "latency_ms_p99": 0.0,
                        "success_rate": 1.0, "http_error_rate": 0.0, "timeout_rate": 0.0, "other_error_rate": 0.0, "total_output_tokens": 0,
                    })
            det = determinism(["dry", "dry"])
        else:
            for s in scenarios:
                for c in s["concurrency"]:
                    rows.append(await run_scenario(args.base_url, args.model, s, int(c), args.warmup_requests))
            if args.determinism_check == "skip":
                det = {"passed": True, "skipped": True, "reason": "disabled by --determinism-check skip"}
            else:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    s1 = await one_request_streaming(client, args.base_url, args.model, 32, 32)
                    s2 = await one_request_streaming(client, args.base_url, args.model, 32, 32)
                det = determinism([s1["output"], s2["output"]])

        if not det["passed"] and args.determinism_check == "strict":
            raise SystemExit("Determinism check failed for temperature=0")
        det["mode"] = args.determinism_check

        sampler_stop.set()
        if sampler_task is not None:
            await sampler_task

        gpu_utils = [u for u, _ in gpu_samples]
        gpu_mems = [m for _, m in gpu_samples]
        gpu_metrics_valid = bool(args.enable_gpu_sampling and gpu_samples)
        inference_mode = args.inference_mode
        if inference_mode == "auto":
            inference_mode = "real_model_inference" if args.system.lower() == "vllm" else "stub_token_generation"
        total_duration_h = sum(r["duration_s"] for r in rows) / 3600.0
        total_output = sum(r["total_output_tokens"] for r in rows)
        cost_usd = args.gpu_hour_usd * args.gpu_count * total_duration_h
        aggregate_est_per_million = (cost_usd / (total_output / 1_000_000.0)) if total_output > 0 else 0.0

        csv_path = out_dir / f"{args.run_id}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            w.writeheader()
            for r in rows:
                w.writerow({
                    "timestamp_utc": ts, "run_id": args.run_id, "system_under_test": args.system, "model_id": args.model,
                    "gpu_type": args.gpu_type, "gpu_count": args.gpu_count, "precision": "fp16",
                    "scenario_id": r["scenario_id"], "concurrency": r["concurrency"], "request_count": r["request_count"],
                    "prompt_tokens_p50": r["prompt_tokens_p50"], "prompt_tokens_mean": r["prompt_tokens_mean"], "prompt_tokens_p95": r["prompt_tokens_p95"],
                    "gen_tokens_mean": r["gen_tokens_mean"], "duration_s": r["duration_s"],
                    "tokens_per_sec_output": r["tokens_per_sec_output"], "tokens_per_sec_total": r["tokens_per_sec_total"],
                    "ttft_ms_p50": r["ttft_ms_p50"], "ttft_ms_p95": r["ttft_ms_p95"], "ttft_ms_p99": r["ttft_ms_p99"],
                    "inter_token_latency_p50": r["inter_token_latency_p50"], "inter_token_latency_p95": r["inter_token_latency_p95"], "inter_token_latency_p99": r["inter_token_latency_p99"],
                    "latency_ms_p50": r["latency_ms_p50"], "latency_ms_p95": r["latency_ms_p95"], "latency_ms_p99": r["latency_ms_p99"],
                    "gpu_util_pct_p50": pctl(gpu_utils, 0.50), "gpu_util_pct_p95": pctl(gpu_utils, 0.95),
                    "gpu_mem_used_mb_p50": pctl(gpu_mems, 0.50), "gpu_mem_used_mb_p95": pctl(gpu_mems, 0.95),
                    "success_rate": r["success_rate"], "http_error_rate": r["http_error_rate"], "timeout_rate": r["timeout_rate"],
                    "other_error_rate": r["other_error_rate"],
                    "est_dollars_per_million_output_tokens": row_cost_per_million(r, args.gpu_hour_usd, args.gpu_count),
                })

        manifest = {
            "run_id": args.run_id,
            "timestamp_utc": ts,
            "system_under_test": args.system,
            "model_id": args.model,
            "headline_vllm_version": args.vllm_version,
            "gpu_type": args.gpu_type,
            "gpu_count": args.gpu_count,
            "gpu_hour_usd": args.gpu_hour_usd,
            "base_url": args.base_url,
            "inference_mode": inference_mode,
            "gpu_metrics_valid": gpu_metrics_valid,
            "determinism": det,
            "aggregate_est_dollars_per_million_output_tokens": aggregate_est_per_million,
            "rows": len(rows),
            "launch_vllm": args.launch_vllm,
            "warmup_requests_per_scenario": args.warmup_requests,
            "gpu_sampler": {
                "enabled": args.enable_gpu_sampling,
                "interval_s": args.gpu_sample_interval_s,
                "samples_collected": len(gpu_samples),
            },
            "metric_notes": {
                "ttft": "Measured from request send to first streamed assistant token chunk.",
                "inter_token_latency": "Measured as gaps between streamed assistant token chunk arrivals.",
                "gpu": "Valid only when gpu_metrics_valid is true; otherwise CSV GPU columns are zeros from missing samples.",
            },
        }
        manifest_path = out_dir / f"{args.run_id}.manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        print(f"wrote: {csv_path}")
        print(f"wrote: {manifest_path}")
    finally:
        sampler_stop.set()
        if sampler_task is not None and not sampler_task.done():
            await sampler_task
        if vllm_proc is not None:
            vllm_proc.terminate()
            try:
                vllm_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                vllm_proc.kill()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1/chat/completions")
    p.add_argument("--system", default="phase1")
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--gpu-type", default="unknown")
    p.add_argument("--gpu-count", type=int, default=1)
    p.add_argument("--gpu-hour-usd", type=float, default=2.5)
    p.add_argument("--vllm-version", default="0.8.5")
    p.add_argument("--scenarios", default="bench/scenarios/baseline.yaml")
    p.add_argument("--output-dir", default="bench/results")
    p.add_argument("--run-id", default=f"run-{dt.datetime.now(dt.UTC).strftime('%Y%m%d%H%M%S')}")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--inference-mode",
        default="auto",
        choices=["auto", "stub_token_generation", "real_model_inference", "unknown"],
        help="Declare whether the endpoint is real model inference or a stub-token runtime.",
    )
    p.add_argument(
        "--determinism-check",
        default="strict",
        choices=["strict", "warn", "skip"],
        help="Use strict to fail on mismatch, warn to record mismatch only, or skip to avoid the check for nondeterministic backends.",
    )
    p.add_argument("--warmup-requests", type=int, default=1, help="Streaming warmup requests per scenario/concurrency before timed measurement.")

    p.add_argument("--launch-vllm", action="store_true")
    p.add_argument("--vllm-host", default="127.0.0.1")
    p.add_argument("--vllm-port", type=int, default=8000)
    p.add_argument("--vllm-max-model-len", type=int, default=4096)
    p.add_argument("--vllm-tensor-parallel-size", type=int, default=1)

    p.add_argument("--enable-gpu-sampling", action="store_true")
    p.add_argument("--gpu-sample-interval-s", type=float, default=0.5)

    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
