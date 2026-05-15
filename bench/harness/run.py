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
    "success_rate","http_error_rate","timeout_rate","est_dollars_per_million_output_tokens"
]


def pctl(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    i = int(round((len(xs) - 1) * q))
    return float(xs[max(0, min(i, len(xs) - 1))])


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
    health_url = base_url.replace("/v1/chat/completions", "/health")
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=5.0) as client:
        while time.perf_counter() - t0 < timeout_s:
            try:
                r = await client.get(health_url)
                if r.status_code < 500:
                    return
            except Exception:
                pass
            await asyncio.sleep(1.0)
    raise RuntimeError(f"Timed out waiting for health endpoint: {health_url}")


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


async def one_request(
    client: httpx.AsyncClient, url: str, model: str, prompt_tokens: int, max_out: int, turns: int = 1
) -> dict:
    prompt = make_prompt(prompt_tokens, turns=turns)
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_out,
        "temperature": 0.0,
        "stream": False,
    }
    t0 = time.perf_counter()
    resp = await client.post(url, json=body)
    elapsed = (time.perf_counter() - t0) * 1000.0
    resp.raise_for_status()
    payload = resp.json()
    out = payload["choices"][0]["message"]["content"]
    prompt_tok = int(payload.get("usage", {}).get("prompt_tokens", prompt_tokens))
    out_tok = int(payload.get("usage", {}).get("completion_tokens", max_out))
    return {
        "latency_ms": elapsed,
        "ttft_ms": elapsed,
        "itok_ms": elapsed / max(1, out_tok),
        "prompt_tokens": prompt_tok,
        "out_tokens": out_tok,
        "output": out,
    }


async def one_request_streaming(
    client: httpx.AsyncClient, url: str, model: str, prompt_tokens: int, max_out: int, turns: int = 1
) -> str:
    prompt = make_prompt(prompt_tokens, turns=turns)
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_out,
        "temperature": 0.0,
        "stream": True,
    }
    chunks: list[str] = []
    async with client.stream("POST", url, json=body) as resp:
        resp.raise_for_status()
        async for chunk in resp.aiter_text():
            chunks.append(chunk)
    return "".join(chunks)


async def run_scenario(base_url: str, model: str, scenario: dict, c: int) -> dict:
    req_count = max(20, c * 6)
    latencies: list[float] = []
    ttfts: list[float] = []
    itoks: list[float] = []
    ptoks: list[int] = []
    outtoks: list[int] = []
    failures = 0

    sem = asyncio.Semaphore(c)
    async with httpx.AsyncClient(timeout=120.0) as client:
        async def run_one() -> None:
            nonlocal failures
            async with sem:
                try:
                    r = await one_request(
                        client, base_url, model, int(scenario["prompt_tokens"]), int(scenario["max_output_tokens"]), turns=turns
                    )
                    latencies.append(r["latency_ms"])
                    ttfts.append(r["ttft_ms"])
                    itoks.append(r["itok_ms"])
                    ptoks.append(r["prompt_tokens"])
                    outtoks.append(r["out_tokens"])
                except Exception:
                    failures += 1

        t0 = time.perf_counter()
        await asyncio.gather(*[run_one() for _ in range(req_count)])
        duration_s = max(1e-9, time.perf_counter() - t0)

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
        "http_error_rate": (failures / req_count) if req_count else 0.0,
        "timeout_rate": 0.0,
        "total_output_tokens": total_out,
    }


def determinism(outputs: list[str]) -> dict:
    h1 = hashlib.sha256(outputs[0].encode("utf-8")).hexdigest()
    h2 = hashlib.sha256(outputs[1].encode("utf-8")).hexdigest()
    return {"passed": h1 == h2, "hash_a": h1, "hash_b": h2}


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
                        "success_rate": 1.0, "http_error_rate": 0.0, "timeout_rate": 0.0, "total_output_tokens": 0,
                    })
            det = determinism(["dry", "dry"])
        else:
            for s in scenarios:
                for c in s["concurrency"]:
                    rows.append(await run_scenario(args.base_url, args.model, s, int(c)))
            async with httpx.AsyncClient(timeout=120.0) as client:
                s1 = await one_request_streaming(client, args.base_url, args.model, 32, 32)
                s2 = await one_request_streaming(client, args.base_url, args.model, 32, 32)
            det = determinism([s1, s2])

        if not det["passed"]:
            raise SystemExit("Determinism check failed for temperature=0")

        sampler_stop.set()
        if sampler_task is not None:
            await sampler_task

        gpu_utils = [u for u, _ in gpu_samples]
        gpu_mems = [m for _, m in gpu_samples]
        total_duration_h = sum(r["duration_s"] for r in rows) / 3600.0
        total_output = sum(r["total_output_tokens"] for r in rows)
        cost_usd = args.gpu_hour_usd * args.gpu_count * total_duration_h
        est_per_million = (cost_usd / (total_output / 1_000_000.0)) if total_output > 0 else 0.0

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
                    "est_dollars_per_million_output_tokens": est_per_million,
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
            "determinism": det,
            "rows": len(rows),
            "launch_vllm": args.launch_vllm,
            "gpu_sampler": {
                "enabled": args.enable_gpu_sampling,
                "interval_s": args.gpu_sample_interval_s,
                "samples_collected": len(gpu_samples),
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
