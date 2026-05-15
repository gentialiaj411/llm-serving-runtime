from __future__ import annotations

import argparse
import asyncio
import csv
import datetime as dt
import hashlib
import json
import statistics
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
    return float(xs[max(0, min(i, len(xs)-1))])


def make_prompt(tokens: int) -> str:
    return " ".join(f"tok{i%100}" for i in range(tokens))


async def one_request(client: httpx.AsyncClient, url: str, model: str, prompt_tokens: int, max_out: int) -> dict:
    prompt = make_prompt(prompt_tokens)
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


async def run_scenario(base_url: str, model: str, scenario: dict) -> dict:
    c = scenario["concurrency"][0]
    req_count = max(20, c * 6)
    latencies: list[float] = []
    ttfts: list[float] = []
    itoks: list[float] = []
    ptoks: list[int] = []
    outtoks: list[int] = []
    failures = 0

    sem = asyncio.Semaphore(c)
    async with httpx.AsyncClient(timeout=60.0) as client:
        async def run_one() -> None:
            nonlocal failures
            async with sem:
                try:
                    r = await one_request(client, base_url, model, int(scenario["prompt_tokens"]), int(scenario["max_output_tokens"]))
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
    tps_out = total_out / duration_s
    tps_total = (total_prompt + total_out) / duration_s
    return {
        "scenario_id": scenario["id"],
        "concurrency": c,
        "request_count": req_count,
        "duration_s": duration_s,
        "prompt_tokens_p50": pctl([float(x) for x in ptoks], 0.50),
        "prompt_tokens_mean": statistics.fmean(ptoks) if ptoks else 0.0,
        "prompt_tokens_p95": pctl([float(x) for x in ptoks], 0.95),
        "gen_tokens_mean": statistics.fmean(outtoks) if outtoks else 0.0,
        "tokens_per_sec_output": tps_out,
        "tokens_per_sec_total": tps_total,
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
    }


def determinism_check(outputs: list[str]) -> dict:
    if len(outputs) < 2:
        return {"passed": False, "reason": "insufficient_samples"}
    h1 = hashlib.sha256(outputs[0].encode("utf-8")).hexdigest()
    h2 = hashlib.sha256(outputs[1].encode("utf-8")).hexdigest()
    return {"passed": h1 == h2, "hash_a": h1, "hash_b": h2}


async def main_async(args: argparse.Namespace) -> None:
    scenarios = yaml.safe_load(Path(args.scenarios).read_text(encoding="utf-8"))["scenarios"]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    rows = []
    if args.dry_run:
        for s in scenarios:
            rows.append(
                {
                    "scenario_id": s["id"],
                    "concurrency": s["concurrency"][0],
                    "request_count": max(20, s["concurrency"][0] * 6),
                    "duration_s": 1.0,
                    "prompt_tokens_p50": float(s.get("prompt_tokens", 0)),
                    "prompt_tokens_mean": float(s.get("prompt_tokens", 0)),
                    "prompt_tokens_p95": float(s.get("prompt_tokens", 0)),
                    "gen_tokens_mean": float(s.get("max_output_tokens", 0)),
                    "tokens_per_sec_output": 0.0,
                    "tokens_per_sec_total": 0.0,
                    "ttft_ms_p50": 0.0,
                    "ttft_ms_p95": 0.0,
                    "ttft_ms_p99": 0.0,
                    "inter_token_latency_p50": 0.0,
                    "inter_token_latency_p95": 0.0,
                    "inter_token_latency_p99": 0.0,
                    "latency_ms_p50": 0.0,
                    "latency_ms_p95": 0.0,
                    "latency_ms_p99": 0.0,
                    "success_rate": 1.0,
                    "http_error_rate": 0.0,
                    "timeout_rate": 0.0,
                }
            )
    else:
        for s in scenarios:
            rows.append(await run_scenario(args.base_url, args.model, s))

    # Determinism probe against live endpoint.
    if args.dry_run:
        det = determinism_check(["dry-run-output", "dry-run-output"])
    else:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r1 = await one_request(client, args.base_url, args.model, 32, 32)
            r2 = await one_request(client, args.base_url, args.model, 32, 32)
        det = determinism_check([r1["output"], r2["output"]])
    if not det["passed"]:
        raise SystemExit("Determinism check failed: outputs differ for temperature=0")

    csv_path = out_dir / f"{args.run_id}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({
                "timestamp_utc": ts,
                "run_id": args.run_id,
                "system_under_test": args.system,
                "model_id": args.model,
                "gpu_type": args.gpu_type,
                "gpu_count": args.gpu_count,
                "precision": "fp16",
                **r,
                "gpu_util_pct_p50": 0,
                "gpu_util_pct_p95": 0,
                "gpu_mem_used_mb_p50": 0,
                "gpu_mem_used_mb_p95": 0,
                "est_dollars_per_million_output_tokens": 0,
            })

    manifest = {
        "run_id": args.run_id,
        "timestamp_utc": ts,
        "system_under_test": args.system,
        "model_id": args.model,
        "headline_vllm_version": args.vllm_version,
        "determinism": det,
        "rows": len(rows),
        "base_url": args.base_url,
    }
    manifest_path = out_dir / f"{args.run_id}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"wrote: {csv_path}")
    print(f"wrote: {manifest_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1/chat/completions")
    p.add_argument("--system", default="phase1")
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--gpu-type", default="unknown")
    p.add_argument("--gpu-count", type=int, default=1)
    p.add_argument("--vllm-version", default="0.8.5")
    p.add_argument("--scenarios", default="bench/scenarios/baseline.yaml")
    p.add_argument("--output-dir", default="bench/results")
    p.add_argument("--run-id", default=f"run-{dt.datetime.now(dt.UTC).strftime('%Y%m%d%H%M%S')}")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
