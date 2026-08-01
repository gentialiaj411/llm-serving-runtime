"""Phase-4 feature ablation matrix.

Toggles one optimization at a time against an all-off baseline on a fixed
scenario (bench/scenarios/ablation_matrix.yaml). Writes:

  - bench/results/ablation_matrix.json
  - bench/results/ablation_matrix.md
  - bench/results/ablation-<cell>.manifest.json  (schema-validated)

Reproduce:
  .venv311\\Scripts\\python.exe bench/scripts/ablation_matrix.py
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
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MODEL_ID = "Qwen/Qwen2-1.5B-Instruct"
AWQ_MODEL_ID = "Qwen/Qwen2-1.5B-Instruct-AWQ"
SCENARIO_FILE = "bench/scenarios/ablation_matrix.yaml"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _gpu_type() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            timeout=5,
        ).strip()
        return out.splitlines()[0].strip() if out else "unknown"
    except Exception:
        return "unknown"


def _build_shared_prompt(model_id: str, target_tokens: int) -> str:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    chunk = (
        "You are a helpful assistant for Orcaforge ablation benchmarks. "
        "Answer concisely and follow instructions. "
    )
    text = chunk
    while True:
        n = int(tokenizer(text, return_tensors="pt")["input_ids"].shape[1])
        if n >= target_tokens:
            break
        text += chunk
    ids = tokenizer(text, return_tensors="pt")["input_ids"][0, :target_tokens].tolist()
    return tokenizer.decode(ids, skip_special_tokens=True)


def _baseline_env() -> dict[str, str]:
    """All optimizations off.

    Uses KV=dynamic (not reserved) so continuous-batching can be toggled via
    PHASE2_MAX_ACTIVE alone — the worker only batch-decodes on dynamic/paged.
    """
    return {
        "PHASE2_BACKEND": "transformers",
        "HF_MODEL_ID": MODEL_ID,
        "HF_TORCH_DTYPE": "float16",
        "HF_DEVICE": "cuda",
        "PHASE2_MAX_ACTIVE": "1",
        "PHASE2_KV_BACKEND": "dynamic",
        "PHASE2_PREFIX_CACHE": "0",
        "PHASE2_SPECULATIVE": "0",
        "PHASE2_QUANT": "none",
        "PHASE2_CUDA_GRAPH": "0",
        "PHASE2_CUDA_GRAPH_TRY_HF": "0",
        "PHASE2_LORA": "0",
        "PHASE2_BATCH_DECODE_STEPS": "1",
        "PHASE2_DECODE_STEP_MS": "1",
        "KV_TOTAL_BLOCKS": "4096",
        "KV_BLOCK_SIZE_TOKENS": "16",
        "PYTHONUNBUFFERED": "1",
    }


FEATURE_OVERRIDES: dict[str, dict[str, str]] = {
    "baseline_all_off": {},
    "continuous_batching": {"PHASE2_MAX_ACTIVE": "8"},
    "paged_kv": {"PHASE2_KV_BACKEND": "paged"},
    "prefix_cache": {"PHASE2_PREFIX_CACHE": "1"},
    # Same-model draft keeps tokenizer compatible with the fixed Qwen scenario.
    "speculative_decoding": {
        "PHASE2_SPECULATIVE": "1",
        "PHASE2_SPEC_K": "4",
        "HF_DRAFT_MODEL_ID": MODEL_ID,
    },
    "int4_awq": {
        "PHASE2_QUANT": "int4",
        "HF_AWQ_MODEL_ID": AWQ_MODEL_ID,
        "HF_MODEL_ID": AWQ_MODEL_ID,
    },
    # CUDA-graph manager only attaches when KV=paged; enable paged so the flag
    # is the sole *extra* vs a pure paged cell (documented in notes).
    "cuda_graphs": {"PHASE2_CUDA_GRAPH": "1", "PHASE2_KV_BACKEND": "paged"},
}


def _start_worker(feature: str) -> tuple[subprocess.Popen[str], int, str | None]:
    port = _free_port()
    env = os.environ.copy()
    env.pop("HF_MODEL_ID", None)
    env.update(_baseline_env())
    env.update(FEATURE_OVERRIDES[feature])
    stderr_path = ROOT / "bench" / "results" / f"_ablation_{feature}.stderr.log"
    stderr_f = open(stderr_path, "w", encoding="utf-8")
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
        stderr=stderr_f,
        text=True,
        env=env,
    )
    health = f"http://127.0.0.1:{port}/healthz"
    deadline = time.time() + 300.0
    while time.time() < deadline:
        try:
            if httpx.get(health, timeout=2.0).status_code == 200:
                break
        except Exception:
            pass
        if proc.poll() is not None:
            stderr_f.flush()
            stderr_f.close()
            err = stderr_path.read_text(encoding="utf-8", errors="replace")[-2000:]
            return proc, port, f"worker exited during startup: {err}"
        time.sleep(0.5)
    else:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        stderr_f.flush()
        stderr_f.close()
        err = stderr_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        return proc, port, f"worker health timeout: {err}"

    # healthz does not load the model; force a one-token warmup so load failures
    # surface as startup errors rather than void timed runs.
    async def _warmup() -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=None) as client:
            return await _one_stream(
                client,
                f"http://127.0.0.1:{port}",
                f"ablation-warmup-{time.time_ns()}",
                "warmup",
                1,
            )

    try:
        warmup = asyncio.run(_warmup())
    except Exception as exc:
        warmup = {"status": "error", "error": str(exc), "output_tokens": 0}
    if warmup.get("status") != "completed":
        err = str(warmup.get("error") or warmup.get("status") or "warmup failed")
        tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-1500:]
        _stop_worker(proc)
        stderr_f.flush()
        stderr_f.close()
        return proc, port, f"model warmup failed: {err}\n{tail}"
    stderr_f.flush()
    return proc, port, None


def _stop_worker(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


async def _one_stream(
    client: httpx.AsyncClient,
    base_url: str,
    request_id: str,
    prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    output_tokens = 0
    status = "error"
    err = ""
    try:
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
                elif kind in {"cancelled", "timed_out", "error", "duplicate"}:
                    status = str(kind)
                    err = str(event.get("error") or event.get("message") or kind)
                    break
    except Exception as exc:
        status = "error"
        err = str(exc)
    return {
        "request_id": request_id,
        "status": status,
        "output_tokens": output_tokens,
        "error": err,
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
                    f"ablation-{i}-{time.time_ns()}",
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
        "tokens_per_sec_output": output_tokens / elapsed,
        "duration_s": elapsed,
        "sample_errors": [r["error"] for r in results if r["error"]][:3],
    }


def _write_manifest(cell: str, metrics: dict[str, Any], ts: str, gpu: str) -> Path:
    run_id = f"ablation-{cell}"
    path = ROOT / "bench" / "results" / f"{run_id}.manifest.json"
    void = float(metrics.get("success_rate", 0.0)) < 0.99
    payload = {
        "run_id": run_id,
        "timestamp_utc": ts,
        "system_under_test": "phase2",
        "model_id": MODEL_ID if cell != "int4_awq" else AWQ_MODEL_ID,
        "gpu_type": gpu,
        "gpu_count": 1,
        "scenario_file": SCENARIO_FILE,
        "scenario_id": "ablation_shared_prefix",
        "feature_cell": cell,
        "feature_env": {**_baseline_env(), **FEATURE_OVERRIDES[cell]},
        "rows": 1,
        "inference_mode": "real_model_inference",
        "tokens_per_sec_output": metrics.get("tokens_per_sec_output"),
        "success_rate": metrics.get("success_rate"),
        "throughput_void_lt_99pct_success": void,
        "error": metrics.get("error"),
        "fairness_notes": {
            "ablation": "one feature on vs all-off baseline; shared-prefix prompts",
            "harness": "bench/harness/run.py scenario schema + manifest.schema.json",
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _run_cell(
    feature: str,
    system_prompt: str,
    request_count: int,
    max_tokens: int,
    concurrency: int,
) -> dict[str, Any]:
    print(f"==> cell {feature}", flush=True)
    proc, port, start_err = _start_worker(feature)
    if start_err:
        _stop_worker(proc)
        return {
            "feature": feature,
            "status": "failed_startup",
            "error": start_err,
            "success_rate": 0.0,
            "tokens_per_sec_output": 0.0,
            "request_count": request_count,
            "concurrency": concurrency,
            "completed_requests": 0,
            "output_tokens": 0,
            "duration_s": 0.0,
        }
    base_url = f"http://127.0.0.1:{port}"
    try:
        metrics = asyncio.run(
            _run_workload(base_url, system_prompt, request_count, max_tokens, concurrency)
        )
        try:
            worker_metrics = httpx.get(f"{base_url}/metrics", timeout=10.0).json()
        except Exception:
            worker_metrics = {}
        metrics["worker_metrics_subset"] = {
            k: worker_metrics.get(k)
            for k in (
                "prefix_cache_hit_rate",
                "prefix_cache_hits",
                "prefix_cache_misses",
                "speculative_acceptance_rate",
                "continuous_batches_total",
                "kv_backend",
            )
            if k in worker_metrics or True
        }
        metrics["feature"] = feature
        metrics["status"] = "ok" if metrics["success_rate"] >= 0.99 else "low_success"
        return metrics
    except Exception as exc:
        return {
            "feature": feature,
            "status": "failed_run",
            "error": str(exc),
            "success_rate": 0.0,
            "tokens_per_sec_output": 0.0,
            "request_count": request_count,
            "concurrency": concurrency,
            "completed_requests": 0,
            "output_tokens": 0,
            "duration_s": 0.0,
        }
    finally:
        _stop_worker(proc)


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    base = payload["cells"]["baseline_all_off"]
    base_tps = float(base.get("tokens_per_sec_output") or 0.0)
    lines = [
        "# Feature ablation matrix",
        "",
        f"- Model: `{payload['model_id']}`",
        f"- GPU: `{payload['gpu_type']}`",
        f"- Scenario: `{payload['scenario_file']}` (`{payload['scenario_id']}`)",
        f"- Shared prefix tokens: `{payload['shared_prefix_tokens']}`",
        f"- Requests: `{payload['request_count']}`, concurrency `{payload['concurrency']}`, "
        f"decode `{payload['max_output_tokens']}` tokens",
        f"- Generated: `{payload['timestamp_utc']}`",
        "",
        "One feature enabled at a time vs all-off baseline. "
        "Throughput at success_rate < 0.99 is void.",
        "",
        "| Feature | tok/s | Δ vs baseline | success | status | notes |",
        "|---------|------:|--------------:|--------:|--------|-------|",
    ]
    for name in payload["feature_order"]:
        cell = payload["cells"][name]
        tps = float(cell.get("tokens_per_sec_output") or 0.0)
        success = float(cell.get("success_rate") or 0.0)
        void = success < 0.99
        if name == "baseline_all_off":
            delta_s = "—"
        elif void or base_tps <= 0:
            delta_s = "void" if void else "n/a"
        else:
            delta_pct = 100.0 * (tps / base_tps - 1.0)
            delta_s = f"{delta_pct:+.1f}%"
        note = cell.get("error") or ""
        if not note and name == "prefix_cache":
            wm = cell.get("worker_metrics_subset") or {}
            note = f"hits={wm.get('prefix_cache_hits')} miss={wm.get('prefix_cache_misses')}"
        if not note and name == "cuda_graphs":
            dvp = payload.get("cuda_graphs_delta_vs_paged_kv_pct")
            note = f"vs paged_kv: {dvp:+.1f}%" if dvp is not None else "vs paged_kv: n/a"
        if not note and name == "continuous_batching":
            wm = cell.get("worker_metrics_subset") or {}
            note = f"continuous_batches_total={wm.get('continuous_batches_total')}"
        if isinstance(note, str) and len(note) > 80:
            note = note[:77] + "..."
        tps_s = f"{tps:.2f}" if not void else f"{tps:.2f} (void)"
        lines.append(
            f"| `{name}` | {tps_s} | {delta_s} | {success:.2f} | {cell.get('status')} | {note} |"
        )
    lines.extend(
        [
            "",
            "Per-cell manifests: `bench/results/ablation-<feature>.manifest.json`",
            "Aggregate: `bench/results/ablation_matrix.json`",
            "",
            "Validate: `python bench/scripts/validate_manifests.py`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=24)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-output-tokens", type=int, default=32)
    parser.add_argument("--shared-prefix-tokens", type=int, default=256)
    parser.add_argument(
        "--features",
        default=",".join(FEATURE_OVERRIDES.keys()),
        help="Comma-separated feature cells to run",
    )
    parser.add_argument("--output-json", default="bench/results/ablation_matrix.json")
    parser.add_argument("--output-md", default="bench/results/ablation_matrix.md")
    args = parser.parse_args()

    scenario_path = ROOT / SCENARIO_FILE
    scenario = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))["scenarios"][0]
    # Prefer CLI; fall back to scenario file.
    concurrency = args.concurrency or int(scenario["concurrency"][0])
    max_out = args.max_output_tokens or int(scenario["max_output_tokens"])
    prefix_toks = args.shared_prefix_tokens or int(scenario["prompt_tokens"])

    features = [f.strip() for f in args.features.split(",") if f.strip()]
    for f in features:
        if f not in FEATURE_OVERRIDES:
            raise SystemExit(f"unknown feature cell: {f}")

    print("Building shared prefix prompt...", flush=True)
    system_prompt = _build_shared_prompt(MODEL_ID, prefix_toks)
    ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    gpu = _gpu_type()

    cells: dict[str, Any] = {}
    for feature in features:
        cell = _run_cell(feature, system_prompt, args.requests, max_out, concurrency)
        cells[feature] = cell
        _write_manifest(feature, cell, ts, gpu)
        print(
            f"    {feature}: tok/s={cell.get('tokens_per_sec_output', 0):.2f} "
            f"success={cell.get('success_rate', 0):.2f} status={cell.get('status')}",
            flush=True,
        )

    base = cells.get("baseline_all_off", {})
    base_tps = float(base.get("tokens_per_sec_output") or 0.0)
    paged_tps = float(cells.get("paged_kv", {}).get("tokens_per_sec_output") or 0.0)
    deltas: dict[str, Any] = {}
    for name, cell in cells.items():
        tps = float(cell.get("tokens_per_sec_output") or 0.0)
        success = float(cell.get("success_rate") or 0.0)
        void = success < 0.99
        if name == "baseline_all_off":
            deltas[name] = {"delta_pct": 0.0, "void": void}
        elif void or base_tps <= 0:
            deltas[name] = {"delta_pct": None, "void": True}
        else:
            deltas[name] = {"delta_pct": 100.0 * (tps / base_tps - 1.0), "void": False}

    cuda_vs_paged = None
    cuda_cell = cells.get("cuda_graphs", {})
    if (
        paged_tps > 0
        and float(cuda_cell.get("success_rate") or 0.0) >= 0.99
        and float(cells.get("paged_kv", {}).get("success_rate") or 0.0) >= 0.99
    ):
        cuda_vs_paged = 100.0 * (
            float(cuda_cell.get("tokens_per_sec_output") or 0.0) / paged_tps - 1.0
        )

    payload = {
        "artifact_type": "feature_ablation_matrix",
        "timestamp_utc": ts,
        "model_id": MODEL_ID,
        "awq_model_id": AWQ_MODEL_ID,
        "gpu_type": gpu,
        "scenario_file": SCENARIO_FILE,
        "scenario_id": scenario["id"],
        "shared_prefix_tokens": prefix_toks,
        "request_count": args.requests,
        "concurrency": concurrency,
        "max_output_tokens": max_out,
        "measurement": "measured_on_gpu",
        "feature_order": features,
        "cells": cells,
        "delta_vs_baseline_pct": deltas,
        "cuda_graphs_delta_vs_paged_kv_pct": cuda_vs_paged,
        "baseline_tokens_per_sec_output": base_tps,
        "notes": [
            "All-off baseline: PHASE2_MAX_ACTIVE=1, KV=dynamic, prefix/spec/quant/cuda_graph off.",
            "continuous_batching enables PHASE2_MAX_ACTIVE=8 (worker batch-decode path requires dynamic/paged).",
            "cuda_graphs sets PHASE2_CUDA_GRAPH=1 and KV=paged (manager only attaches on paged); compare vs paged_kv cell for graph delta.",
            "speculative_decoding uses the same Qwen checkpoint as draft for tokenizer compatibility.",
            "Throughput with success_rate < 0.99 is void per campaign rules.",
            "prefix_cache cell reported 0 hits under this fixed scenario (see worker_metrics_subset); do not claim a prefix win from this matrix.",
        ],
    }

    out_json = ROOT / args.output_json
    out_md = ROOT / args.output_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_md(out_md, payload)
    print(f"wrote: {out_json}")
    print(f"wrote: {out_md}")


if __name__ == "__main__":
    main()
