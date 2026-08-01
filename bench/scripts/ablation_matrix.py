"""Phase-4 feature ablation matrix (corrected).

Table A — each feature in the config where it is *meaningful*
  (paged/graphs require max_active=8; speculative uses a smaller draft).

Table B — cumulative ladder (how a serving stack is actually built):
  baseline -> +CB(8) -> +paged -> +prefix -> +graphs

Outputs:
  - bench/results/ablation_matrix.json
  - bench/results/ablation_matrix.md
  - bench/results/ablation-<cell>.manifest.json

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
DRAFT_MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"
AWQ_MODEL_ID = "Qwen/Qwen2-1.5B-Instruct-AWQ"
SCENARIO_FILE = "bench/scenarios/ablation_matrix.yaml"

# Table A: isolation controls in configs where each feature is meaningful.
TABLE_A: dict[str, dict[str, str]] = {
    "baseline_all_off": {},
    "continuous_batching": {"PHASE2_MAX_ACTIVE": "8"},
    "paged_kv": {"PHASE2_MAX_ACTIVE": "8", "PHASE2_KV_BACKEND": "paged"},
    "prefix_cache": {"PHASE2_MAX_ACTIVE": "8", "PHASE2_PREFIX_CACHE": "1"},
    "speculative_decoding": {
        "PHASE2_SPECULATIVE": "1",
        "PHASE2_SPEC_K": "4",
        "HF_DRAFT_MODEL_ID": DRAFT_MODEL_ID,
    },
    "int4_awq": {
        "PHASE2_QUANT": "int4",
        "HF_AWQ_MODEL_ID": AWQ_MODEL_ID,
        "HF_MODEL_ID": AWQ_MODEL_ID,
    },
    "cuda_graphs": {
        "PHASE2_MAX_ACTIVE": "8",
        "PHASE2_KV_BACKEND": "paged",
        "PHASE2_CUDA_GRAPH": "1",
    },
}

# Table B: cumulative ladder (publish this story).
TABLE_B: dict[str, dict[str, str]] = {
    "ladder_baseline": {},
    "ladder_plus_cb": {"PHASE2_MAX_ACTIVE": "8"},
    "ladder_plus_paged": {"PHASE2_MAX_ACTIVE": "8", "PHASE2_KV_BACKEND": "paged"},
    "ladder_plus_prefix": {
        "PHASE2_MAX_ACTIVE": "8",
        "PHASE2_KV_BACKEND": "paged",
        "PHASE2_PREFIX_CACHE": "1",
    },
    "ladder_plus_graphs": {
        "PHASE2_MAX_ACTIVE": "8",
        "PHASE2_KV_BACKEND": "paged",
        "PHASE2_PREFIX_CACHE": "1",
        "PHASE2_CUDA_GRAPH": "1",
    },
}

ALL_CELLS = {**TABLE_A, **TABLE_B}


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
        "PREFIX_CACHE_MAX_ENTRIES": "512",
        "PYTHONUNBUFFERED": "1",
    }


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


def _start_worker(cell: str, overrides: dict[str, str]) -> tuple[subprocess.Popen[str], int, str | None]:
    port = _free_port()
    env = os.environ.copy()
    env.pop("HF_MODEL_ID", None)
    env.update(_baseline_env())
    env.update(overrides)
    stderr_path = ROOT / "bench" / "results" / f"_ablation_{cell}.stderr.log"
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
    deadline = time.time() + 360.0
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


def _write_manifest(
    cell: str,
    overrides: dict[str, str],
    metrics: dict[str, Any],
    ts: str,
    gpu: str,
    table: str,
) -> Path:
    run_id = f"ablation-{cell}"
    path = ROOT / "bench" / "results" / f"{run_id}.manifest.json"
    void = float(metrics.get("success_rate", 0.0)) < 0.99
    model = AWQ_MODEL_ID if cell == "int4_awq" else MODEL_ID
    payload = {
        "run_id": run_id,
        "timestamp_utc": ts,
        "system_under_test": "phase2",
        "model_id": model,
        "gpu_type": gpu,
        "gpu_count": 1,
        "scenario_file": SCENARIO_FILE,
        "scenario_id": "ablation_shared_prefix",
        "feature_cell": cell,
        "ablation_table": table,
        "feature_env": {**_baseline_env(), **overrides},
        "rows": 1,
        "inference_mode": "real_model_inference",
        "tokens_per_sec_output": metrics.get("tokens_per_sec_output"),
        "success_rate": metrics.get("success_rate"),
        "throughput_void_lt_99pct_success": void,
        "error": metrics.get("error"),
        "fairness_notes": {
            "ablation": (
                "Table A: feature in meaningful config; "
                "Table B: cumulative ladder. Paged/graphs require max_active>=batch."
            ),
            "phase2_proxy_reference": "bench/results/paged_kv_launch_profile.meta.json B8 ~91.8 tok/s proxy",
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _run_cell(
    cell: str,
    overrides: dict[str, str],
    system_prompt: str,
    request_count: int,
    max_tokens: int,
    concurrency: int,
) -> dict[str, Any]:
    print(f"==> cell {cell}", flush=True)
    proc, port, start_err = _start_worker(cell, overrides)
    if start_err:
        _stop_worker(proc)
        return {
            "feature": cell,
            "status": "failed_startup",
            "error": start_err,
            "success_rate": 0.0,
            "tokens_per_sec_output": 0.0,
            "request_count": request_count,
            "concurrency": concurrency,
            "completed_requests": 0,
            "output_tokens": 0,
            "duration_s": 0.0,
            "env_overrides": overrides,
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
                "prefix_cache_inserts",
                "speculative_acceptance_rate",
                "continuous_batches_total",
                "kv_backend",
                "max_batch_size",
                "peak_active_requests",
            )
        }
        metrics["feature"] = cell
        metrics["env_overrides"] = overrides
        metrics["status"] = "ok" if metrics["success_rate"] >= 0.99 else "low_success"
        return metrics
    except Exception as exc:
        return {
            "feature": cell,
            "status": "failed_run",
            "error": str(exc),
            "success_rate": 0.0,
            "tokens_per_sec_output": 0.0,
            "request_count": request_count,
            "concurrency": concurrency,
            "completed_requests": 0,
            "output_tokens": 0,
            "duration_s": 0.0,
            "env_overrides": overrides,
        }
    finally:
        _stop_worker(proc)


def _delta_map(cells: dict[str, Any], baseline_key: str) -> dict[str, Any]:
    base_tps = float(cells.get(baseline_key, {}).get("tokens_per_sec_output") or 0.0)
    out: dict[str, Any] = {}
    for name, cell in cells.items():
        tps = float(cell.get("tokens_per_sec_output") or 0.0)
        success = float(cell.get("success_rate") or 0.0)
        void = success < 0.99
        if name == baseline_key:
            out[name] = {"delta_pct": 0.0, "void": void, "tokens_per_sec_output": tps}
        elif void or base_tps <= 0:
            out[name] = {"delta_pct": None, "void": True, "tokens_per_sec_output": tps}
        else:
            out[name] = {
                "delta_pct": 100.0 * (tps / base_tps - 1.0),
                "void": False,
                "tokens_per_sec_output": tps,
                "ratio_vs_baseline": tps / base_tps,
            }
    return out


def _ladder_step_deltas(order: list[str], cells: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prev_name: str | None = None
    prev_tps: float | None = None
    for name in order:
        cell = cells[name]
        tps = float(cell.get("tokens_per_sec_output") or 0.0)
        success = float(cell.get("success_rate") or 0.0)
        void = success < 0.99
        step = None
        if prev_tps is not None and prev_tps > 0 and not void:
            step = 100.0 * (tps / prev_tps - 1.0)
        rows.append(
            {
                "cell": name,
                "tokens_per_sec_output": tps,
                "success_rate": success,
                "void": void,
                "delta_vs_previous_pct": step,
                "previous": prev_name,
            }
        )
        prev_name = name
        prev_tps = tps if not void else prev_tps
    return rows


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Feature ablation matrix (corrected)",
        "",
        f"- Model: `{payload['model_id']}`",
        f"- Draft (speculative): `{payload.get('draft_model_id')}`",
        f"- GPU: `{payload['gpu_type']}`",
        f"- Scenario: `{payload['scenario_file']}` (`{payload['scenario_id']}`)",
        f"- Shared prefix tokens: `{payload['shared_prefix_tokens']}`",
        f"- Requests: `{payload['request_count']}`, concurrency `{payload['concurrency']}`, "
        f"decode `{payload['max_output_tokens']}` tokens",
        f"- Generated: `{payload['timestamp_utc']}`",
        "",
        "Throughput at success_rate < 0.99 is void.",
        "",
        "## Table B — cumulative ladder (publish this)",
        "",
        "| Step | tok/s | Δ vs previous | Δ vs baseline | success | notes |",
        "|------|------:|--------------:|--------------:|--------:|-------|",
    ]
    base_tps = float(payload["table_b"]["baseline_tokens_per_sec_output"] or 0.0)
    for row in payload["table_b"]["step_deltas"]:
        name = row["cell"]
        cell = payload["table_b"]["cells"][name]
        tps = float(row["tokens_per_sec_output"])
        success = float(row["success_rate"])
        void = bool(row["void"])
        d_prev = row["delta_vs_previous_pct"]
        d_prev_s = "—" if d_prev is None else f"{d_prev:+.1f}%"
        if name.endswith("baseline") or void or base_tps <= 0:
            d_base_s = "—" if not void else "void"
        else:
            d_base_s = f"{100.0 * (tps / base_tps - 1.0):+.1f}%"
        wm = cell.get("worker_metrics_subset") or {}
        note = cell.get("error") or ""
        if "prefix" in name and not note:
            note = f"hits={wm.get('prefix_cache_hits')} miss={wm.get('prefix_cache_misses')}"
        if "paged" in name and not note:
            note = f"max_batch={wm.get('max_batch_size')} peak_active={wm.get('peak_active_requests')}"
        if isinstance(note, str) and len(note) > 70:
            note = note[:67] + "..."
        tps_s = f"{tps:.2f}" if not void else f"{tps:.2f} (void)"
        lines.append(
            f"| `{name}` | {tps_s} | {d_prev_s} | {d_base_s} | {success:.2f} | {note} |"
        )

    lines.extend(
        [
            "",
            "## Table A — meaningful one-at-a-time (isolation control)",
            "",
            "Caveat: paged KV / CUDA graphs are measured at `PHASE2_MAX_ACTIVE=8` "
            "because batching is required for those features to be meaningful.",
            "",
            "| Feature | tok/s | Δ vs baseline | success | status | notes |",
            "|---------|------:|--------------:|--------:|--------|-------|",
        ]
    )
    for name in payload["table_a"]["feature_order"]:
        cell = payload["table_a"]["cells"][name]
        tps = float(cell.get("tokens_per_sec_output") or 0.0)
        success = float(cell.get("success_rate") or 0.0)
        void = success < 0.99
        d = payload["table_a"]["delta_vs_baseline_pct"].get(name, {})
        if name == "baseline_all_off":
            delta_s = "—"
        elif d.get("void") or d.get("delta_pct") is None:
            delta_s = "void"
        else:
            delta_s = f"{float(d['delta_pct']):+.1f}%"
        note = cell.get("error") or ""
        wm = cell.get("worker_metrics_subset") or {}
        if name == "prefix_cache" and not note:
            note = f"hits={wm.get('prefix_cache_hits')} miss={wm.get('prefix_cache_misses')}"
        if name == "paged_kv" and not note:
            note = f"max_batch={wm.get('max_batch_size')} peak_active={wm.get('peak_active_requests')}"
        if name == "cuda_graphs" and not note:
            dvp = payload["table_a"].get("cuda_graphs_delta_vs_paged_kv_pct")
            note = f"vs paged_kv: {dvp:+.1f}%" if dvp is not None else "vs paged_kv: n/a"
        if name == "speculative_decoding" and not note:
            note = f"accept={wm.get('speculative_acceptance_rate')}"
        if isinstance(note, str) and len(note) > 70:
            note = note[:67] + "..."
        tps_s = f"{tps:.2f}" if not void else f"{tps:.2f} (void)"
        lines.append(
            f"| `{name}` | {tps_s} | {delta_s} | {success:.2f} | {cell.get('status')} | {note} |"
        )

    phase2 = payload.get("phase2_cross_check") or {}
    lines.extend(
        [
            "",
            "## Phase 2 cross-check",
            "",
            f"- Phase 2 batched paged B=8 proxy: `{phase2.get('phase2_b8_tok_per_s_proxy')}` "
            f"(`paged_kv_launch_profile.meta.json`)",
            f"- Table A `paged_kv` harness tok/s: `{phase2.get('table_a_paged_kv_tok_per_s')}`",
            f"- Ratio harness/proxy: `{phase2.get('harness_over_proxy_ratio')}`",
            f"- Verdict: `{phase2.get('verdict')}`",
            "",
            "Per-cell manifests: `bench/results/ablation-<feature>.manifest.json`",
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
        "--tables",
        default="A,B",
        help="Comma-separated: A (isolation), B (ladder), or both",
    )
    parser.add_argument(
        "--features",
        default="",
        help="Optional explicit cell list (overrides --tables)",
    )
    parser.add_argument("--output-json", default="bench/results/ablation_matrix.json")
    parser.add_argument("--output-md", default="bench/results/ablation_matrix.md")
    args = parser.parse_args()

    scenario = yaml.safe_load((ROOT / SCENARIO_FILE).read_text(encoding="utf-8"))["scenarios"][0]
    concurrency = args.concurrency or int(scenario["concurrency"][0])
    max_out = args.max_output_tokens or int(scenario["max_output_tokens"])
    prefix_toks = args.shared_prefix_tokens or int(scenario["prompt_tokens"])

    if args.features.strip():
        selected = [f.strip() for f in args.features.split(",") if f.strip()]
    else:
        selected = []
        tables = {t.strip().upper() for t in args.tables.split(",") if t.strip()}
        if "A" in tables:
            selected.extend(TABLE_A.keys())
        if "B" in tables:
            selected.extend(TABLE_B.keys())
    for f in selected:
        if f not in ALL_CELLS:
            raise SystemExit(f"unknown cell: {f}")

    print("Building shared prefix prompt...", flush=True)
    system_prompt = _build_shared_prompt(MODEL_ID, prefix_toks)
    ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    gpu = _gpu_type()

    cells: dict[str, Any] = {}
    for cell in selected:
        overrides = ALL_CELLS[cell]
        table = "B" if cell in TABLE_B else "A"
        result = _run_cell(cell, overrides, system_prompt, args.requests, max_out, concurrency)
        cells[cell] = result
        _write_manifest(cell, overrides, result, ts, gpu, table)
        print(
            f"    {cell}: tok/s={float(result.get('tokens_per_sec_output') or 0):.2f} "
            f"success={float(result.get('success_rate') or 0):.2f} status={result.get('status')}",
            flush=True,
        )

    table_a_cells = {k: cells[k] for k in TABLE_A if k in cells}
    table_b_cells = {k: cells[k] for k in TABLE_B if k in cells}
    table_a_order = [k for k in TABLE_A if k in cells]
    table_b_order = [k for k in TABLE_B if k in cells]

    a_deltas = _delta_map(table_a_cells, "baseline_all_off") if table_a_cells else {}
    b_base = float(table_b_cells.get("ladder_baseline", {}).get("tokens_per_sec_output") or 0.0)
    b_steps = _ladder_step_deltas(table_b_order, table_b_cells) if table_b_cells else []

    paged_tps = float(table_a_cells.get("paged_kv", {}).get("tokens_per_sec_output") or 0.0)
    cuda_tps = float(table_a_cells.get("cuda_graphs", {}).get("tokens_per_sec_output") or 0.0)
    cuda_vs_paged = None
    if (
        paged_tps > 0
        and float(table_a_cells.get("cuda_graphs", {}).get("success_rate") or 0) >= 0.99
        and float(table_a_cells.get("paged_kv", {}).get("success_rate") or 0) >= 0.99
    ):
        cuda_vs_paged = 100.0 * (cuda_tps / paged_tps - 1.0)

    phase2_proxy = 91.80450932044542
    try:
        meta = json.loads(
            (ROOT / "bench/results/paged_kv_launch_profile.meta.json").read_text(encoding="utf-8")
        )
        phase2_proxy = float(meta["scaling"]["paged"]["b8_tok_per_s_proxy"])
    except Exception:
        pass

    harness_over_proxy = (paged_tps / phase2_proxy) if phase2_proxy > 0 and paged_tps > 0 else None
    if paged_tps <= 0:
        verdict = "paged_kv cell missing or failed"
    elif float(table_a_cells.get("paged_kv", {}).get("success_rate") or 0) < 0.99:
        verdict = "paged_kv throughput void (<99% success)"
    elif harness_over_proxy is not None and harness_over_proxy >= 0.5:
        verdict = "harness paged@ma=8 is in the same ballpark as Phase 2 proxy (not a pure overhead tax)"
    else:
        verdict = "harness paged@ma=8 far below Phase 2 proxy — investigate before README claims"

    payload = {
        "artifact_type": "feature_ablation_matrix_corrected",
        "timestamp_utc": ts,
        "model_id": MODEL_ID,
        "draft_model_id": DRAFT_MODEL_ID,
        "awq_model_id": AWQ_MODEL_ID,
        "gpu_type": gpu,
        "scenario_file": SCENARIO_FILE,
        "scenario_id": scenario["id"],
        "shared_prefix_tokens": prefix_toks,
        "request_count": args.requests,
        "concurrency": concurrency,
        "max_output_tokens": max_out,
        "measurement": "measured_on_gpu",
        "methodology": {
            "table_a": "Feature in meaningful config (paged/graphs at max_active=8)",
            "table_b": "Cumulative ladder: baseline -> +CB -> +paged -> +prefix -> +graphs",
            "prior_misconfig": (
                "First Phase-4 run measured paged/graphs at max_active=1 and used "
                "same-model draft; those cells are superseded by this artifact."
            ),
        },
        "table_a": {
            "feature_order": table_a_order,
            "cells": table_a_cells,
            "delta_vs_baseline_pct": a_deltas,
            "baseline_tokens_per_sec_output": float(
                table_a_cells.get("baseline_all_off", {}).get("tokens_per_sec_output") or 0.0
            ),
            "cuda_graphs_delta_vs_paged_kv_pct": cuda_vs_paged,
        },
        "table_b": {
            "feature_order": table_b_order,
            "cells": table_b_cells,
            "step_deltas": b_steps,
            "baseline_tokens_per_sec_output": b_base,
        },
        "phase2_cross_check": {
            "phase2_b8_tok_per_s_proxy": phase2_proxy,
            "table_a_paged_kv_tok_per_s": paged_tps,
            "harness_over_proxy_ratio": harness_over_proxy,
            "verdict": verdict,
        },
        "notes": [
            "Do not cite the superseded max_active=1 paged/graph cells from the first Phase-4 commit.",
            "Prefix cache registers intermediate trie nodes; batch prefill paths now call insert.",
            "Table A prefix_cache (dynamic+ma=8) is the valid prefix measurement (hits>0).",
            "Table B ladder_plus_prefix/graphs are VOID: prefix hit restores DynamicCache into paged path (BlockPagedCache required).",
            "Phase 2 B=8 proxy (~91.8 tok/s) does NOT reproduce in this harness (~23 tok/s paged@ma=8); do not README a 3.8x claim.",
            "INT4 remains void on .venv311 without autoawq.",
            "cuda_graphs vs paged delta is not an HF-capture win (PHASE2_CUDA_GRAPH_TRY_HF=0); treat as noise unless capture lands.",
        ],
    }

    out_json = ROOT / args.output_json
    out_md = ROOT / args.output_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_md(out_md, payload)
    print(f"wrote: {out_json}")
    print(f"wrote: {out_md}")
    print(f"phase2 cross-check: {verdict}", flush=True)


if __name__ == "__main__":
    main()
