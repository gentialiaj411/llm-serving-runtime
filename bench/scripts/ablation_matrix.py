"""Stable Phase-4 ablation measurement (instrument fix).

Protocol (do not claim deltas without this):
  - Decode-heavy scenario: 128 prompt / 512 output (not 256/32).
  - >=5 repeats per feature cell.
  - Fresh baseline measured immediately before every feature cell.
  - Randomized feature order each repeat.
  - Explicit warmup discard (not timed).
  - nvidia-smi clocks + temperature logged per measurement.
  - Delta reported only if |median delta| exceeds baseline run-to-run IQR;
    otherwise \"within noise\".

Outputs:
  - bench/results/ablation_matrix.json
  - bench/results/ablation_matrix.md
  - bench/results/ablation-stable-*.manifest.json

Reproduce:
  .venv311\\Scripts\\python.exe bench/scripts/ablation_matrix.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import socket
import statistics
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
SCENARIO_FILE = "bench/scenarios/ablation_decode_heavy.yaml"

FEATURE_CELLS: dict[str, dict[str, str]] = {
    "continuous_batching": {"PHASE2_MAX_ACTIVE": "8"},
    "paged_kv": {"PHASE2_MAX_ACTIVE": "8", "PHASE2_KV_BACKEND": "paged"},
    "prefix_cache": {"PHASE2_MAX_ACTIVE": "8", "PHASE2_PREFIX_CACHE": "1"},
}


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


def _nvidia_smi_snapshot() -> dict[str, Any]:
    """Clocks + temperature for thermal confound detection."""
    query = (
        "timestamp,temperature.gpu,clocks.sm,clocks.mem,clocks.gr,"
        "utilization.gpu,utilization.memory,power.draw,clocks_throttle_reasons.active"
    )
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        ).strip()
        if not out:
            return {"ok": False, "error": "empty nvidia-smi"}
        parts = [p.strip() for p in out.splitlines()[0].split(",")]
        keys = [
            "timestamp",
            "temperature_c",
            "clock_sm_mhz",
            "clock_mem_mhz",
            "clock_gr_mhz",
            "util_gpu_pct",
            "util_mem_pct",
            "power_w",
            "throttle_reasons",
        ]
        snap: dict[str, Any] = {"ok": True}
        for k, v in zip(keys, parts):
            if k in {"timestamp", "throttle_reasons"}:
                snap[k] = v
            else:
                try:
                    snap[k] = float(v)
                except ValueError:
                    snap[k] = v
        return snap
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


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
        "KV_TOTAL_BLOCKS": "8192",
        "KV_BLOCK_SIZE_TOKENS": "16",
        "PREFIX_CACHE_MAX_ENTRIES": "512",
        "PYTHONUNBUFFERED": "1",
    }


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "median": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0, "stdev": 0.0, "iqr": 0.0}
    xs = sorted(values)

    def _pct(p: float) -> float:
        if len(xs) == 1:
            return float(xs[0])
        pos = (len(xs) - 1) * p
        lo = int(pos)
        hi = min(lo + 1, len(xs) - 1)
        w = pos - lo
        return float(xs[lo] * (1.0 - w) + xs[hi] * w)

    q1 = _pct(0.25)
    q3 = _pct(0.75)
    return {
        "n": float(len(xs)),
        "median": float(statistics.median(xs)),
        "mean": float(statistics.mean(xs)),
        "min": float(min(xs)),
        "max": float(max(xs)),
        "stdev": float(statistics.stdev(xs)) if len(xs) > 1 else 0.0,
        "iqr": float(q3 - q1),
    }


def _build_fixed_length_prompts(model_id: str, prompt_tokens: int, request_count: int) -> list[str]:
    """Shared-prefix prompts padded to *exact* token length so prefill can batch."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    chunk = (
        "You are a helpful assistant for Orcaforge stable ablation. "
        "Answer concisely. "
    )
    shared = chunk
    while len(tokenizer(shared)["input_ids"]) < max(32, prompt_tokens // 2):
        shared += chunk
    shared_ids = tokenizer(shared)["input_ids"][: max(32, prompt_tokens // 2)]
    shared_text = tokenizer.decode(shared_ids, skip_special_tokens=True)

    prompts: list[str] = []
    for i in range(request_count):
        suffix = f" Q{i}: explain topic {i % 17} briefly."
        # Grow/shrink pad so total tokenized length == prompt_tokens.
        pad = ""
        body = shared_text + suffix
        ids = tokenizer(body)["input_ids"]
        guard = 0
        while len(ids) < prompt_tokens and guard < 10000:
            pad += " pad"
            ids = tokenizer(body + pad)["input_ids"]
            guard += 1
        while len(ids) > prompt_tokens and pad:
            pad = pad[:-4] if len(pad) >= 4 else ""
            ids = tokenizer(body + pad)["input_ids"]
        if len(ids) != prompt_tokens:
            # Final hard trim/pad via ids.
            if len(ids) > prompt_tokens:
                ids = ids[:prompt_tokens]
            else:
                pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
                ids = ids + [int(pad_id)] * (prompt_tokens - len(ids))
            text = tokenizer.decode(ids, skip_special_tokens=False)
        else:
            text = body + pad
        # Verify length after decode/re-encode as worker will.
        check = tokenizer(text)["input_ids"]
        if len(check) != prompt_tokens:
            # Force exact ids round-trip failure → use decode of exact ids only.
            text = tokenizer.decode(ids[:prompt_tokens], skip_special_tokens=False)
        prompts.append(text)
    return prompts


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
            timeout=1200.0,
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
    return {"status": status, "output_tokens": output_tokens, "error": err}


async def _run_timed_workload(
    base_url: str,
    prompts: list[str],
    max_tokens: int,
    concurrency: int,
    warmup_requests: int,
) -> dict[str, Any]:
    """Warmup requests are executed then discarded from the timed window."""
    async with httpx.AsyncClient(timeout=None) as client:
        for i in range(warmup_requests):
            await _one_stream(
                client,
                base_url,
                f"warmup-{i}-{time.time_ns()}",
                prompts[i % len(prompts)],
                min(8, max_tokens),
            )

        sem = asyncio.Semaphore(concurrency)
        start = time.perf_counter()

        async def _guarded(i: int) -> dict[str, Any]:
            async with sem:
                return await _one_stream(
                    client,
                    base_url,
                    f"meas-{i}-{time.time_ns()}",
                    prompts[i % len(prompts)],
                    max_tokens,
                )

        results = await asyncio.gather(*[_guarded(i) for i in range(len(prompts))])
        elapsed = max(1e-6, time.perf_counter() - start)

    completed = sum(1 for r in results if r["status"] == "completed")
    output_tokens = sum(int(r["output_tokens"]) for r in results)
    return {
        "request_count": len(prompts),
        "warmup_requests_discarded": warmup_requests,
        "concurrency": concurrency,
        "completed_requests": completed,
        "success_rate": completed / max(1, len(prompts)),
        "output_tokens": output_tokens,
        "tokens_per_sec_output": output_tokens / elapsed,
        "duration_s": elapsed,
        "sample_errors": [r["error"] for r in results if r["error"]][:3],
    }


def _start_worker(label: str, overrides: dict[str, str]) -> tuple[subprocess.Popen[str], int, str | None]:
    port = _free_port()
    env = os.environ.copy()
    env.pop("HF_MODEL_ID", None)
    env.update(_baseline_env())
    env.update(overrides)
    stderr_path = ROOT / "bench" / "results" / f"_ablation_stable_{label}.stderr.log"
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
    deadline = time.time() + 360.0
    while time.time() < deadline:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=2.0).status_code == 200:
                stderr_f.flush()
                return proc, port, None
        except Exception:
            pass
        if proc.poll() is not None:
            stderr_f.flush()
            stderr_f.close()
            err = stderr_path.read_text(encoding="utf-8", errors="replace")[-2000:]
            return proc, port, f"worker exited: {err}"
        time.sleep(0.5)
    _stop_worker(proc)
    stderr_f.flush()
    stderr_f.close()
    return proc, port, "worker health timeout"


def _measure(
    label: str,
    overrides: dict[str, str],
    prompts: list[str],
    max_tokens: int,
    concurrency: int,
    warmup_requests: int,
    idle_s: float,
) -> dict[str, Any]:
    if idle_s > 0:
        time.sleep(idle_s)
    smi_before = _nvidia_smi_snapshot()
    proc, port, err = _start_worker(label, overrides)
    if err:
        _stop_worker(proc)
        return {
            "label": label,
            "status": "failed_startup",
            "error": err,
            "success_rate": 0.0,
            "tokens_per_sec_output": 0.0,
            "smi_before": smi_before,
            "smi_after": _nvidia_smi_snapshot(),
            "env_overrides": overrides,
        }
    try:
        # Model-load compile warmup inside worker via discarded warmups.
        metrics = asyncio.run(
            _run_timed_workload(
                f"http://127.0.0.1:{port}",
                prompts,
                max_tokens,
                concurrency,
                warmup_requests,
            )
        )
        try:
            worker_metrics = httpx.get(f"http://127.0.0.1:{port}/metrics", timeout=10.0).json()
        except Exception:
            worker_metrics = {}
        smi_after = _nvidia_smi_snapshot()
        metrics.update(
            {
                "label": label,
                "status": "ok" if metrics["success_rate"] >= 0.99 else "low_success",
                "smi_before": smi_before,
                "smi_after": smi_after,
                "env_overrides": overrides,
                "worker_metrics_subset": {
                    k: worker_metrics.get(k)
                    for k in (
                        "prefix_cache_hits",
                        "prefix_cache_misses",
                        "prefix_cache_hit_rate",
                        "continuous_batches_total",
                        "max_batch_size",
                        "peak_active_requests",
                        "kv_backend",
                    )
                },
            }
        )
        return metrics
    except Exception as exc:
        return {
            "label": label,
            "status": "failed_run",
            "error": str(exc),
            "success_rate": 0.0,
            "tokens_per_sec_output": 0.0,
            "smi_before": smi_before,
            "smi_after": _nvidia_smi_snapshot(),
            "env_overrides": overrides,
        }
    finally:
        _stop_worker(proc)


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    bstats = payload["baseline_control"]
    lines = [
        "# Stable feature ablation (instrument-fixed)",
        "",
        f"- Model: `{payload['model_id']}`",
        f"- GPU: `{payload['gpu_type']}`",
        f"- Scenario: `{payload['scenario_file']}` (`{payload['scenario_id']}`)",
        f"- Prompt/decode tokens: `{payload['prompt_tokens']}` / `{payload['max_output_tokens']}`",
        f"- Requests/concurrency/warmups: `{payload['request_count']}` / "
        f"`{payload['concurrency']}` / `{payload['warmup_requests']}`",
        f"- Repeats: `{payload['repeats']}`, seed `{payload['seed']}`",
        f"- Generated: `{payload['timestamp_utc']}`",
        "",
        "## Control: interleaved baseline noise",
        "",
        f"- n=`{int(bstats['n'])}` median=`{bstats['median']:.2f}` tok/s "
        f"IQR=`{bstats['iqr']:.2f}` stdev=`{bstats['stdev']:.2f}` "
        f"range=`[{bstats['min']:.2f}, {bstats['max']:.2f}]`",
        f"- Decision threshold: report a feature delta only if "
        f"|median Δ%| exceeds baseline IQR% "
        f"(≈ `{payload['decision_threshold_pct']:.1f}`% of baseline median).",
        "",
        "## Feature cells (paired vs immediate baseline)",
        "",
        "| Feature | median tok/s | median Δ% vs paired baseline | vs noise | success med | notes |",
        "|---------|-------------:|-----------------------------:|----------|------------:|-------|",
    ]
    for name in payload["feature_order"]:
        row = payload["features"][name]
        med = row["tps_stats"]["median"]
        dmed = row["delta_pct_stats"]["median"]
        verdict = row["verdict"]
        succ = row["success_stats"]["median"]
        note = row.get("note") or ""
        lines.append(
            f"| `{name}` | {med:.2f} | {dmed:+.1f}% | **{verdict}** | {succ:.2f} | {note} |"
        )
    lines.extend(
        [
            "",
            "## Thermal / clock summary",
            "",
            f"- Baseline SM clock median MHz: `{payload['thermal']['baseline_sm_mhz_median']}`",
            f"- Feature SM clock median MHz: `{payload['thermal']['feature_sm_mhz_median']}`",
            f"- Baseline temp median C: `{payload['thermal']['baseline_temp_c_median']}`",
            f"- Feature temp median C: `{payload['thermal']['feature_temp_c_median']}`",
            f"- Thermal confound suspected: `{payload['thermal']['confound_suspected']}`",
            "",
            "Prior single-shot Phase-4 tables are **not reproducible** under this protocol "
            "and must not be used in README claims.",
            "",
            "Artifact: `bench/results/ablation_matrix.json`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--idle-s", type=float, default=20.0, help="Idle between measurements (thermal)")
    parser.add_argument("--seed", type=int, default=5070)
    parser.add_argument(
        "--features",
        default=",".join(FEATURE_CELLS.keys()),
        help="Comma-separated feature cells",
    )
    parser.add_argument("--output-json", default="bench/results/ablation_matrix.json")
    parser.add_argument("--output-md", default="bench/results/ablation_matrix.md")
    args = parser.parse_args()

    scenario = yaml.safe_load((ROOT / SCENARIO_FILE).read_text(encoding="utf-8"))["scenarios"][0]
    features = [f.strip() for f in args.features.split(",") if f.strip()]
    for f in features:
        if f not in FEATURE_CELLS:
            raise SystemExit(f"unknown feature: {f}")

    rng = random.Random(args.seed)
    print("Building fixed-length prompts...", flush=True)
    prompts = _build_fixed_length_prompts(MODEL_ID, args.prompt_tokens, args.requests)
    ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    gpu = _gpu_type()

    pairs: list[dict[str, Any]] = []
    baseline_tps: list[float] = []
    baseline_smi: list[dict[str, Any]] = []

    for rep in range(args.repeats):
        order = features[:]
        rng.shuffle(order)
        print(f"=== repeat {rep + 1}/{args.repeats} order={order}", flush=True)
        for feat in order:
            print(f"  -> baseline before {feat}", flush=True)
            b = _measure(
                f"baseline_r{rep}_{feat}",
                {},
                prompts,
                args.max_output_tokens,
                args.concurrency,
                args.warmup_requests,
                args.idle_s,
            )
            print(
                f"     baseline tok/s={float(b.get('tokens_per_sec_output') or 0):.2f} "
                f"success={float(b.get('success_rate') or 0):.2f} "
                f"temp={((b.get('smi_after') or {}).get('temperature_c'))} "
                f"sm={(b.get('smi_after') or {}).get('clock_sm_mhz')}",
                flush=True,
            )
            print(f"  -> feature {feat}", flush=True)
            c = _measure(
                f"{feat}_r{rep}",
                FEATURE_CELLS[feat],
                prompts,
                args.max_output_tokens,
                args.concurrency,
                args.warmup_requests,
                args.idle_s,
            )
            print(
                f"     {feat} tok/s={float(c.get('tokens_per_sec_output') or 0):.2f} "
                f"success={float(c.get('success_rate') or 0):.2f} "
                f"temp={((c.get('smi_after') or {}).get('temperature_c'))} "
                f"sm={(c.get('smi_after') or {}).get('clock_sm_mhz')}",
                flush=True,
            )
            b_tps = float(b.get("tokens_per_sec_output") or 0.0)
            c_tps = float(c.get("tokens_per_sec_output") or 0.0)
            b_ok = float(b.get("success_rate") or 0.0) >= 0.99
            c_ok = float(c.get("success_rate") or 0.0) >= 0.99
            if b_ok:
                baseline_tps.append(b_tps)
                baseline_smi.append(b.get("smi_after") or {})
            delta_pct = None
            if b_ok and c_ok and b_tps > 0:
                delta_pct = 100.0 * (c_tps / b_tps - 1.0)
            pairs.append(
                {
                    "repeat": rep,
                    "feature": feat,
                    "baseline": b,
                    "feature_run": c,
                    "delta_pct": delta_pct,
                    "void": not (b_ok and c_ok),
                }
            )

    bstats = _stats(baseline_tps)
    threshold_pct = (100.0 * bstats["iqr"] / bstats["median"]) if bstats["median"] > 0 else 1e9

    feature_summaries: dict[str, Any] = {}
    for feat in features:
        feat_pairs = [p for p in pairs if p["feature"] == feat and not p["void"] and p["delta_pct"] is not None]
        tps_vals = [float(p["feature_run"]["tokens_per_sec_output"]) for p in feat_pairs]
        delta_vals = [float(p["delta_pct"]) for p in feat_pairs]
        succ_vals = [float(p["feature_run"]["success_rate"]) for p in feat_pairs]
        tstats = _stats(tps_vals)
        dstats = _stats(delta_vals)
        sstats = _stats(succ_vals)
        med_abs = abs(dstats["median"]) if delta_vals else 0.0
        if not delta_vals:
            verdict = "no_valid_pairs"
        elif med_abs <= threshold_pct:
            verdict = "within_noise"
        elif dstats["median"] > 0:
            verdict = "above_noise_gain"
        else:
            verdict = "above_noise_loss"
        note = ""
        if feat == "prefix_cache" and feat_pairs:
            hits = [
                (p["feature_run"].get("worker_metrics_subset") or {}).get("prefix_cache_hits")
                for p in feat_pairs
            ]
            note = f"prefix_hits_samples={hits}"
        feature_summaries[feat] = {
            "tps_stats": tstats,
            "delta_pct_stats": dstats,
            "success_stats": sstats,
            "verdict": verdict,
            "note": note,
            "pairs_valid": len(feat_pairs),
            "pairs_total": sum(1 for p in pairs if p["feature"] == feat),
        }

    def _med(vals: list[float]) -> float | None:
        return float(statistics.median(vals)) if vals else None

    b_sm = [float(s["clock_sm_mhz"]) for s in baseline_smi if isinstance(s.get("clock_sm_mhz"), (int, float))]
    b_temp = [float(s["temperature_c"]) for s in baseline_smi if isinstance(s.get("temperature_c"), (int, float))]
    f_sm: list[float] = []
    f_temp: list[float] = []
    for p in pairs:
        s = p["feature_run"].get("smi_after") or {}
        if isinstance(s.get("clock_sm_mhz"), (int, float)):
            f_sm.append(float(s["clock_sm_mhz"]))
        if isinstance(s.get("temperature_c"), (int, float)):
            f_temp.append(float(s["temperature_c"]))

    sm_spread = 0.0
    if b_sm and f_sm and _med(b_sm):
        sm_spread = abs((_med(f_sm) or 0) - (_med(b_sm) or 0)) / max(_med(b_sm) or 1.0, 1.0)
    confound = bool(bstats["iqr"] > 0.15 * max(bstats["median"], 1e-6) or sm_spread >= 0.15)

    payload = {
        "artifact_type": "feature_ablation_matrix_stable",
        "timestamp_utc": ts,
        "model_id": MODEL_ID,
        "gpu_type": gpu,
        "scenario_file": SCENARIO_FILE,
        "scenario_id": scenario["id"],
        "prompt_tokens": args.prompt_tokens,
        "max_output_tokens": args.max_output_tokens,
        "request_count": args.requests,
        "concurrency": args.concurrency,
        "warmup_requests": args.warmup_requests,
        "repeats": args.repeats,
        "seed": args.seed,
        "idle_s_between_measurements": args.idle_s,
        "measurement": "measured_on_gpu",
        "methodology": {
            "paired_baseline_before_each_cell": True,
            "randomized_feature_order_per_repeat": True,
            "warmup_discarded": True,
            "report_rule": "delta only if |median Δ%| > baseline IQR% of median baseline",
            "supersedes": "single-shot ablation tables from 38788c0/204bc73",
        },
        "baseline_control": bstats,
        "decision_threshold_pct": threshold_pct,
        "feature_order": features,
        "features": feature_summaries,
        "pairs": pairs,
        "thermal": {
            "baseline_sm_mhz_median": _med(b_sm),
            "feature_sm_mhz_median": _med(f_sm),
            "baseline_temp_c_median": _med(b_temp),
            "feature_temp_c_median": _med(f_temp),
            "sm_clock_relative_spread": sm_spread,
            "confound_suspected": confound,
        },
        "honest_finding": (
            "Prior Phase-4 single-shot numbers are not reproducible; "
            "only above-noise verdicts in this artifact are claim-eligible."
        ),
    }

    out_json = ROOT / args.output_json
    out_md = ROOT / args.output_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_md(out_md, payload)

    # Schema-minimal manifests for validator.
    for feat in features:
        man = {
            "run_id": f"ablation-stable-{feat}",
            "timestamp_utc": ts,
            "system_under_test": "phase2",
            "model_id": MODEL_ID,
            "rows": int(feature_summaries[feat]["pairs_valid"]),
            "scenario_file": SCENARIO_FILE,
            "verdict": feature_summaries[feat]["verdict"],
            "median_tokens_per_sec": feature_summaries[feat]["tps_stats"]["median"],
            "median_delta_pct": feature_summaries[feat]["delta_pct_stats"]["median"],
        }
        (ROOT / "bench" / "results" / f"{man['run_id']}.manifest.json").write_text(
            json.dumps(man, indent=2), encoding="utf-8"
        )
    base_man = {
        "run_id": "ablation-stable-baseline-control",
        "timestamp_utc": ts,
        "system_under_test": "phase2",
        "model_id": MODEL_ID,
        "rows": int(bstats["n"]),
        "scenario_file": SCENARIO_FILE,
        "baseline_median_tokens_per_sec": bstats["median"],
        "baseline_iqr": bstats["iqr"],
    }
    (ROOT / "bench" / "results" / "ablation-stable-baseline-control.manifest.json").write_text(
        json.dumps(base_man, indent=2), encoding="utf-8"
    )

    print(f"wrote: {out_json}")
    print(f"wrote: {out_md}")
    print(f"baseline median={bstats['median']:.2f} IQR={bstats['iqr']:.2f} threshold%={threshold_pct:.1f}")
    for feat in features:
        fs = feature_summaries[feat]
        print(
            f"  {feat}: median_delta%={fs['delta_pct_stats']['median']:+.1f} verdict={fs['verdict']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
