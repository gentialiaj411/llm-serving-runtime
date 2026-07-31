"""Concurrency scaling: paged vs contiguous reserved under fixed KV block budget.

Sweeps max_active with long sequences so the contiguous per-request worst-case
reservation path hits VRAM / admission pressure before the shared block pool does.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import importlib.util

_paged_gpu_path = ROOT / "bench" / "scripts" / "paged_kv_real_gpu.py"
_spec = importlib.util.spec_from_file_location("paged_kv_real_gpu", _paged_gpu_path)
_paged_gpu = importlib.util.module_from_spec(_spec)
sys.modules["paged_kv_real_gpu"] = _paged_gpu
assert _spec.loader is not None
_spec.loader.exec_module(_paged_gpu)

Request = _paged_gpu.Request
SmiSampler = _paged_gpu.SmiSampler
_run_workload = _paged_gpu._run_workload
_start_worker = _paged_gpu._start_worker
_reset_cuda_peak = _paged_gpu._reset_cuda_peak
_peak_cuda_bytes = _paged_gpu._peak_cuda_bytes

import httpx


def _long_workload(count: int, prompt_tokens: int, output_tokens: int) -> list[Request]:
    return [
        Request(
            request_id=f"req-{i}",
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            timeout_step=9999,
        )
        for i in range(count)
    ]


def _run_mode_at_concurrency(
    kv_backend: str,
    workload: list[Request],
    model_id: str,
    max_active: int,
    total_blocks: int,
    block_size: int,
) -> dict[str, Any]:
    _reset_cuda_peak()
    proc, port = _start_worker(kv_backend, model_id, total_blocks, block_size)
    base_url = f"http://127.0.0.1:{port}"
    sampler = SmiSampler(interval_s=0.25)
    sampler.start()
    status = "ok"
    err_msg = ""
    try:
        metrics = asyncio.run(_run_workload(base_url, workload, max_active))
        worker_metrics = httpx.get(f"{base_url}/metrics", timeout=10.0).json()
    except Exception as exc:
        status = "error"
        err_msg = str(exc)
        metrics = {
            "request_count": len(workload),
            "max_active": max_active,
            "completed_requests": 0,
            "success_rate": 0.0,
            "output_tokens": 0,
            "output_tokens_per_sec": 0.0,
            "duration_s": 0.0,
        }
        worker_metrics = {}
    finally:
        sampler.stop()
        proc.terminate()
        try:
            proc.wait(timeout=120)
        except Exception:
            proc.kill()
            proc.wait(timeout=30)

    peak_torch = int(worker_metrics.get("peak_torch_cuda_bytes", _peak_cuda_bytes()))
    return {
        "status": status,
        "error": err_msg,
        "kv_backend": kv_backend,
        "max_active": max_active,
        **metrics,
        "peak_torch_cuda_bytes": peak_torch,
        "peak_torch_cuda_mb": peak_torch / (1024 * 1024),
        "peak_nvidia_smi_mb": sampler.peak_mb(),
        "allocation_failures_total": int(worker_metrics.get("allocation_failures_total", 0)),
        "peak_active_allocations": int(worker_metrics.get("peak_active_allocations", 0)),
        "paged_kv_pool_peak_bytes": int(worker_metrics.get("paged_kv_pool_peak_bytes", 0)),
        "worker_peak_kv_bytes": int(worker_metrics.get("peak_kv_bytes", 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=os.getenv("HF_MODEL_ID", "Qwen/Qwen2-1.5B-Instruct"))
    parser.add_argument("--requests", type=int, default=24)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--max-active-sweep", default="4,8,12,16,20,24")
    parser.add_argument("--total-blocks", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--output-json", default="bench/results/paged_kv_concurrency_scale.json")
    parser.add_argument("--output-md", default="bench/results/paged_kv_concurrency_scale.md")
    parser.add_argument("--output-manifest", default="bench/results/paged_kv_concurrency_scale.manifest.json")
    args = parser.parse_args()

    try:
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("CUDA required")
        gpu_name = torch.cuda.get_device_name(0)
    except ImportError as exc:
        raise SystemExit(f"torch required: {exc}") from exc

    sweep = [int(x.strip()) for x in args.max_active_sweep.split(",") if x.strip()]
    workload = _long_workload(args.requests, args.prompt_tokens, args.output_tokens)
    token_capacity = args.prompt_tokens + args.output_tokens
    blocks_per_req = max(1, (token_capacity + args.block_size - 1) // args.block_size)
    total_blocks = max(args.total_blocks, blocks_per_req * max(sweep) + args.requests)

    points: list[dict[str, Any]] = []
    for max_active in sweep:
        print(f"[sweep] max_active={max_active} ...", flush=True)
        cont = _run_mode_at_concurrency(
            "reserved", workload, args.model_id, max_active, total_blocks, args.block_size
        )
        paged = _run_mode_at_concurrency(
            "paged", workload, args.model_id, max_active, total_blocks, args.block_size
        )
        points.append(
            {
                "max_active": max_active,
                "token_capacity": token_capacity,
                "blocks_per_request": blocks_per_req,
                "contiguous": cont,
                "paged": paged,
            }
        )
        print(
            f"  cont success={cont['success_rate']:.2f} smi={cont['peak_nvidia_smi_mb']:.0f}MB | "
            f"paged success={paged['success_rate']:.2f} smi={paged['peak_nvidia_smi_mb']:.0f}MB",
            flush=True,
        )

    # Highest max_active where each path completes with success_rate == 1.0
    def _max_sustainable(rows: list[dict], mode: str) -> int | None:
        ok = [int(r["max_active"]) for r in rows if r[mode].get("success_rate", 0) >= 1.0]
        return max(ok) if ok else None

    cont_sust = _max_sustainable(points, "contiguous")
    paged_sust = _max_sustainable(points, "paged")

    timestamp = datetime.now(timezone.utc).isoformat()
    payload: dict[str, Any] = {
        "timestamp_utc": timestamp,
        "model_id": args.model_id,
        "gpu_name": gpu_name,
        "measurement": "concurrency_scaling",
        "workload": {
            "request_count": args.requests,
            "prompt_tokens": args.prompt_tokens,
            "output_tokens": args.output_tokens,
            "token_capacity": token_capacity,
            "total_blocks": total_blocks,
            "block_size": args.block_size,
            "max_active_sweep": sweep,
        },
        "summary": {
            "max_sustainable_max_active_contiguous": cont_sust,
            "max_sustainable_max_active_paged": paged_sust,
            "paged_advantage_max_active_delta": (paged_sust or 0) - (cont_sust or 0),
        },
        "points": points,
    }

    out_json = ROOT / args.output_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = [
        "# Paged KV — concurrency scaling (long sequences)",
        "",
        f"- Model: `{args.model_id}` on `{gpu_name}`",
        f"- Workload: {args.requests} requests, {args.prompt_tokens}+{args.output_tokens} tokens each",
        f"- KV block budget: `{total_blocks}` blocks × {args.block_size} tokens",
        f"- max_active sweep: `{sweep}`",
        "",
        "| max_active | cont success | cont smi peak MB | paged success | paged smi peak MB | cont alloc fails | paged alloc fails |",
        "|------------|--------------|------------------|---------------|-------------------|------------------|-------------------|",
    ]
    for pt in points:
        c, p = pt["contiguous"], pt["paged"]
        md.append(
            f"| {pt['max_active']} | {c['success_rate']:.2f} | {c['peak_nvidia_smi_mb']:.0f} | "
            f"{p['success_rate']:.2f} | {p['peak_nvidia_smi_mb']:.0f} | "
            f"{c.get('allocation_failures_total', 0)} | {p.get('allocation_failures_total', 0)} |"
        )
    md.extend(
        [
            "",
            f"- Max sustainable max_active (success=1.0): contiguous **{cont_sust}**, paged **{paged_sust}**",
            "",
            "See `docs/adr/0005-paged-attention-kernel.md` for which memory metric to headline.",
        ]
    )
    out_md = ROOT / args.output_md
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    manifest = {
        "run_id": "paged-kv-concurrency-scale",
        "timestamp_utc": timestamp,
        "system_under_test": "phase2",
        "model_id": args.model_id,
        "rows": len(points),
        "gpu_count": 1,
    }
    out_manifest = ROOT / args.output_manifest
    out_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"wrote: {out_json}")
    print(f"wrote: {out_md}")
    print(f"wrote: {out_manifest}")


if __name__ == "__main__":
    main()
