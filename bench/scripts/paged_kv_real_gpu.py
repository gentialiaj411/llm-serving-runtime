"""Measure peak GPU KV memory: contiguous StaticCache vs block-paged kernel path.

Runs the same variable-length workload twice on Qwen2-1.5B and records:
  - nvidia-smi memory.used time series
  - torch.cuda.max_memory_allocated() peak
  - per-request output throughput

Outputs:
  bench/results/paged_kv_real_gpu.json
  bench/results/paged_kv_real_gpu_smi.csv
  bench/results/paged_kv_real_gpu.md

Reproduce (RTX 5070, .venv311):
  .venv311\\Scripts\\python.exe bench/scripts/paged_kv_real_gpu.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

_STREAM_TIMEOUT_S = float(os.getenv("PAGED_KV_BENCH_STREAM_TIMEOUT_S", "7200"))
_WORKER_SHUTDOWN_S = float(os.getenv("PAGED_KV_BENCH_WORKER_SHUTDOWN_S", "120"))


def _paged_vs_contiguous_peak_delta_percent(paged_peak: float, contiguous_peak: float) -> float:
    """Percent change of paged peak over contiguous peak.

    Positive values mean paged used *more* peak memory than contiguous (a regression),
    not a reduction win.
    """
    if contiguous_peak <= 0:
        return 0.0
    return 100.0 * (paged_peak / contiguous_peak - 1.0)


def _format_peak_delta_markdown_line(
    label: str,
    delta_percent: float,
    field_name: str,
) -> str:
    direction = "higher" if delta_percent >= 0 else "lower"
    if delta_percent >= 0:
        signed = f"+{delta_percent:.2f}%"
    else:
        signed = f"{delta_percent:.2f}%"
    return (
        f"- Paged peak memory ({label}): **{signed} {direction}** vs contiguous "
        f"(`{field_name}`; positive = paged uses more memory)"
    )


@dataclass(frozen=True)
class Request:
    request_id: str
    prompt_tokens: int
    output_tokens: int
    timeout_step: int

    @property
    def token_capacity(self) -> int:
        return self.prompt_tokens + self.output_tokens


def make_workload(count: int, seed: int) -> list[Request]:
    rng = random.Random(seed)
    return [
        Request(
            request_id=f"req-{i}",
            prompt_tokens=rng.randint(32, 256),
            output_tokens=rng.randint(16, 128),
            timeout_step=rng.randint(20, 80),
        )
        for i in range(count)
    ]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _prompt_for_tokens(n: int) -> str:
  words = [f"w{i % 97}" for i in range(max(1, n))]
  return " ".join(words)


class SmiSampler:
    def __init__(self, interval_s: float = 0.25) -> None:
        self.interval_s = interval_s
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _loop(self) -> None:
        while not self._stop.is_set():
            ts = datetime.now(timezone.utc).isoformat()
            mb = _query_gpu_mem_mb()
            self.samples.append({"timestamp_utc": ts, "memory_used_mb": mb})
            self._stop.wait(self.interval_s)

    def peak_mb(self) -> float:
        if not self.samples:
            return 0.0
        return float(max(s["memory_used_mb"] for s in self.samples))


def _query_gpu_mem_mb() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
        return float(out.strip().splitlines()[0].strip())
    except Exception:
        return 0.0


def _start_worker(kv_backend: str, model_id: str, total_blocks: int, block_size: int) -> tuple[subprocess.Popen[str], int]:
    port = _free_port()
    env = os.environ.copy()
    env.pop("HF_MODEL_ID", None)
    env.update(
        {
            "PHASE2_BACKEND": "transformers",
            "PHASE2_KV_BACKEND": kv_backend,
            "HF_MODEL_ID": model_id,
            "HF_TORCH_DTYPE": env.get("HF_TORCH_DTYPE", "float16"),
            "HF_DEVICE": env.get("HF_DEVICE", "cuda"),
            "KV_TOTAL_BLOCKS": str(total_blocks),
            "KV_BLOCK_SIZE_TOKENS": str(block_size),
            "PHASE2_BATCH_DECODE_STEPS": "1",
            "PHASE2_DECODE_STEP_MS": "1",
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
    deadline = time.time() + 120.0
    while time.time() < deadline:
        try:
            if httpx.get(health, timeout=2.0).status_code == 200:
                return proc, port
        except Exception:
            pass
        if proc.poll() is not None:
            err = proc.stderr.read() if proc.stderr else ""
            raise RuntimeError(f"worker exited early: {err}")
        time.sleep(0.5)
    proc.terminate()
    proc.wait(timeout=10)
    raise RuntimeError("worker did not become healthy")


async def _one_stream(
    client: httpx.AsyncClient,
    base_url: str,
    req: Request,
) -> dict[str, Any]:
    prompt = _prompt_for_tokens(req.prompt_tokens)
    output_tokens = 0
    status = "error"
    async with client.stream(
        "POST",
        f"{base_url}/generate_stream",
        json={
            "request_id": req.request_id,
            "prompt": prompt,
            "max_tokens": req.output_tokens,
            "temperature": 0.0,
        },
        timeout=httpx.Timeout(_STREAM_TIMEOUT_S, connect=60.0),
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
    return {"request_id": req.request_id, "status": status, "output_tokens": output_tokens}


async def _run_workload(
    base_url: str,
    workload: list[Request],
    max_active: int,
) -> dict[str, Any]:
    sem = asyncio.Semaphore(max_active)
    start = time.perf_counter()

    async with httpx.AsyncClient(timeout=None) as client:

        async def _guarded(req: Request) -> dict[str, Any]:
            async with sem:
                return await _one_stream(client, base_url, req)

        results = await asyncio.gather(*[_guarded(r) for r in workload])

    elapsed = max(1e-6, time.perf_counter() - start)
    completed = sum(1 for r in results if r["status"] == "completed")
    output_tokens = sum(int(r["output_tokens"]) for r in results)
    return {
        "request_count": len(workload),
        "max_active": max_active,
        "completed_requests": completed,
        "success_rate": completed / max(1, len(workload)),
        "output_tokens": output_tokens,
        "output_tokens_per_sec": output_tokens / elapsed,
        "duration_s": elapsed,
    }


def _reset_cuda_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _peak_cuda_bytes() -> int:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated())
    except Exception:
        pass
    return 0


def _run_mode(
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
    try:
        metrics = asyncio.run(_run_workload(base_url, workload, max_active))
        worker_metrics = httpx.get(f"{base_url}/metrics", timeout=10.0).json()
    finally:
        sampler.stop()
        proc.terminate()
        try:
            proc.wait(timeout=int(_WORKER_SHUTDOWN_S))
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)

    peak_torch = int(worker_metrics.get("peak_torch_cuda_bytes", 0))
    smi_samples = list(sampler.samples)
    return {
        "kv_backend": kv_backend,
        "label": (
            "contiguous_worst_case_reservation"
            if kv_backend == "reserved"
            else ("contiguous_static_cache" if kv_backend == "static" else "paged_block_kernel")
        ),
        **metrics,
        "peak_torch_cuda_bytes": peak_torch,
        "peak_torch_cuda_mb": peak_torch / (1024 * 1024),
        "peak_nvidia_smi_mb": sampler.peak_mb(),
        "nvidia_smi_samples": smi_samples,
        "worker_peak_kv_bytes": worker_metrics.get("peak_kv_bytes", 0),
        "paged_kv_pool_peak_bytes": int(worker_metrics.get("paged_kv_pool_peak_bytes", 0)),
        "worker_kv_backend": worker_metrics.get("kv_backend"),
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    cont = payload["modes"]["contiguous"]
    paged = payload["modes"]["paged"]
    delta_torch = payload["paged_vs_contiguous_torch_peak_delta_percent"]
    delta_smi = payload["paged_vs_contiguous_nvidia_smi_peak_delta_percent"]
    lines = [
        "# Paged KV — measured GPU memory (Qwen2-1.5B)",
        "",
        f"- Run UTC: `{payload['timestamp_utc']}`",
        f"- Model: `{payload['model_id']}`",
        f"- GPU: `{payload.get('gpu_name', 'unknown')}`",
        f"- Workload: `{payload['workload']['request_count']}` requests, variable prompt/output (seed `{payload['workload']['seed']}`), max_active `{payload['workload']['max_active']}`",
        "",
        "## Peak memory",
        "",
        "| Path | torch.cuda.max_memory_allocated (MB) | nvidia-smi peak used (MB) | tokens/sec |",
        "|------|--------------------------------------|---------------------------|------------|",
        f"| Contiguous (`PHASE2_KV_BACKEND=reserved`) | {cont['peak_torch_cuda_mb']:.1f} | {cont['peak_nvidia_smi_mb']:.1f} | {cont['output_tokens_per_sec']:.2f} |",
        f"| Paged kernel (`PHASE2_KV_BACKEND=paged`) | {paged['peak_torch_cuda_mb']:.1f} | {paged['peak_nvidia_smi_mb']:.1f} | {paged['output_tokens_per_sec']:.2f} |",
        "",
        _format_peak_delta_markdown_line(
            "PyTorch peak",
            delta_torch,
            "paged_vs_contiguous_torch_peak_delta_percent",
        ),
        _format_peak_delta_markdown_line(
            "nvidia-smi peak",
            delta_smi,
            "paged_vs_contiguous_nvidia_smi_peak_delta_percent",
        ),
        "## Which memory metric to headline",
        "",
        "**Headline metric:** logical KV efficiency and concurrency",
        "under fixed block budget (`bench/results/paged_kv_concurrency_scale.json`).",
        "Use **`paged_kv_pool_peak_bytes`** from worker `/metrics` for KV-specific bytes.",
        "",
        "PyTorch `max_memory_allocated()` and nvidia-smi can diverge; paged is not always",
        "lower on both at moderate concurrency:",
        "",
        "- **Contiguous (`reserved`)** admits each request with a worst-case `contiguous_kv_hold`",
        "  tensor (`[layers, 2, kv_heads, token_capacity, head_dim]`) *plus* Hugging Face",
        "  `StaticCache` decode storage. Driver-resident footprint can spike at high",
        "  `max_active × token_capacity`.",
        "- **Paged (`paged`)** stores KV in a shared `GpuKVBlockPool`; only touched physical",
        "  blocks are populated. Logical block usage (`peak_kv_bytes` from `PagedKVAllocator`)",
        "  matches contiguous on this workload (~806 MB); **`paged_kv_pool_peak_bytes`** tracks",
        "  actual pool block bytes (~134 MB in repeat runs).",
        "- **Why PyTorch peak is higher on paged:** the pool uses a growable slab",
        "  (`_grow_pools` doubles `[num_blocks, layers, …]` capacity).",
        "  `torch.cuda.max_memory_allocated()` counts the expanded slab plus fp32 Triton",
        "  scratch (`m_parts`/`l_parts`/`acc_parts` in `paged_attention_triton.py`).",
        "- **Why nvidia-smi can be higher on paged at max_active=8:** the pool slab is",
        "  driver-resident even when logical fill is low; repeat runs (5×) show paged",
        "  nvidia-smi median ~5444 MB vs contiguous ~4797 MB on this workload.",
        "",
        f"Throughput: see `bench/results/paged_kv_repeat.json` for repeat medians; "
        f"check per-run `modes.paged.success_rate` — paged tok/s is only comparable at success_rate=1.0.",
        "",
        "## Notes",
        "",
        "- Contiguous baseline uses explicit worst-case GPU KV reservation (`PHASE2_KV_BACKEND=reserved`) plus `StaticCache` for decode.",
        "- Paged path uses `GpuKVBlockPool` + `BlockPagedCache` with `PagedKVAllocator` block admission/free.",
        "- Supersedes modeled-only claim in `bench/results/kv-pressure.json` (see `CLAIMS_MATRIX.md`).",
        "",
        "Raw traces: `bench/results/paged_kv_real_gpu.json`, `bench/results/paged_kv_real_gpu_smi.csv`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=os.getenv("HF_MODEL_ID", "Qwen/Qwen2-1.5B-Instruct"))
    parser.add_argument("--requests", type=int, default=48)
    parser.add_argument("--seed", type=int, default=5070)
    parser.add_argument("--max-active", type=int, default=8)
    parser.add_argument("--total-blocks", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--output-json", default="bench/results/paged_kv_real_gpu.json")
    parser.add_argument("--output-csv", default="bench/results/paged_kv_real_gpu_smi.csv")
    parser.add_argument("--output-md", default="bench/results/paged_kv_real_gpu.md")
    args = parser.parse_args()

    try:
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("CUDA required for paged_kv_real_gpu benchmark")
        gpu_name = torch.cuda.get_device_name(0)
    except ImportError as exc:
        raise SystemExit(f"torch required: {exc}") from exc

    workload = make_workload(args.requests, args.seed)
    total_tokens = sum(req.token_capacity for req in workload)
    total_blocks = max(args.total_blocks, total_tokens // args.block_size + args.requests + args.max_active)
    timestamp = datetime.now(timezone.utc).isoformat()

    contiguous = _run_mode("reserved", workload, args.model_id, args.max_active, total_blocks, args.block_size)
    paged = _run_mode("paged", workload, args.model_id, args.max_active, total_blocks, args.block_size)

    delta_torch = _paged_vs_contiguous_peak_delta_percent(
        paged["peak_torch_cuda_bytes"], contiguous["peak_torch_cuda_bytes"]
    )
    delta_smi = _paged_vs_contiguous_peak_delta_percent(
        paged["peak_nvidia_smi_mb"], contiguous["peak_nvidia_smi_mb"]
    )

    payload: dict[str, Any] = {
        "timestamp_utc": timestamp,
        "model_id": args.model_id,
        "gpu_name": gpu_name,
        "measurement": "measured_on_gpu",
        "metric_notes": {
            "paged_vs_contiguous_torch_peak_delta_percent": (
                "100 * (paged/contiguous - 1). Positive = paged peak uses more memory than contiguous."
            ),
            "paged_vs_contiguous_nvidia_smi_peak_delta_percent": (
                "100 * (paged/contiguous - 1) on nvidia-smi peak MB. Positive = paged uses more memory."
            ),
        },
        "evidence": {
            "torch_peak": "torch.cuda.max_memory_allocated() after workload",
            "nvidia_smi": "nvidia-smi --query-gpu=memory.used --format=csv -l equivalent sampling",
        },
        "workload": {
            "request_count": args.requests,
            "seed": args.seed,
            "max_active": args.max_active,
            "workload_model": "synthetic_uniform_prompt_output_with_timeout",
        },
        "modes": {
            "contiguous": contiguous,
            "paged": paged,
        },
        "paged_vs_contiguous_torch_peak_delta_percent": delta_torch,
        "paged_vs_contiguous_nvidia_smi_peak_delta_percent": delta_smi,
    }

    out_json = ROOT / args.output_json
    out_json.parent.mkdir(parents=True, exist_ok=True)

    csv_path = ROOT / args.output_csv
    rows = ["timestamp_utc,mode,memory_used_mb\n"]
    for mode_name, mode in [("contiguous", contiguous), ("paged", paged)]:
        for sample in mode.get("nvidia_smi_samples", []):
            if isinstance(sample, dict):
                rows.append(f"{sample.get('timestamp_utc', timestamp)},{mode_name},{sample.get('memory_used_mb', 0)}\n")
    csv_path.write_text("".join(rows), encoding="utf-8")
    # Drop verbose time series from JSON artifact (keep peaks only)
    for mode in (payload["modes"]["contiguous"], payload["modes"]["paged"]):
        mode.pop("nvidia_smi_samples", None)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_path = ROOT / args.output_md
    _write_markdown(md_path, payload)

    print(f"wrote: {out_json}")
    print(f"wrote: {csv_path}")
    print(f"wrote: {md_path}")
    print(f"paged vs contiguous torch peak delta: {delta_torch:+.2f}% (positive = paged uses more memory)")
    print(f"paged vs contiguous nvidia-smi peak delta: {delta_smi:+.2f}% (positive = paged uses more memory)")


if __name__ == "__main__":
    main()
