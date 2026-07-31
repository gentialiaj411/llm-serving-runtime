# vLLM baseline — historical blocker (`vllm-baseline-final`)

Timestamp: **2026-06-16** overnight run (WSL2, RTX 5070 Laptop sm_120)

## Resolution

This blocker was later cleared on **2026-06-16**. The successful rerun produced:

- `bench/results/vllm-baseline-final.csv`
- `bench/results/vllm-baseline-final.manifest.json`

The fix was:

1. Switch launcher readiness from `/health` to `/v1/models`.
2. Run vLLM in eager mode with a longer readiness window.
3. Apply a local compatibility patch to `prometheus-fastapi-instrumentator`
   route walking in `.venv-vllm-bench` so wrapped routers no longer crash
   request handling.

## Outcome

This file records the overnight failure only. It is superseded by the successful
artifacts above.

## Attempts

### 1. `scripts/run_vllm_baseline_ubuntu.sh` (default)

- **Command:** `RUN_ID=vllm-baseline-final GPU_TYPE=rtx5070-laptop VLLM_VERSION=0.21.0 TORCH_CUDA_ARCH_LIST=12.0 bash scripts/run_vllm_baseline_ubuntu.sh`
- **Log:** `bench/results/_overnight_logs/phase4_vllm.log`
- **Server log:** `/tmp/vllm-baseline-final.vllm.log`
- **Result:** Health check failed after **240 s**. Engine reached `Starting to load model TinyLlama/...` but `/health` never returned 200 before timeout.
- **vLLM:** `0.21.0` in `.venv-vllm-bench` (Python 3.11)

### 2. Eager mode + 600 s health wait (`bench/scripts/_overnight_phase4_vllm.sh`)

- **Flags:** `--enforce-eager`, `gpu-memory-utilization=0.55`, `VLLM_USE_FLASHINFER_SAMPLER=0`
- **Log:** `bench/results/_overnight_logs/phase4_vllm_retry.log`
- **Server log:** `/tmp/vllm-baseline-final.vllm.eager.log`
- **Result:** Health check failed after **600 s**. Server process running but **`GET /health` returns HTTP 500**:

```
AttributeError: '_IncludedRouter' object has no attribute 'path'
  prometheus_fastapi_instrumentator/routing.py", line 55, in _get_route_name
    route_name = route.path
```

This is a **FastAPI / prometheus-fastapi-instrumentator** compatibility issue in the vLLM 0.21.0 API server stack — not a GPU OOM or FlashInfer JIT compiler failure on this attempt.

### 3. Not attempted (time / blocker type)

- **Build vLLM from source** against CUDA 12.8 — deferred; attempts 1–2 did not reach a healthy server suitable for `bench/harness/run.py`.
- **Conda `gcc_linux-64`** — not required on retry (no `x86_64-conda-linux-gnu-cc` error observed).

## Next steps (for a future run)

1. Pin compatible `prometheus-fastapi-instrumentator` / Starlette versions in `.venv-vllm-bench`, or use a vLLM release that fixes `_IncludedRouter` routing.
2. Alternatively run baseline on a supported Linux host via `scripts/gpu_host_handoff.sh` (A100/H100 / Ada).
3. After `/health` returns 200, re-run harness to emit `vllm-baseline-final.csv` + manifest and `plot_comparison.py`.

## Claims impact

- Historical note only. The blocker is resolved by `bench/results/vllm-baseline-final.{csv,manifest.json}`.
- `docs/reports/engineering-deep-dive.md` P4–P10 are filled from the successful rerun artifacts.
