# Performance SLOs

This repository enforces per-scenario benchmark SLOs using `bench/slo/scenario_slos.yaml`.

## Policy location
- `bench/slo/scenario_slos.yaml`

## Enforced metrics
- `success_rate` (minimum)
- `timeout_rate` (maximum)
- `http_error_rate` (maximum)
- `ttft_ms_p95` (maximum)
- `latency_ms_p95` (maximum)

## Enforcement command
```bash
python bench/scripts/enforce_slos.py --csv <path-to-benchmark.csv>
```

## CI enforcement
- CI runs SLO checks for:
- `bench/results/ci-smoke.csv`
- `bench/results/ci-phase2-smoke.csv`
- Workflow file: `.github/workflows/ci.yml`

## Dashboard snapshot workflow
- Script: `bench/scripts/dashboard_snapshot.py`
- Optional coordinator metrics source: `http://127.0.0.1:8001/metrics`
- Snapshot outputs:
- `bench/results/<name>.json`
- `bench/results/<name>.md`
