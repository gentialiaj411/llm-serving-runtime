# Feature ablation matrix (corrected)

- Model: `Qwen/Qwen2-1.5B-Instruct`
- Draft (speculative): `Qwen/Qwen2-0.5B-Instruct`
- GPU: `NVIDIA GeForce RTX 5070 Laptop GPU`
- Scenario: `bench/scenarios/ablation_matrix.yaml` (`ablation_shared_prefix`)
- Shared prefix tokens: `256`
- Requests: `24`, concurrency `8`, decode `32` tokens
- Generated: `2026-08-01T22:27:55Z`

Throughput at success_rate < 0.99 is void.

## Table B — cumulative ladder (publish this)

| Step | tok/s | Δ vs previous | Δ vs baseline | success | notes |
|------|------:|--------------:|--------------:|--------:|-------|
| `ladder_baseline` | 87.26 | — | — | 1.00 |  |
| `ladder_plus_cb` | 63.81 | -26.9% | -26.9% | 1.00 |  |
| `ladder_plus_paged` | 23.11 | -63.8% | -73.5% | 1.00 | max_batch=8 peak_active=8 |
| `ladder_plus_prefix` | 43.90 (void) | — | void | 0.33 | hits=16 miss=9 |
| `ladder_plus_graphs` | 2.84 (void) | — | void | 0.00 |  |

## Table A — meaningful one-at-a-time (isolation control)

Caveat: paged KV / CUDA graphs are measured at `PHASE2_MAX_ACTIVE=8` because batching is required for those features to be meaningful.

| Feature | tok/s | Δ vs baseline | success | status | notes |
|---------|------:|--------------:|--------:|--------|-------|
| `baseline_all_off` | 68.61 | — | 1.00 | ok |  |
| `continuous_batching` | 79.54 | +15.9% | 1.00 | ok |  |
| `paged_kv` | 23.09 | -66.3% | 1.00 | ok | max_batch=8 peak_active=8 |
| `prefix_cache` | 79.07 | +15.2% | 1.00 | ok | hits=18 miss=7 |
| `speculative_decoding` | 12.70 | -81.5% | 1.00 | ok | accept=0.5383615084525357 |
| `int4_awq` | 0.00 (void) | void | 0.00 | failed_startup | model warmup failed: AWQ runtime unavailable for PHASE2_QUANT=int4:... |
| `cuda_graphs` | 27.62 | -59.7% | 1.00 | ok | vs paged_kv: +19.6% |

## Phase 2 cross-check

- Phase 2 batched paged B=8 proxy: `91.80450932044542` (`paged_kv_launch_profile.meta.json`)
- Table A `paged_kv` harness tok/s: `23.08985110397588`
- Ratio harness/proxy: `0.25151107799487615`
- Verdict: `harness paged@ma=8 far below Phase 2 proxy — investigate before README claims`

Per-cell manifests: `bench/results/ablation-<feature>.manifest.json`

Validate: `python bench/scripts/validate_manifests.py`
