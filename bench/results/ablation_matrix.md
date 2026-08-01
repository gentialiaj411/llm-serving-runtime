# Stable feature ablation (instrument-fixed)

- Model: `Qwen/Qwen2-1.5B-Instruct`
- GPU: `NVIDIA GeForce RTX 5070 Laptop GPU`
- Scenario: `bench/scenarios/ablation_decode_heavy.yaml` (`ablation_decode_heavy`)
- Prompt/decode tokens: `128` / `512`
- Requests/concurrency/warmups: `8` / `8` / `2`
- Repeats: `5`, seed `5070`
- Generated: `2026-08-01T22:40:22Z`

## Control: interleaved baseline noise

- n=`15` median=`132.24` tok/s IQR=`7.07` stdev=`34.15` range=`[108.39, 226.03]`
- Decision threshold: report a feature delta only if |median Δ%| exceeds baseline IQR% (≈ `5.3`% of baseline median).

## Feature cells (paired vs immediate baseline)

| Feature | median tok/s | median Δ% vs paired baseline | vs noise | success med | notes |
|---------|-------------:|-----------------------------:|----------|------------:|-------|
| `continuous_batching` | 136.35 | -0.3% | **within_noise** | 1.00 |  |
| `paged_kv` | 66.45 | -50.5% | **above_noise_loss** | 1.00 |  |
| `prefix_cache` | 97.80 | -28.9% | **above_noise_loss** | 1.00 | prefix_hits_samples=[9, 9, 9, 9, 9] |

## Thermal / clock summary

- Baseline SM clock median MHz: `2887.0`
- Feature SM clock median MHz: `2895.0`
- Baseline temp median C: `64.0`
- Feature temp median C: `63.0`
- Thermal confound suspected: `False`

Prior single-shot Phase-4 tables are **not reproducible** under this protocol and must not be used in README claims.

Artifact: `bench/results/ablation_matrix.json`
