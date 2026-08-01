# Feature ablation matrix

- Model: `Qwen/Qwen2-1.5B-Instruct`
- GPU: `NVIDIA GeForce RTX 5070 Laptop GPU`
- Scenario: `bench/scenarios/ablation_matrix.yaml` (`ablation_shared_prefix`)
- Shared prefix tokens: `256`
- Requests: `24`, concurrency `8`, decode `32` tokens
- Generated: `2026-08-01T21:42:34Z`

One feature enabled at a time vs all-off baseline. Throughput at success_rate < 0.99 is void.

| Feature | tok/s | Δ vs baseline | success | status | notes |
|---------|------:|--------------:|--------:|--------|-------|
| `baseline_all_off` | 23.87 | — | 1.00 | ok |  |
| `continuous_batching` | 17.37 | -27.2% | 1.00 | ok | continuous_batches_total=107 |
| `paged_kv` | 7.74 | -67.6% | 1.00 | ok |  |
| `prefix_cache` | 18.34 | -23.2% | 1.00 | ok | hits=0 miss=25 |
| `speculative_decoding` | 6.98 | -70.8% | 1.00 | ok |  |
| `int4_awq` | 0.00 (void) | void | 0.00 | failed_startup | model warmup failed: AWQ runtime unavailable for PHASE2_QUANT=int4: No module... |
| `cuda_graphs` | 7.37 | -69.1% | 1.00 | ok | vs paged_kv: -4.7% |

Per-cell manifests: `bench/results/ablation-<feature>.manifest.json`
Aggregate: `bench/results/ablation_matrix.json`

Validate: `python bench/scripts/validate_manifests.py`
