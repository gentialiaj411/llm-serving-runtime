# Paged KV — repeat-run throughput stability

- Runs: `5` (same workload: 32 req, max_active 8, seed 5070)
- Model: `Qwen/Qwen2-1.5B-Instruct`
- Aggregate UTC: `2026-06-16T02:53:03.355838+00:00`

## Throughput (tokens/sec)

| Path | median | stdev | min | max |
|------|--------|-------|-----|-----|
| Contiguous | 19.91 | 0.98 | 19.09 | 21.42 |
| Paged | 3.20 | 1.02 | 1.51 | 4.12 |
| Paged − contiguous | -17.12 | 1.30 | -18.40 | -15.50 |

**Conclusion:** throughput at parity within noise = **False** (median gap -17.12 tok/s, -83.6%).

## Peak nvidia-smi (MB)

- Contiguous median: 4859.0 (spread 4859.0–5056.0)
- Paged median: 5855.0 (spread 5855.0–6052.0)

Individual runs: `bench/results/paged_kv_repeat/runs/`.
