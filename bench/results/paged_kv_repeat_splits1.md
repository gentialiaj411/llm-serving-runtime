# Paged KV — repeat-run throughput stability

- Runs: `5` (same workload: 32 req, max_active 8, seed 5070)
- Model: `Qwen/Qwen2-1.5B-Instruct`
- Aggregate UTC: `2026-06-14T22:34:14.033774+00:00`

## Throughput (tokens/sec)

| Path | median | stdev | min | max |
|------|--------|-------|-----|-----|
| Contiguous | 19.89 | 0.89 | 19.39 | 21.64 |
| Paged | 19.16 | 1.22 | 16.95 | 19.86 |
| Paged − contiguous | -1.19 | 1.58 | -3.69 | -0.03 |

**Conclusion:** throughput at parity within noise = **True** (median gap -1.19 tok/s, -5.9%).

## Peak nvidia-smi (MB)

- Contiguous median: 5303.0 (spread 5238.0–5565.0)
- Paged median: 5886.0 (spread 5783.0–6056.0)

Individual runs: `bench/results/paged_kv_repeat/runs/`.
