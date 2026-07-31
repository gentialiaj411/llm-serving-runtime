# Paged KV — repeat-run throughput stability

- Runs: `5` (same workload: 32 req, max_active 8, seed 5070)
- Model: `Qwen/Qwen2-1.5B-Instruct`
- Aggregate UTC: `2026-06-14T22:13:27.600796+00:00`

## Throughput (tokens/sec)

| Path | median | stdev | min | max |
|------|--------|-------|-----|-----|
| Contiguous | 19.10 | 0.37 | 18.48 | 19.36 |
| Paged | 17.69 | 0.62 | 16.75 | 18.33 |
| Paged − contiguous | -1.03 | 0.70 | -2.35 | -0.47 |

**Conclusion:** throughput at parity within noise = **False** (median gap -1.03 tok/s, -5.3%).

## Peak nvidia-smi (MB)

- Contiguous median: 5088.0 (spread 5061.0–5321.0)
- Paged median: 5758.0 (spread 5672.0–5928.0)

Individual runs: `bench/results/paged_kv_repeat/runs/`.
