# Bench Budget

## Pricing assumptions
- `gpu_hour_usd`: set per run manifest (example: A100-80GB on-demand rate)
- `wall_clock_hours`: measured harness runtime
- `cost_usd = gpu_hour_usd * gpu_count * wall_clock_hours`
- `est_dollars_per_million_output_tokens = cost_usd / (output_tokens / 1_000_000)`

## Cumulative tracker
| date_utc | run_id | gpu_type | gpu_count | gpu_hour_usd | wall_clock_hours | cost_usd | notes |
|---|---|---:|---:|---:|---:|---:|---|
|  |  |  |  |  |  |  |  |
