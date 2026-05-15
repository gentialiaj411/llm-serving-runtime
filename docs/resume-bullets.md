# Resume Bullet Traceability Matrix

| Resume claim | Required metric/artifact | Source path |
|---|---|---|
| Throughput at concurrency | `tokens_per_sec_output`, `concurrency`, model/hw manifest | `bench/results/*.csv`, `bench/results/*manifest.json` |
| p99 latency improvement | `latency_ms_p99` vs baseline run | `bench/results/*.csv` |
| Reliability under failures | success rate under injected faults | `bench/results/*chaos*.json` (Phase 6+) |
| TTFT quality | `ttft_ms_p50/p95/p99` | `bench/results/*.csv` |
