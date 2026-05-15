# Final Notes

## Completed locally
- OpenAI-compatible frontend + coordinator/worker runtime
- Continuous batching (iteration-level scheduler)
- Paged KV allocator with fragmentation metrics
- Deadline-aware admission, cancellation, health-based routing
- Replay-safe request logging + recovery endpoint
- Benchmark harness with CSV + manifest outputs
- Chaos harness with worker kill injection and SLA pass/fail report
- C++ runtime skeleton with protobuf/gRPC generation and build via vcpkg

## Key artifacts produced
- `bench/results/runtime-final-local.csv`
- `bench/results/runtime-final-local.manifest.json`
- `bench/results/chaos-final-local.json`
- `docs/reports/final-report-local.json`

## Remaining external artifact (blocked by local GPU arch compatibility)
- `bench/results/vllm-baseline-*.csv` on a supported vLLM host (A100/H100/compatible Ada)

## Why blocked here
- vLLM 0.8.5 + available torch binaries do not support this GPU compute capability (`sm_120`), causing kernel image load failure at engine init.
