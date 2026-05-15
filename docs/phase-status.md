# Phase Status

## Phase 1 (functional baseline)
- Implemented: OpenAI-compatible `/v1/chat/completions` server.
- Implemented: request-by-request serving (no batching).
- Artifact: `bench/results/phase1-baseline.csv`
- Artifact: `bench/results/phase1-baseline.manifest.json`

## Phase 2 (functional architecture reset)
- Implemented: split coordinator/worker services.
- Implemented: naive batching in worker queue loop.
- Artifact: `bench/results/phase2-baseline.csv`
- Artifact: `bench/results/phase2-baseline.manifest.json`

## Gaps vs locked stack
- Current Phase 2 services are Python stand-ins for fast iteration.
- Strict locked-stack compliance still requires C++ coordinator and C++/CUDA worker with gRPC hot path.
- Harness currently measures endpoint behavior and core latency/throughput columns, but GPU metrics are placeholder zeros until GPU telemetry integration.
