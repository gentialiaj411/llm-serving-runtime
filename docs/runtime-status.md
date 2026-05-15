# Runtime Implementation Status

## Implemented now
- C++ coordinator/worker skeleton binaries with optional CUDA/grpc wiring at build-time.
- Python split runtime with production-style control-plane behaviors:
  - health-based worker selection (`/healthz`, background probes)
  - deadline-aware admission/timeout propagation
  - request cancellation endpoint (`/v1/requests/{id}/cancel`)
  - append-only request log (`runtime/logs/requests.jsonl`)
  - replay-safe completion cache for duplicate request IDs
  - startup recovery view (`/admin/recovery`)
- Worker iteration-level continuous batching loop:
  - request admission while active decode is running
  - one-token-per-request decode steps per scheduler tick
  - cancellation and deadline checks on each decode iteration

## Verified artifacts
- `bench/results/phase2-smoke-scheduler.csv`
- `bench/results/phase2-smoke-scheduler.manifest.json`

## Remaining to fully lock architecture
- Port the above control/data-plane behaviors from Python runtime to C++ coordinator/worker.
- Replace HTTP coordinator->worker link with gRPC per proto contract.
- Replace token stub generation with real model inference kernels.
