# Runtime Implementation Status

## Implemented now
- C++ coordinator/worker skeleton binaries with optional CUDA/grpc wiring at build-time.
- Python split runtime with production-style control-plane behaviors:
  - health-based worker selection (`/healthz`, background probes)
  - deadline-aware admission/timeout propagation
  - request cancellation endpoint (`/v1/requests/{id}/cancel`)
  - append-only request log (`runtime/logs/requests.jsonl`)
  - replay-safe completion cache for duplicate request IDs with bounded TTL retention
  - startup recovery view (`/admin/recovery`)
- Worker iteration-level continuous batching loop:
  - request admission while active decode is running
  - one-token-per-request decode steps per scheduler tick
  - streaming token events emitted from the decode loop for harness TTFT/ITL measurement
  - cancellation and deadline checks on each decode iteration
- Paged KV block allocator wired into worker admission/release:
  - fixed-size block allocation per request
  - free-list based block reuse on completion/cancel/timeout
  - allocator stats: used/free blocks, occupancy, active allocations, fragmentation ratio
  - metrics endpoints: worker `/metrics`, coordinator `/admin/kv-metrics`

## Verified artifacts
- `bench/results/phase2-smoke-scheduler.csv`
- `bench/results/phase2-smoke-scheduler.manifest.json`
- `bench/results/phase2-kv-paged.csv`
- `bench/results/phase2-kv-paged.manifest.json`
- `bench/results/phase1-transformers-smoke-20260515.csv`
- `bench/results/phase1-transformers-smoke-20260515.manifest.json`
- `bench/results/vllm-smoke-20260515.csv`
- `bench/results/vllm-smoke-20260515.manifest.json`

## Remaining to fully lock architecture
- Port the above control/data-plane behaviors from Python runtime to C++ coordinator/worker.
- Replace HTTP coordinator->worker link with gRPC per proto contract.
- Replace token stub generation with real model inference kernels.

## Benchmark validity notes
- The Python Phase 1/Phase 2 services produce deterministic stub tokens. They are useful for API, scheduling, streaming timing, and fault-path validation, but they are not real model inference benchmarks.
- Phase 1 also has an optional `PHASE1_BACKEND=transformers` mode that performs real model inference through PyTorch/Transformers when those packages and model weights are available.
- Harness manifests include `inference_mode`; report Phase 1/Phase 2 results as `stub_token_generation` unless a real model backend is explicitly wired in.
- Harness manifests include `gpu_metrics_valid`; GPU utilization and memory columns are valid only when that flag is true.
- The committed TinyLlama Phase 1 Transformers and vLLM smoke runs are real model inference on one RTX 5070 Laptop GPU; they validate the harness path, but broader claims still need larger scenario coverage on a stable GPU host.
- CI chaos now runs against a two-worker Phase 2 coordinator path and requires at least one actual worker kill event.
- C++ skeletons are covered by the CI `cmake` configure/build step with CUDA disabled.

## Runtime limitations
- Coordinator streaming retries transport failures before any assistant token is emitted. Once a partial stream has reached the client, retrying on another worker is intentionally not attempted because it would risk duplicate or corrupted output.
- Completion, cancellation, and active-request caches are bounded in memory with TTL/max-size controls. They are process-local and are not a durable production store.
- The coordinator request log rotates locally at `COORDINATOR_REQUEST_LOG_MAX_BYTES` with one retained `.1` file; production deployments should replace this with structured log shipping or durable storage.
