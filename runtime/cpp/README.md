# C++ Runtime Skeleton

This directory holds the C++ coordinator/worker **skeleton** that may eventually replace Python runtime stages. The Python phase2 stack is the primary runtime today.

## Targets
- `runtime_coordinator_cpp`: coordinator process skeleton
- `runtime_worker_cpp`: worker process skeleton with optional gRPC `WorkerService::Generate` stub when protobuf/gRPC are available, otherwise placeholder heartbeat mode

## Build
```bash
cmake -S . -B build
cmake --build build --config Release
```

## Notes
- If gRPC/Protobuf are installed, CMake will generate stubs from `runtime/proto/runtime.proto`.
- In gRPC-enabled builds, `runtime_worker_cpp` exposes a **skeleton** `WorkerService::Generate` that returns deterministic stub tokens derived from the prompt — **not** transformer inference.
- **No `Generate` RPC has ever executed in-repo** (no `bench/results/cpp-grpc-smoke.json`). See `docs/adr/0011-cpp-grpc-worker-skeleton.md`.
- If gRPC/Protobuf are not installed, placeholders still build so integration can proceed incrementally.
