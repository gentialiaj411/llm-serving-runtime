# C++ Runtime Skeleton

This directory holds the C++ coordinator/worker path that replaces Python runtime stages.

## Targets
- `runtime_coordinator_cpp`: coordinator process skeleton
- `runtime_worker_cpp`: worker process with a functional gRPC `WorkerService::Generate` path when protobuf/gRPC are available, otherwise placeholder heartbeat mode

## Build
```bash
cmake -S . -B build
cmake --build build --config Release
```

## Notes
- If gRPC/Protobuf are installed, CMake will generate stubs from `runtime/proto/runtime.proto`.
- In gRPC-enabled builds, `runtime_worker_cpp` serves `WorkerService::Generate` and returns deterministic token output based on the prompt.
- If gRPC/Protobuf are not installed, placeholders still build so integration can proceed incrementally.
