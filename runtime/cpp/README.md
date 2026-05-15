# C++ Runtime Skeleton

This directory holds the C++ coordinator/worker path that replaces Python runtime stages.

## Targets
- `runtime_coordinator_cpp`: coordinator process skeleton
- `runtime_worker_cpp`: worker process skeleton with CUDA execution stub

## Build
```bash
cmake -S . -B build
cmake --build build --config Release
```

## Notes
- If gRPC/Protobuf are installed, CMake will generate stubs from `runtime/proto/runtime.proto`.
- If they are not installed, placeholders still build so integration can proceed incrementally.
