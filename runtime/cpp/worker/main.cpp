#include <chrono>
#include <iostream>
#include <string>
#include <thread>

#if RUNTIME_ENABLE_CUDA_STUB
extern "C" void run_cuda_stub();
#endif

#if RUNTIME_ENABLE_GRPC_PROTO
#include "worker/grpc_worker_service.h"
#endif

int main(int argc, char** argv) {
  std::string bind_addr = "0.0.0.0:50052";
  if (argc > 1) {
    bind_addr = argv[1];
  }

  std::cout << "[worker] starting C++ worker skeleton on " << bind_addr << "\n";
#if RUNTIME_ENABLE_CUDA_STUB
  std::cout << "[worker] running CUDA execution stub...\n";
  run_cuda_stub();
#else
  std::cout << "[worker] CUDA stub disabled (no CUDA compiler/toolchain detected).\n";
#endif

#if RUNTIME_ENABLE_GRPC_PROTO
  runtime::worker::GrpcWorkerService service;
  auto server = runtime::worker::BuildGrpcWorkerServer(bind_addr, &service);
  if (!server) {
    std::cerr << "[worker] failed to start gRPC WorkerService on " << bind_addr << "\n";
    return 1;
  }
  std::cout << "[worker] gRPC WorkerService::Generate active on " << bind_addr << "\n";
  server->Wait();
  return 0;
#else
  std::cout << "[worker] gRPC stubs unavailable; running placeholder heartbeat mode.\n";

  while (true) {
    std::this_thread::sleep_for(std::chrono::seconds(5));
    std::cout << "[worker] heartbeat\n";
  }

  return 0;
#endif
}
