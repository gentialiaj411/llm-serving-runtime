#include <chrono>
#include <iostream>
#include <string>
#include <thread>

#if RUNTIME_ENABLE_CUDA_STUB
extern "C" void run_cuda_stub();
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
  std::cout << "[worker] TODO: wire gRPC WorkerService::Generate and naive batch queue.\n";

  while (true) {
    std::this_thread::sleep_for(std::chrono::seconds(5));
    std::cout << "[worker] heartbeat\n";
  }

  return 0;
}
