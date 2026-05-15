#include <chrono>
#include <iostream>
#include <string>
#include <thread>

int main(int argc, char** argv) {
  std::string bind_addr = "0.0.0.0:50051";
  if (argc > 1) {
    bind_addr = argv[1];
  }

  std::cout << "[coordinator] starting C++ coordinator skeleton on " << bind_addr << "\n";
  std::cout << "[coordinator] TODO: wire gRPC service and request scheduling pipeline.\n";

  // Keep process alive to make integration wiring easy while gRPC service is being connected.
  while (true) {
    std::this_thread::sleep_for(std::chrono::seconds(5));
    std::cout << "[coordinator] heartbeat\n";
  }

  return 0;
}
