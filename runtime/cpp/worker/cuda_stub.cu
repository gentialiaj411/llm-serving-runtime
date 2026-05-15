#include <cuda_runtime.h>

#include <iostream>

__global__ void no_op_kernel() {}

extern "C" void run_cuda_stub() {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess) {
    std::cout << "[worker/cuda] cudaGetDeviceCount failed: " << cudaGetErrorString(err) << "\n";
    return;
  }

  std::cout << "[worker/cuda] visible CUDA devices: " << device_count << "\n";
  if (device_count <= 0) {
    return;
  }

  cudaSetDevice(0);
  no_op_kernel<<<1, 1>>>();
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cout << "[worker/cuda] kernel sync failed: " << cudaGetErrorString(err) << "\n";
    return;
  }
  std::cout << "[worker/cuda] stub kernel completed\n";
}
