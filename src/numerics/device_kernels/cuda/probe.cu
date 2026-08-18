#include <format>

#include "utilities/check.hpp"

__global__ void probe_kernel() {}

void check_probe_kernel() {
  probe_kernel<<<1, 1>>>();
  auto err = cudaGetLastError();
  if(err == cudaSuccess) {
    err = cudaDeviceSynchronize();
  }
  
  std::string tip = "";
  if(err == cudaErrorNoKernelImageForDevice) {
    cudaDeviceProp dev{};
    cudaGetDeviceProperties(&dev, 0);
    tip = std::format("\nTry compiling with -DCUDA_ARCH={}.", dev.major * 10 + dev.minor);
  }
  sfqmc::utils::check(err == cudaSuccess, "Problem launching CUDA kernels: {}: {}{}", cudaGetErrorName(err), cudaGetErrorString(err), tip);
}
