#include <cuda_runtime.h>

#include <iostream>

int main() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  // Memory bandwidth in GB/s
  float mem_bandwidth = (prop.memoryClockRate * 1000.0) *
                        (prop.memoryBusWidth / 8.0) * 2.0 / 1.0e9;

  std::cout << "Memory Clock Rate (MHz): " << prop.memoryClockRate / 1000.0
            << std::endl;
  std::cout << "Memory Bus Width (bits): " << prop.memoryBusWidth << std::endl;
  std::cout << "Total Memory Bandwidth (GB/s): " << mem_bandwidth << std::endl;

  return 0;
}
