#include <iostream>

using namespace std;

int main() {
  int device_count;
  cudaGetDeviceCount(&device_count);

  cout << "Device count: " << device_count << endl;

  for (int i = 0; i < device_count; i++) {
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, i);
    cout << "Device " << i << ": " << device_prop.name << endl;

    cout << "Compute capability: " << device_prop.major << "."
         << device_prop.minor << endl;
    cout << "Clock rate: " << device_prop.clockRate / 1e6 << " GHz" << endl;
    cout << "Device memory: " << device_prop.totalGlobalMem << " bytes" << endl;
    cout << "Shared memory per block: " << device_prop.sharedMemPerBlock
         << " bytes" << endl;
    cout << "Registers per block: " << device_prop.regsPerBlock << endl;

    cout << "Warp size: " << device_prop.warpSize << endl;
    cout << "Maximum threads per block: " << device_prop.maxThreadsPerBlock
         << endl;
    cout << "Maximum blocks per multiprocessor: "
         << device_prop.maxBlocksPerMultiProcessor << endl;
    cout << "Maximum threads per multiprocessor: "
         << device_prop.maxThreadsPerMultiProcessor << endl;
    cout << "Multiprocessor count: " << device_prop.multiProcessorCount << endl;

    cout << endl;
  }

  return 0;
}