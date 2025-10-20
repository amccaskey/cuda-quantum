#include "cuda_runtime.h"
#include <stdio.h>

namespace cudaq::simulator::mqpu::details {
int get_gpu_device_count() {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  return device_count;
}
void set_gpu_device(int i) { cudaSetDevice(static_cast<int>(i)); }

} // namespace cudaq::simulation::mqpu::details
