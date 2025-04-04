#include "cudaq.h"
#include <cudaq/algorithms/device.h>

// For this target yaml
// -------------
// description: "Custom target for DGX-Q experimentation."

// name: entangler
// controller-ip: "127.0.0.1"
// config:
//   gen-target-backend: true
//   platform-qpu: driver_qpu
//   link-libs: ["-lcudaq-driver-qpu", "-lcudaq-driver"]
// devices:
//   - name: SharedChannel
//     config:
//       channel: shared_memory
//   - name: "MyA6000GPU_0"
//     config:
//       channel: cuda_channel
//       cuda_device: 0
//   - name: MyA6000GPU_1
//     config:
//       channel: cuda_channel
//       cuda_device: 1
// -------------

// and the add.cpp with the following contents
//
// -------------
// #include <stdio.h>

// extern "C" {
// int add(int i, int j) {
//   printf("libadd.so - add(%d, %d) called.\n", i, j);
//   return i + j;
// }
// }
//
// -------------
//
// We can compile the following code together and have the
// add function called across the shared_memory channel
//
// clang++ add.cpp -o add.o -c && clang++ -shared add.o -o libadd.so
// bin/nvq++ eric.cpp -v --target entangler -ladd
//
// start up the controller in a separate terminal
//
// CUDAQ_LOG_LEVEL=info bin/cudaq-controller --port 8080
//
// Then run the code
//
// CUDAQ_LOG_LEVEL=info ./a.out

extern "C" int add(int i, int j);

__qpu__ auto test() {
  cudaq::qubit q;
  h(q);
  auto res = cudaq::device_call<int>(add, 1, 2);
  return res;
}

int main() { printf("Added %d\n", test()); }
