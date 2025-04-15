// #include <cuda.h>

#include "cudaq.h"
#include <stdio.h>

extern "C" {
// __global__
void incrementInt(int *i);
//  {
//   printf("Calling GPU.\n");
//   *i += 1;
// }

__qpu__ void callGpu(cudaq::device_ptr in) {
  cudaq::qubit q;
  h(q);
  cudaq::device_call<1, 1>(incrementInt, in);
}
}

// int main() {

//   int i = 2;
//   int *intPtr = nullptr;

//   cudaMalloc(&intPtr, sizeof(int));

//   cudaMemcpy(intPtr, &i, sizeof(int),
//   cudaMemcpyKind::cudaMemcpyHostToDevice);

//   incrementInt<<<1, 1>>>(intPtr);

//   cudaMemcpy(&i, intPtr, sizeof(int),
//   cudaMemcpyKind::cudaMemcpyDeviceToHost);

//   printf("RESULT %d\n", i);

//   cudaFree(intPtr);

//   return 0;
// }
