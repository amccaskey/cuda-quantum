// #include <cuda.h>

#include "cudaq.h"
#include <stdio.h>

extern "C" {
// void incrementInt(int *i);
void vectorAdd(const float *a, const float *b, float *c, int n);

__qpu__ void callVectorAdd(cudaq::device_ptr a, cudaq::device_ptr b,
                           cudaq::device_ptr c, int n) {
  cudaq::qubit q;
  h(q);
  cudaq::device_call<256, 256>(vectorAdd, a, b, c, n);
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
