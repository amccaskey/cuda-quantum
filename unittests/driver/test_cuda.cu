#include <cuda.h>
#include <stdio.h>

extern "C" {
__global__ void incrementInt(int *i) {
  printf("Calling GPU %d.\n", *i);
  *i += 1;
  printf("Calling GPU %d.\n", *i);
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
