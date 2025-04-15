#include <cuda.h>
#include <stdio.h>

extern "C" {
__global__ void incrementInt(int *i) {
  printf("Calling GPU %d.\n", *i);
  *i += 1;
  printf("Calling GPU %d.\n", *i);
}
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
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
