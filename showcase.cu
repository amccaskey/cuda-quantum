#include <cuda.h>
#include <stdio.h>

extern "C" __global__ void add(int a) {
  printf("Hello from the GPU %d\n", a);
}