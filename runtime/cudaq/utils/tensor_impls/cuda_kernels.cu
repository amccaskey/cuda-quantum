/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cuComplex.h"
#include "cuda_kernels.h"

__device__ __host__ cuDoubleComplex operator*(cuDoubleComplex a,
                                              cuDoubleComplex b) {
  return cuCmul(a, b);
}
__device__ __host__ cuDoubleComplex operator+(cuDoubleComplex a,
                                              cuDoubleComplex b) {
  return cuCadd(a, b);
}
__device__ __host__ cuFloatComplex operator*(cuFloatComplex a,
                                             cuFloatComplex b) {
  return cuCmulf(a, b);
}
__device__ __host__ cuFloatComplex operator+(cuFloatComplex a,
                                             cuFloatComplex b) {
  return cuCaddf(a, b);
}

__device__ inline cuDoubleComplex operator%(const cuDoubleComplex &a,
                                            const cuDoubleComplex &b) {
  double real = fmod(cuCreal(a), cuCreal(b));
  double imag = fmod(cuCimag(a), cuCimag(b));
  return make_cuDoubleComplex(real, imag);
}

__device__ inline cuFloatComplex operator%(const cuFloatComplex &a,
                                           const cuFloatComplex &b) {
  float real = fmodf(cuCrealf(a), cuCrealf(b));
  float imag = fmodf(cuCimagf(a), cuCimagf(b));
  return make_cuFloatComplex(real, imag);
}

__device__ inline cuFloatComplex operator%(const std::complex<float> &a,
                                           const std::complex<float> &b) {
  auto *aa =
      reinterpret_cast<cuFloatComplex *>(const_cast<std::complex<float> *>(&a));
  auto *bb =
      reinterpret_cast<cuFloatComplex *>(const_cast<std::complex<float> *>(&b));
  return *aa % *bb;
  //   float real = fmodf(cuCrealf(a), cuCrealf(b));
  //   float imag = fmodf(cuCimagf(a), cuCimagf(b));
  //   return make_cuFloatComplex(real, imag);
}

__device__ inline cuFloatComplex operator%(const std::complex<double> &a,
                                           const std::complex<double> &b) {
  auto *aa = reinterpret_cast<cuFloatComplex *>(
      const_cast<std::complex<double> *>(&a));
  auto *bb = reinterpret_cast<cuFloatComplex *>(
      const_cast<std::complex<double> *>(&b));
  return *aa % *bb;
  //   float real = fmodf(cuCrealf(a), cuCrealf(b));
  //   float imag = fmodf(cuCimagf(a), cuCimagf(b));
  //   return make_cuFloatComplex(real, imag);
}

__device__ inline std::complex<float> toStd(cuFloatComplex &a) {
  return {cuCrealf(a), cuCimagf(a)};
}

namespace cudaq {
namespace kernels {

template <typename Scalar>
__global__ void elementwise_add_kernel(const Scalar *a, const Scalar *b,
                                       Scalar *c, size_t n) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < n) {
    c[tid] = a[tid] + b[tid];
  }
}

template <typename Scalar>
void elementwise_add_kernel(int threads, int blocks, const Scalar *a,
                            const Scalar *b, Scalar *c, size_t n) {
  elementwise_add_kernel<<<blocks, threads>>>(a, b, c, n);
}

template <typename Scalar>
__global__ void elementwise_multiply_kernel(const Scalar *a, const Scalar *b,
                                            Scalar *c, size_t n) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < n) {
    c[tid] = a[tid] * b[tid];
  }
}

template <typename Scalar>
void elementwise_multiply_kernel(int threads, int blocks, const Scalar *a,
                                 const Scalar *b, Scalar *c, size_t n) {
  elementwise_multiply_kernel<<<blocks, threads>>>(a, b, c, n);
}

// template <typename Scalar>
// __global__ void slice_kernel(const Scalar* input, Scalar* output,
//                            const size_t* input_shape, const size_t*
//                            output_shape, const slice* slices, const int rank)
//                            {
//     // Get global thread ID
//     const int tid = blockDim.x * blockIdx.x + threadIdx.x;

//     // Calculate total output size
//     size_t output_size = 1;
//     for (int i = 0; i < rank; i++) {
//         output_size *= output_shape[i];
//     }

//     if (tid < output_size) {
//         // Convert linear index to multi-dimensional indices
//         size_t remaining = tid;
//         size_t input_index = 0;
//         size_t stride = 1;

//         for (int i = rank - 1; i >= 0; i--) {
//             const size_t output_idx = remaining % output_shape[i];
//             remaining /= output_shape[i];

//             // Calculate input index using slice information
//             const size_t start = slices[i].start ? *slices[i].start : 0;
//             const size_t step = slices[i].step ? *slices[i].step : 1;
//             const size_t input_idx = start + output_idx * step;

//             input_index += input_idx * stride;
//             stride *= input_shape[i];
//         }

//         output[tid] = input[input_index];
//     }
// }

template <typename Scalar>
__global__ void elementwise_modulo_kernel(const Scalar *a, const Scalar *b,
                                          Scalar *c, size_t n) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < n) {
    // if constexpr (std::is_integral_v<Scalar>) {
    //   c[tid] = a[tid] % b[tid];
    // } else if constexpr (std::is_same_v<Scalar, std::complex<float>> ||
    //                      std::is_same_v<Scalar, std::complex<double>>) {
    //   c[tid] = toStd(a[tid] % b[tid]);
    // } else {
    //   c[tid] = fmod(a[tid], b[tid]);
    // }
  }
}

template <typename Scalar>
void elementwise_modulo_kernel(int threads, int blocks, const Scalar *a,
                               const Scalar *b, Scalar *c, size_t n) {
  elementwise_modulo_kernel<<<blocks, threads>>>(a, b, c, n);
}

template <typename Scalar>
__global__ void scalar_modulo_kernel(const Scalar *a, Scalar value, Scalar *c,
                                     size_t n) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < n) {
    // if constexpr (std::is_integral_v<Scalar> ||
    //               std::is_same_v<Scalar, std::complex<float>> ||
    //               std::is_same_v<Scalar, std::complex<double>>) {
    //   c[tid] = a[tid] % value;
    // } else {
    //   c[tid] = fmod(a[tid], value);
    // }
  }
}

template <typename Scalar>
void scalar_modulo_kernel(int threads, int blocks, const Scalar *a,
                          Scalar value, Scalar *c, size_t n) {
  scalar_modulo_kernel<<<blocks, threads>>>(a, value, c, n);
}

// Explicit template instantiations
#define INSTANTIATE_KERNELS(TYPE)                                              \
  template __global__ void elementwise_add_kernel<TYPE>(                       \
      const TYPE *, const TYPE *, TYPE *, size_t);                             \
  template void elementwise_add_kernel<TYPE>(int, int, const TYPE *,           \
                                             const TYPE *, TYPE *, size_t);    \
  template __global__ void elementwise_multiply_kernel<TYPE>(                  \
      const TYPE *, const TYPE *, TYPE *, size_t);                             \
  template void elementwise_multiply_kernel<TYPE>(                             \
      int, int, const TYPE *, const TYPE *, TYPE *, size_t);                   \
  template __global__ void elementwise_modulo_kernel<TYPE>(                    \
      const TYPE *, const TYPE *, TYPE *, size_t);                             \
  template void elementwise_modulo_kernel<TYPE>(int, int, const TYPE *,        \
                                                const TYPE *, TYPE *, size_t); \
  template __global__ void scalar_modulo_kernel<TYPE>(const TYPE *, TYPE,      \
                                                      TYPE *, size_t);         \
  template void scalar_modulo_kernel<TYPE>(int, int, const TYPE *, TYPE,       \
                                           TYPE *, size_t);

INSTANTIATE_KERNELS(int)
INSTANTIATE_KERNELS(uint8_t)
INSTANTIATE_KERNELS(float)
INSTANTIATE_KERNELS(double)
INSTANTIATE_KERNELS(cuFloatComplex)
INSTANTIATE_KERNELS(cuDoubleComplex)
INSTANTIATE_KERNELS(size_t)

#undef INSTANTIATE_KERNELS

} // namespace kernels
} // namespace cudaq