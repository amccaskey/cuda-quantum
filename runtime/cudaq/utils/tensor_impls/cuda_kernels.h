// cuda_tensor_kernels.cuh
#pragma once
#include <complex>
#include <cuda_runtime.h>

namespace cudaq {
namespace kernels {

// Kernel declarations
template <typename Scalar>
void elementwise_add_kernel(int threads, int blocks, const Scalar *a,
                            const Scalar *b, Scalar *c, size_t n);
template <typename Scalar>
void elementwise_multiply_kernel(int threads, int blocks, const Scalar *a,
                                 const Scalar *b, Scalar *c, size_t n);

template <typename Scalar>
void elementwise_modulo_kernel(int threads, int blocks, const Scalar *a,
                               const Scalar *b, Scalar *c, size_t n);
template <typename Scalar>
void scalar_modulo_kernel(int threads, int blocks, const Scalar *a,
                          Scalar value, Scalar *c, size_t n);

} // namespace kernels
} // namespace cudaq
