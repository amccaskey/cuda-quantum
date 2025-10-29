/****************************************************************-*- CUDA -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cstdint>

#define THRDS_ARRAY_PRODUCT 256

// Operator overloads for complex arithmetic
__device__ __host__ inline cuDoubleComplex operator*(cuDoubleComplex a,
                                                      cuDoubleComplex b) {
  return cuCmul(a, b);
}

__device__ __host__ inline cuDoubleComplex operator+(cuDoubleComplex a,
                                                      cuDoubleComplex b) {
  return cuCadd(a, b);
}

__device__ __host__ inline cuFloatComplex operator*(cuFloatComplex a,
                                                     cuFloatComplex b) {
  return cuCmulf(a, b);
}

__device__ __host__ inline cuFloatComplex operator+(cuFloatComplex a,
                                                     cuFloatComplex b) {
  return cuCaddf(a, b);
}

// Kernel to initialize state vector to |0...0⟩ (single precision)
__global__ void initialize_state_kernel_f32(cuFloatComplex *state,
                                             size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    if (idx == 0) {
      state[idx] = make_cuFloatComplex(1.0f, 0.0f);
    } else {
      state[idx] = make_cuFloatComplex(0.0f, 0.0f);
    }
  }
}

// Kernel to initialize state vector to |0...0⟩ (double precision)
__global__ void initialize_state_kernel_f64(cuDoubleComplex *state,
                                             size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    if (idx == 0) {
      state[idx] = make_cuDoubleComplex(1.0, 0.0);
    } else {
      state[idx] = make_cuDoubleComplex(0.0, 0.0);
    }
  }
}

// Kronecker product kernel (float)
template <typename CudaDataType>
__global__ void kronprod_kernel(size_t tsize1, const CudaDataType *arr1,
                                size_t tsize2, const CudaDataType *arr2,
                                CudaDataType *arr0) {
  __shared__ CudaDataType lbuf[THRDS_ARRAY_PRODUCT + 1],
      rbuf[THRDS_ARRAY_PRODUCT];
  size_t _ib, _in, _jb, _jn, _tx, _jc, _ja;

  _tx = (size_t)threadIdx.x;
  for (_jb = blockIdx.y * THRDS_ARRAY_PRODUCT; _jb < tsize2;
       _jb += gridDim.y * THRDS_ARRAY_PRODUCT) {
    if (_jb + THRDS_ARRAY_PRODUCT > tsize2) {
      _jn = tsize2 - _jb;
    } else {
      _jn = THRDS_ARRAY_PRODUCT;
    }

    if (_tx < _jn)
      rbuf[_tx] = arr2[_jb + _tx];

    for (_ib = blockIdx.x * THRDS_ARRAY_PRODUCT; _ib < tsize1;
         _ib += gridDim.x * THRDS_ARRAY_PRODUCT) {
      if (_ib + THRDS_ARRAY_PRODUCT > tsize1) {
        _in = tsize1 - _ib;
      } else {
        _in = THRDS_ARRAY_PRODUCT;
      }

      if (_tx < _in)
        lbuf[_tx] = arr1[_ib + _tx];

      __syncthreads();
      for (_jc = 0; _jc < _jn; _jc++) {
        if (_tx < _in) {
          _ja = (_jb + _jc) * tsize1 + (_ib + _tx);
          arr0[_ja] = arr0[_ja] + lbuf[_tx] * rbuf[_jc];
        }
      }
      __syncthreads();
    }
  }
}

// Set first N elements kernel
template <typename CudaDataType>
__global__ void set_first_n_elements_kernel(CudaDataType *sv,
                                            const CudaDataType *__restrict__ sv2,
                                            int64_t N, int64_t total_size) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < total_size) {
    if (i < N) {
      sv[i] = sv2[i];
    } else {
      sv[i].x = 0.0;
      sv[i].y = 0.0;
    }
  }
}

extern "C" {

void initialize_device_state_vector_f32(void *device_ptr, size_t size) {
  constexpr int threads_per_block = 256;
  int num_blocks = (size + threads_per_block - 1) / threads_per_block;

  initialize_state_kernel_f32<<<num_blocks, threads_per_block>>>(
      static_cast<cuFloatComplex *>(device_ptr), size);

  cudaDeviceSynchronize();
}

void initialize_device_state_vector_f64(void *device_ptr, size_t size) {
  constexpr int threads_per_block = 256;
  int num_blocks = (size + threads_per_block - 1) / threads_per_block;

  initialize_state_kernel_f64<<<num_blocks, threads_per_block>>>(
      static_cast<cuDoubleComplex *>(device_ptr), size);

  cudaDeviceSynchronize();
}

void kronprod_f32(uint32_t n_blocks, int32_t threads_per_block, size_t tsize1,
                  const void *arr1, size_t tsize2, const void *arr2,
                  void *arr0) {
  dim3 blocks(n_blocks, n_blocks);
  kronprod_kernel<<<blocks, threads_per_block>>>(
      tsize1, static_cast<const cuFloatComplex *>(arr1), tsize2,
      static_cast<const cuFloatComplex *>(arr2),
      static_cast<cuFloatComplex *>(arr0));
}

void kronprod_f64(uint32_t n_blocks, int32_t threads_per_block, size_t tsize1,
                  const void *arr1, size_t tsize2, const void *arr2,
                  void *arr0) {
  dim3 blocks(n_blocks, n_blocks);
  kronprod_kernel<<<blocks, threads_per_block>>>(
      tsize1, static_cast<const cuDoubleComplex *>(arr1), tsize2,
      static_cast<const cuDoubleComplex *>(arr2),
      static_cast<cuDoubleComplex *>(arr0));
}

void set_first_n_elements_f32(uint32_t n_blocks, int32_t threads_per_block,
                              void *new_state, void *old_state, size_t old_size,
                              size_t new_size) {
  set_first_n_elements_kernel<<<n_blocks, threads_per_block>>>(
      static_cast<cuFloatComplex *>(new_state),
      static_cast<const cuFloatComplex *>(old_state), old_size, new_size);
}

void set_first_n_elements_f64(uint32_t n_blocks, int32_t threads_per_block,
                              void *new_state, void *old_state, size_t old_size,
                              size_t new_size) {
  set_first_n_elements_kernel<<<n_blocks, threads_per_block>>>(
      static_cast<cuDoubleComplex *>(new_state),
      static_cast<const cuDoubleComplex *>(old_state), old_size, new_size);
}

} // extern "C"

