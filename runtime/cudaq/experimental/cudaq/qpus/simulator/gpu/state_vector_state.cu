/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "state_vector_state.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <thrust/complex.h>
#include <thrust/device_ptr.h>
#include <thrust/inner_product.h>

#define HANDLE_CUDA_ERROR(x)                                                   \
  {                                                                            \
    const auto err = x;                                                        \
    if (err != cudaSuccess) {                                                  \
      throw std::runtime_error(                                                \
          std::string("[gpu_state_vector] CUDA error: ") +                    \
          cudaGetErrorString(err) + " in " + __FUNCTION__ + " (line " +       \
          std::to_string(__LINE__) + ")");                                     \
    }                                                                          \
  }

namespace cudaq::simulator::gpu {

// ============================================================================
// Helper Functions
// ============================================================================

template <typename ScalarType>
complex_value<ScalarType> compute_inner_product(void *device_ptr1,
                                                 void *device_ptr2,
                                                 std::size_t size,
                                                 bool copy_to_device) {
  using ThrustComplex = thrust::complex<ScalarType>;

  thrust::device_ptr<ThrustComplex> d_ptr1(
      static_cast<ThrustComplex *>(device_ptr1));

  ThrustComplex *d_ptr2_raw = nullptr;
  bool allocated = false;

  if (copy_to_device) {
    // Copy from host to device
    HANDLE_CUDA_ERROR(cudaMalloc(&d_ptr2_raw, size * sizeof(ThrustComplex)));
    HANDLE_CUDA_ERROR(cudaMemcpy(d_ptr2_raw, device_ptr2,
                                 size * sizeof(ThrustComplex),
                                 cudaMemcpyHostToDevice));
    allocated = true;
  } else {
    d_ptr2_raw = static_cast<ThrustComplex *>(device_ptr2);
  }

  thrust::device_ptr<ThrustComplex> d_ptr2(d_ptr2_raw);

  // Compute inner product using thrust
  auto init = ThrustComplex(0.0, 0.0);
  auto result = thrust::inner_product(
      d_ptr1, d_ptr1 + size, d_ptr2, init,
      thrust::plus<ThrustComplex>(),
      [] __device__ (const ThrustComplex &a, const ThrustComplex &b) {
        return thrust::conj(a) * b;
      });

  if (allocated) {
    HANDLE_CUDA_ERROR(cudaFree(d_ptr2_raw));
  }

  return {result.real(), result.imag()};
}

// Explicit instantiations
template complex_value<float> compute_inner_product<float>(void *, void *,
                                                            std::size_t, bool);
template complex_value<double>
compute_inner_product<double>(void *, void *, std::size_t, bool);

// ============================================================================
// gpu_state_vector Implementation
// ============================================================================

template <typename ScalarType>
gpu_state_vector<ScalarType>::gpu_state_vector(std::size_t size,
                                                void *device_ptr, bool owns)
    : size_(size), device_ptr_(device_ptr), owns_device_ptr_(owns) {}

template <typename ScalarType>
void gpu_state_vector<ScalarType>::check_and_set_device() const {
  int dev = 0;
  HANDLE_CUDA_ERROR(cudaGetDevice(&dev));
  auto current_device = device_from_pointer(device_ptr_);
  if (dev != current_device) {
    HANDLE_CUDA_ERROR(cudaSetDevice(current_device));
  }
}

template <typename ScalarType>
void gpu_state_vector<ScalarType>::extract_values(
    std::complex<ScalarType> *host_ptr, std::size_t start,
    std::size_t end) const {
  check_and_set_device();
  HANDLE_CUDA_ERROR(cudaMemcpy(
      host_ptr,
      reinterpret_cast<std::complex<ScalarType> *>(device_ptr_) + start,
      (end - start) * sizeof(std::complex<ScalarType>),
      cudaMemcpyDeviceToHost));
}

template <typename ScalarType>
bool gpu_state_vector<ScalarType>::is_device_pointer(void *ptr) const {
  cudaPointerAttributes attributes;
  HANDLE_CUDA_ERROR(cudaPointerGetAttributes(&attributes, ptr));
  return attributes.type > 1;
}

template <typename ScalarType>
int gpu_state_vector<ScalarType>::device_from_pointer(void *ptr) const {
  cudaPointerAttributes attributes;
  HANDLE_CUDA_ERROR(cudaPointerGetAttributes(&attributes, ptr));
  return attributes.device;
}

template <typename ScalarType>
void *gpu_state_vector<ScalarType>::maybe_copy_to_device(std::size_t size,
                                                          void *data_ptr) {
  if (is_device_pointer(data_ptr))
    return data_ptr;

  std::complex<ScalarType> *ptr = nullptr;
  HANDLE_CUDA_ERROR(
      cudaMalloc((void **)&ptr, size * sizeof(std::complex<ScalarType>)));
  HANDLE_CUDA_ERROR(cudaMemcpy(ptr, data_ptr,
                               size * sizeof(std::complex<ScalarType>),
                               cudaMemcpyHostToDevice));
  return reinterpret_cast<void *>(ptr);
}

template <typename ScalarType>
std::size_t gpu_state_vector<ScalarType>::getNumQubits() const {
  return std::log2(size_);
}

template <typename ScalarType>
std::complex<double>
gpu_state_vector<ScalarType>::overlap(const cudaq::SimulationState &other) {
  if (getTensor().extents != other.getTensor().extents) {
    throw std::runtime_error(
        "[gpu_state_vector] overlap error - dimension mismatch");
  }

  if (other.getPrecision() != getPrecision()) {
    throw std::runtime_error("[gpu_state_vector] overlap error - precision "
                             "mismatch");
  }

  int current_dev;
  cudaGetDevice(&current_dev);
  auto data_dev = device_from_pointer(device_ptr_);
  if (current_dev != data_dev) {
    cudaSetDevice(data_dev);
  }

  bool need_copy = !is_device_pointer(other.getTensor().data);

  if (!need_copy &&
      device_from_pointer(device_ptr_) !=
          device_from_pointer(other.getTensor().data)) {
    throw std::runtime_error(
        "[gpu_state_vector] overlap on different GPU devices not supported");
  }

  auto result =
      compute_inner_product<ScalarType>(device_ptr_, other.getTensor().data,
                                        size_, need_copy);

  return std::abs(std::complex<ScalarType>(result.real, result.imaginary));
}

template <typename ScalarType>
std::complex<double>
gpu_state_vector<ScalarType>::getAmplitude(const std::vector<int> &basisState) {
  if (getNumQubits() != basisState.size()) {
    throw std::runtime_error(
        "[gpu_state_vector] getAmplitude with invalid basis state size");
  }

  if (std::any_of(basisState.begin(), basisState.end(),
                  [](int x) { return x != 0 && x != 1; })) {
    throw std::runtime_error(
        "[gpu_state_vector] getAmplitude with invalid basis state values");
  }

  // Convert basis state to index
  const std::size_t idx =
      std::accumulate(std::make_reverse_iterator(basisState.end()),
                      std::make_reverse_iterator(basisState.begin()), 0ull,
                      [](std::size_t acc, int bit) { return (acc << 1) + bit; });

  std::complex<ScalarType> value;
  extract_values(&value, idx, idx + 1);
  return {value.real(), value.imag()};
}

template <typename ScalarType>
void gpu_state_vector<ScalarType>::dump(std::ostream &os) const {
  std::vector<std::complex<ScalarType>> tmp(size_);
  HANDLE_CUDA_ERROR(cudaMemcpy(tmp.data(), device_ptr_,
                               size_ * sizeof(std::complex<ScalarType>),
                               cudaMemcpyDeviceToHost));
  for (auto &t : tmp) {
    os << t << "\n";
  }
}

template <typename ScalarType>
typename SimulationState::precision
gpu_state_vector<ScalarType>::getPrecision() const {
  if constexpr (std::is_same_v<ScalarType, float>) {
    return SimulationState::precision::fp32;
  }
  return SimulationState::precision::fp64;
}

template <typename ScalarType>
std::unique_ptr<SimulationState>
gpu_state_vector<ScalarType>::createFromSizeAndPtr(std::size_t size,
                                                    void *ptr,
                                                    std::size_t type) {
  // If type < 2, it's a vector (owned), otherwise it's a pair (not owned)
  bool we_own = type < 2;
  void *dev_ptr = maybe_copy_to_device(size, ptr);
  return std::make_unique<gpu_state_vector<ScalarType>>(size, dev_ptr,
                                                         we_own);
}

template <typename ScalarType>
typename SimulationState::Tensor
gpu_state_vector<ScalarType>::getTensor(std::size_t tensorIdx) const {
  if (tensorIdx != 0) {
    throw std::runtime_error("[gpu_state_vector] invalid tensor index");
  }
  return Tensor{device_ptr_, std::vector<std::size_t>{size_}, getPrecision()};
}

template <typename ScalarType>
std::vector<typename SimulationState::Tensor>
gpu_state_vector<ScalarType>::getTensors() const {
  return {getTensor()};
}

template <typename ScalarType>
std::complex<double> gpu_state_vector<ScalarType>::operator()(
    std::size_t tensorIdx, const std::vector<std::size_t> &indices) {
  if (tensorIdx != 0) {
    throw std::runtime_error("[gpu_state_vector] invalid tensor index");
  }

  if (indices.size() != 1) {
    throw std::runtime_error("[gpu_state_vector] invalid index extraction");
  }

  auto idx = indices[0];
  std::complex<ScalarType> value;
  extract_values(&value, idx, idx + 1);
  return {value.real(), value.imag()};
}

template <typename ScalarType>
void gpu_state_vector<ScalarType>::toHost(std::complex<double> *host_ptr,
                                           std::size_t num_elements) const {
  if constexpr (std::is_same_v<ScalarType, float>) {
    throw std::runtime_error("[gpu_state_vector] precision mismatch: state is "
                             "FP32 but FP64 buffer requested");
  }

  if (num_elements != size_) {
    throw std::runtime_error(
        "[gpu_state_vector] toHost with incorrect number of elements");
  }

  extract_values(reinterpret_cast<std::complex<ScalarType> *>(host_ptr), 0,
                 size_);
}

template <typename ScalarType>
void gpu_state_vector<ScalarType>::toHost(std::complex<float> *host_ptr,
                                           std::size_t num_elements) const {
  if constexpr (std::is_same_v<ScalarType, double>) {
    throw std::runtime_error("[gpu_state_vector] precision mismatch: state is "
                             "FP64 but FP32 buffer requested");
  }

  if (num_elements != size_) {
    throw std::runtime_error(
        "[gpu_state_vector] toHost with incorrect number of elements");
  }

  extract_values(reinterpret_cast<std::complex<ScalarType> *>(host_ptr), 0,
                 size_);
}

template <typename ScalarType>
void gpu_state_vector<ScalarType>::destroyState() {
  if (!owns_device_ptr_)
    return;

  if (device_ptr_) {
    int current_dev;
    cudaGetDevice(&current_dev);
    auto device = device_from_pointer(device_ptr_);
    if (current_dev != device) {
      cudaSetDevice(device);
    }

    HANDLE_CUDA_ERROR(cudaFree(device_ptr_));
    device_ptr_ = nullptr;
  }
}

// Explicit template instantiations
template class gpu_state_vector<float>;
template class gpu_state_vector<double>;

} // namespace cudaq::simulator::gpu

