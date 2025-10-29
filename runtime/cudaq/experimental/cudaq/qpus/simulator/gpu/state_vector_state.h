/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/qis/detail/simulation_state.h"
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>

namespace cudaq::simulator::gpu {

/// @brief GPU state vector implementation of SimulationState
///
/// This class encapsulates the GPU device state vector data and provides
/// the SimulationState interface for the experimental runtime. It manages
/// GPU memory and ensures efficient operations without unnecessary data
/// transfers.
template <typename ScalarType>
class gpu_state_vector : public cudaq::SimulationState {
private:
  /// @brief Size of the state vector (2^num_qubits)
  std::size_t size_ = 0;

  /// @brief Device pointer to state data
  void *device_ptr_ = nullptr;

  /// @brief Whether this instance owns the device data
  bool owns_device_ptr_ = true;

  /// @brief Check and set the correct CUDA device
  void check_and_set_device() const;

  /// @brief Extract state amplitudes from device to host
  void extract_values(std::complex<ScalarType> *host_ptr, std::size_t start,
                      std::size_t end) const;

  /// @brief Check if pointer is a device pointer
  bool is_device_pointer(void *ptr) const;

  /// @brief Get CUDA device from pointer
  int device_from_pointer(void *ptr) const;

  /// @brief Copy data to device if needed
  void *maybe_copy_to_device(std::size_t size, void *data_ptr);

public:
  /// @brief Constructor taking device pointer
  gpu_state_vector(std::size_t size, void *device_ptr,
                   bool owns = true);

  /// @brief Return the number of qubits
  std::size_t getNumQubits() const override;

  /// @brief Compute overlap with another state
  std::complex<double> overlap(const cudaq::SimulationState &other) override;

  /// @brief Get amplitude of a basis state
  std::complex<double>
  getAmplitude(const std::vector<int> &basisState) override;

  /// @brief Dump state to output stream
  void dump(std::ostream &os) const override;

  /// @brief Check if data is on device
  bool isDeviceData() const override { return true; }

  /// @brief Get device pointer (non-virtual, GPU-specific)
  const void *getDevicePointer() const { return device_ptr_; }

  /// @brief Get precision
  precision getPrecision() const override;

  /// @brief Create new state from size and pointer
  std::unique_ptr<SimulationState>
  createFromSizeAndPtr(std::size_t size, void *ptr,
                       std::size_t type) override;

  /// @brief Get tensor representation
  Tensor getTensor(std::size_t tensorIdx = 0) const override;

  /// @brief Get all tensors
  std::vector<Tensor> getTensors() const override;

  /// @brief Get number of tensors
  std::size_t getNumTensors() const override { return 1; }

  /// @brief Extract element by index
  std::complex<double>
  operator()(std::size_t tensorIdx,
             const std::vector<std::size_t> &indices) override;

  /// @brief Copy state to host (double precision)
  void toHost(std::complex<double> *host_ptr,
              std::size_t num_elements) const override;

  /// @brief Copy state to host (float precision)
  void toHost(std::complex<float> *host_ptr,
              std::size_t num_elements) const override;

  /// @brief Destroy the state and free GPU memory
  void destroyState() override;

  /// @brief Destructor
  ~gpu_state_vector() override { destroyState(); }
};

// Helper function for inner product computation on GPU
template <typename ScalarType>
struct complex_value {
  ScalarType real;
  ScalarType imaginary;
};

template <typename ScalarType>
complex_value<ScalarType> compute_inner_product(void *device_ptr1,
                                                 void *device_ptr2,
                                                 std::size_t size,
                                                 bool copy_to_device);

} // namespace cudaq::simulator::gpu

