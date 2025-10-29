/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

// This header is internal and should NOT be included by user code
// It contains CUDA and cuStateVec specific headers

#include "cudaq/policies/sample/policy.h"

#include <complex>
#include <cuda_runtime.h>
#include <custatevec.h>
#include <cuComplex.h>
#include <vector>

namespace cudaq::simulator::gpu::detail {

/// @brief Internal implementation class that encapsulates CUDA/cuStateVec
/// details
template <typename ScalarType = double>
class state_vector_impl {
public:
  using CudaDataType =
      std::conditional_t<std::is_same_v<ScalarType, float>, cuFloatComplex,
                         cuDoubleComplex>;

  /// @brief Device state vector pointer
  void *device_state_vector_ = nullptr;

  /// @brief cuStateVec handle
  custatevecHandle_t handle_ = nullptr;

  /// @brief Extra workspace for cuStateVec operations
  void *extra_workspace_ = nullptr;
  std::size_t extra_workspace_size_ = 0;

  /// @brief cuStateVec compute and data types
  custatevecComputeType_t compute_type_;
  cudaDataType_t cuda_data_type_;

  /// @brief Whether this instance owns the device vector
  bool owns_device_vector_ = true;

  /// @brief Current state dimension
  std::size_t state_dimension_ = 0;

  /// @brief Number of qubits
  std::size_t num_qubits_ = 0;

  /// @brief Constructor
  state_vector_impl();

  /// @brief Destructor
  ~state_vector_impl();

  /// @brief Initialize state for given number of qubits
  void initialize_state(std::size_t num_qubits);

  /// @brief Allocate and set state to |0...0⟩
  void allocate_zero_state(std::size_t num_qubits);

  /// @brief Allocate state from provided data
  void allocate_from_data(std::size_t num_qubits, const void *host_data,
                          std::size_t data_size);

  /// @brief Apply a gate matrix
  void apply_gate(const std::vector<std::complex<double>> &matrix,
                  const std::vector<int> &controls,
                  const std::vector<int> &targets);

  /// @brief Apply Pauli rotation
  void apply_pauli_rotation(double theta, const std::vector<int> &controls,
                            const std::vector<int> &targets,
                            const std::vector<custatevecPauli_t> &paulis);

  /// @brief Measure a qubit in Z basis
  bool measure_qubit(std::size_t qubit_idx, double random_value);

  /// @brief Reset a qubit to |0⟩
  void reset_qubit(std::size_t qubit_idx, double random_value);

  /// @brief Copy state to host
  void copy_to_host(void *host_ptr, std::size_t size) const;

  /// @brief Copy state from host
  void copy_from_host(const void *host_ptr, std::size_t size);

  /// @brief Get device pointer
  void *get_device_pointer() const { return device_state_vector_; }

  /// @brief Get state dimension
  std::size_t get_state_dimension() const { return state_dimension_; }

  /// @brief Get number of qubits
  std::size_t get_num_qubits() const { return num_qubits_; }

  /// @brief Cleanup
  void deallocate();

  /// @brief Synchronize device
  void synchronize();

  /// @brief Expand state by adding qubits (using Kronecker product)
  void expand_state_with_qubits(std::size_t num_new_qubits, const void *new_qubit_state = nullptr);

  /// @brief Expand state by one qubit (efficient version)
  void expand_state_by_one_qubit();

  sample_result sample_kernel(std::size_t shots);

};

// CUDA kernel declarations (to be defined in .cu file)
extern "C" {
void initialize_device_state_vector_f32(void *device_ptr, std::size_t size);
void initialize_device_state_vector_f64(void *device_ptr, std::size_t size);
void kronprod_f32(uint32_t n_blocks, int32_t threads_per_block, 
                  std::size_t tsize1, const void *arr1,
                  std::size_t tsize2, const void *arr2, void *arr0);
void kronprod_f64(uint32_t n_blocks, int32_t threads_per_block,
                  std::size_t tsize1, const void *arr1,
                  std::size_t tsize2, const void *arr2, void *arr0);
void set_first_n_elements_f32(uint32_t n_blocks, int32_t threads_per_block,
                              void *new_state, void *old_state,
                              std::size_t old_size, std::size_t new_size);
void set_first_n_elements_f64(uint32_t n_blocks, int32_t threads_per_block,
                              void *new_state, void *old_state,
                              std::size_t old_size, std::size_t new_size);
}

} // namespace cudaq::simulator::gpu::detail

