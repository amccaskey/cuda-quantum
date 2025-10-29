/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/host_config.h"
#include "cudaq/policies/sample/policy.h"
#include "cudaq/qis/state.h"
#include "cudaq/qpu.h"
#include "cudaq/spin_op.h"
#include "cudaq/traits/local.h"
#include "cudaq/traits/simulator.h"

#include <complex>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace cudaq::simulator::gpu {

// Forward declarations to hide implementation details
namespace detail {
template <typename ScalarType>
class state_vector_impl;
}

/// @brief GPU-accelerated state vector simulator using NVIDIA cuStateVec
///
/// This simulator leverages the NVIDIA cuStateVec library for high-performance
/// quantum circuit simulation on GPUs. It follows the experimental trait-based
/// design pattern, implementing both simulator and local_trait interfaces.
///
/// Example usage:
/// ```cpp
/// state_vector qpu;
/// auto results = cudaq::launch(qpu, cudaq::sample_policy{}, my_kernel);
/// ```
template <typename ScalarType = double>
class state_vector : public qpu<state_vector<ScalarType>,
                                traits::simulator<state_vector<ScalarType>>,
                                traits::local_trait<state_vector<ScalarType>>> {
  using BaseType =
      qpu<state_vector<ScalarType>, traits::simulator<state_vector<ScalarType>>,
          traits::local_trait<state_vector<ScalarType>>>;

private:
  /// @brief Pointer to implementation (PIMPL pattern to hide CUDA headers)
  std::unique_ptr<detail::state_vector_impl<ScalarType>> impl_;

  /// @brief Random number generator for measurements
  std::random_device random_device_;
  std::mt19937 random_engine_;

  /// @brief Number of currently allocated qubits
  std::size_t num_qubits_ = 0;

  /// @brief Qudit levels tracker (maps qudit index to number of levels)
  std::vector<std::size_t> qudit_levels_;

  /// @brief Track allocated qudit indices
  std::vector<std::size_t> allocated_qudits_;

  /// @brief Next available qudit index
  std::size_t next_qudit_idx_ = 0;

  /// @brief Noise model pointer (optional)
  const cudaq::noise_model *noise_model_ = nullptr;

  /// @brief Initialize the implementation
  void ensure_initialized();

  /// @brief Helper to get total state dimension
  std::size_t get_state_dimension() const;

  using BaseType::m_configuration;

public:
  /// @brief Default constructor
  state_vector();

  /// @brief Constructor with configuration
  explicit state_vector(const qpu_configuration &config);

  /// @brief Destructor
  ~state_vector();

  /// @brief Get the name of this QPU
  std::string name() const { return "gpu::state_vector"; }

  // ============================================================================
  // Simulator Trait Interface
  // ============================================================================

  /// @brief Dump the current state to output stream
  void dump_state(std::ostream &os);

  /// @brief Get the current quantum state
  cudaq::state get_state();

  /// @brief Get state from user-provided data
  cudaq::state get_state(const state_data &data);

  /// @brief Get internal simulation state
  std::unique_ptr<cudaq::SimulationState>
  get_internal_state(const state_data &data);

  /// @brief Get the simulation precision
  simulation_precision get_precision() const;

  /// @brief Allocate a single qudit
  std::size_t allocateQudit(std::size_t numLevels = 2);

  /// @brief Allocate multiple qudits
  std::vector<std::size_t> allocateQudits(std::size_t numQudits,
                                          std::size_t numLevels = 2);

  /// @brief Allocate qudits from raw state data
  std::vector<std::size_t> allocateQudits(std::size_t numQudits,
                                          std::size_t numLevels,
                                          const void *state,
                                          simulation_precision precision);

  /// @brief Allocate qudits from SimulationState
  std::vector<std::size_t> allocateQudits(std::size_t numQudits,
                                          std::size_t numLevels,
                                          const SimulationState *state);

  /// @brief Deallocate a single qudit
  void deallocate(std::size_t idx);

  /// @brief Deallocate multiple qudits
  void deallocate(const std::vector<std::size_t> &idxs);

  /// @brief Apply a quantum gate (implementation - called by base trait)
  /// Note: applyControlRegion and applyAdjointRegion are now handled by the
  /// base simulator trait, so derived classes only need to implement apply_impl
  void apply_impl(const std::vector<std::complex<double>> &matrixRowMajor,
                  const std::vector<std::size_t> &controls,
                  const std::vector<std::size_t> &targets,
                  const traits::operation_metadata &metadata);

  /// @brief Reset a qubit to |0⟩
  void reset(std::size_t qidx);

  /// @brief Apply exponential of Pauli operator
  void apply_exp_pauli(double theta, const std::vector<std::size_t> &controls,
                       const std::vector<std::size_t> &qubitIds,
                       const cudaq::spin_op_term &term);

  /// @brief Measure a qubit in Z basis
  std::size_t mz(std::size_t idx, const std::string regName = "");

  /// @brief Set random seed for reproducible measurements
  void set_random_seed(std::size_t seed);

  // ============================================================================
  // Noise Model Support
  // ============================================================================

  /// @brief Set the noise model
  void set_noise(const cudaq::noise_model *model) { noise_model_ = model; }

  /// @brief Get the current noise model
  const cudaq::noise_model *get_noise() const { return noise_model_; }

  /// @brief Apply noise channel to qubits (stub for now)
  void apply_noise(const std::vector<cudaq::kraus_op> &ops,
                   const std::vector<std::size_t> &qubits) {
    // TODO: Implement Kraus operator application
    // For now, this is a no-op
  }

  // ============================================================================
  // Local Trait Interface
  // ============================================================================

  /// @brief Launch with sample policy
  template <typename QuantumKernel, typename... Args>
  auto launch(const cudaq::sample_policy &policy, QuantumKernel &&kernel,
              Args &&...args) {
    // Set up the kernel API for this thread
    cudaq::set_kernel_api(*this);

    // Execute the kernel (which will call back into our simulator methods)
    kernel(std::forward<Args>(args)...);

    auto result = sample_kernel(policy.shots);

    // Clean up API
    cudaq::clear_kernel_api();

    return result;
  }

private:
  sample_result sample_kernel(std::size_t shots);
};

} // namespace cudaq::simulator::gpu
