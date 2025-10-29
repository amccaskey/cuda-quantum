/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

// This header is internal and should NOT be included by user code
// It contains QPP specific headers

#include "cudaq/qis/detail/simulation_state.h"

#include <complex>
#include <qpp.h>
#include <vector>

namespace cudaq::simulator::cpu {

/// @brief Simulation state for QPP state vector
struct qpp_state : public cudaq::SimulationState {
  /// @brief The state vector
  qpp::ket state;

  qpp_state(qpp::ket &&data) : state(std::move(data)) {}
  
  qpp_state(const std::vector<std::size_t> &shape,
            const std::vector<std::complex<double>> &data) {
    if (shape.size() != 1)
      throw std::runtime_error(
          "qpp_state must be created from data with 1D shape.");

    state = Eigen::Map<const qpp::ket>(data.data(), shape[0]);
  }

  std::size_t getNumQubits() const override { return std::log2(state.size()); }

  std::complex<double> overlap(const cudaq::SimulationState &other) override {
    if (other.getNumTensors() != 1 ||
        (other.getTensor().extents != getTensor().extents))
      throw std::runtime_error("[qpp-state] overlap error - other state "
                               "dimension not equal to this state dimension.");
    
    // Create ket vectors
    Eigen::VectorXcd psi1 = state;
    Eigen::VectorXcd psi2 = Eigen::Map<const Eigen::VectorXcd>(
        reinterpret_cast<const std::complex<double> *>(other.getTensor().data),
        other.getTensor().extents[0]);

    // Compute <psi1|psi2>
    return psi1.dot(psi2);
  }

  std::complex<double>
  getAmplitude(const std::vector<int> &basisState) override {
    if (getNumQubits() != basisState.size())
      throw std::runtime_error(
          "[qpp-state] getAmplitude with an invalid number of bits");
    
    // Convert basis state to index
    const std::size_t idx = std::accumulate(
        std::make_reverse_iterator(basisState.end()),
        std::make_reverse_iterator(basisState.begin()), 0ull,
        [](std::size_t acc, int bit) { return (acc << 1) + bit; });
    
    return state(idx);
  }

  Tensor getTensor(std::size_t tensorIdx = 0) const override {
    if (tensorIdx != 0)
      throw std::runtime_error("[qpp-state] invalid tensor requested.");
    return Tensor{
        reinterpret_cast<void *>(const_cast<std::complex<double> *>(state.data())),
        std::vector<std::size_t>{static_cast<std::size_t>(state.size())},
        getPrecision()};
  }

  std::vector<Tensor> getTensors() const override { return {getTensor()}; }
  std::size_t getNumTensors() const override { return 1; }

  std::complex<double>
  operator()(std::size_t tensorIdx,
             const std::vector<std::size_t> &indices) override {
    if (tensorIdx != 0)
      throw std::runtime_error("[qpp-state] invalid tensor requested.");
    if (indices.size() != 1)
      throw std::runtime_error("[qpp-state] invalid element extraction.");
    return state(indices[0]);
  }

  std::unique_ptr<SimulationState>
  createFromSizeAndPtr(std::size_t size, void *ptr, std::size_t) override {
    return std::make_unique<qpp_state>(
        Eigen::Map<qpp::ket>(reinterpret_cast<std::complex<double> *>(ptr),
                             size));
  }

  void dump(std::ostream &os) const override { os << state << "\n"; }

  precision getPrecision() const override {
    return cudaq::SimulationState::precision::fp64;
  }

  void destroyState() override {
    qpp::ket k;
    state = k;
  }
  
  bool isArrayLike() const override { return false; }
  bool isDeviceData() const override { return false; }

  void toHost(std::complex<float> *, std::size_t) const override {
    throw std::runtime_error("qpp_state::toHost(float) not implemented");
  }

  void toHost(std::complex<double> *ptr, std::size_t numElements) const override {
    std::copy(state.data(), state.data() + numElements, ptr);
  }
};

} // namespace cudaq::simulator::cpu

namespace cudaq::simulator::cpu::detail {

/// @brief Internal implementation class that encapsulates QPP details
template <typename ScalarType = double>
class state_vector_impl {
public:
  /// @brief The state vector (ket)
  qpp::ket state_;

  /// @brief Current state dimension
  std::size_t state_dimension_ = 0;

  /// @brief Number of qubits
  std::size_t num_qubits_ = 0;

  /// @brief Constructor
  state_vector_impl() = default;

  /// @brief Destructor
  ~state_vector_impl() = default;

  /// @brief Initialize state for given number of qubits
  void initialize_state(std::size_t num_qubits);

  /// @brief Allocate and set state to |0...0⟩
  void allocate_zero_state(std::size_t num_qubits);

  /// @brief Expand state by adding qubits
  void expand_state_with_qubits(std::size_t num_new_qubits);

  /// @brief Apply a unitary gate
  void apply_gate(const std::vector<std::complex<double>> &matrix,
                  const std::vector<std::size_t> &controls,
                  const std::vector<std::size_t> &targets);

  /// @brief Measure a qubit in Z basis
  bool measure_qubit(std::size_t qubit_idx, double random_value);

  /// @brief Reset a qubit to |0⟩
  void reset_qubit(std::size_t qubit_idx);

  /// @brief Get state dimension
  std::size_t get_state_dimension() const { return state_dimension_; }

  /// @brief Get number of qubits
  std::size_t get_num_qubits() const { return num_qubits_; }

  /// @brief Get the state vector
  const qpp::ket &get_state() const { return state_; }
  
  /// @brief Get mutable state vector
  qpp::ket &get_state_mut() { return state_; }
};

} // namespace cudaq::simulator::cpu::detail

