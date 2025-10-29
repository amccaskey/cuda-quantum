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
#include "cudaq/qis/noise_model.h"

#include <complex>
#include <qpp.h>
#include <vector>

namespace cudaq::simulator::cpu {

/// @brief Simulation state for QPP density matrix
struct qpp_dm_state : public cudaq::SimulationState {
  /// @brief The density matrix state
  qpp::cmat state;

  qpp_dm_state(qpp::cmat &&data) : state(std::move(data)) {}
  
  qpp_dm_state(const std::vector<std::size_t> &shape,
               const std::vector<std::complex<double>> &data) {
    if (shape.size() != 2)
      throw std::runtime_error(
          "qpp_dm_state must be created from data with 2D shape.");

    state = Eigen::Map<const qpp::cmat>(data.data(), shape[0], shape[1]);
  }

  std::size_t getNumQubits() const override { return std::log2(state.rows()); }

  std::complex<double> overlap(const cudaq::SimulationState &other) override {
    if (other.getNumTensors() != 1 ||
        (other.getTensor().extents != getTensor().extents))
      throw std::runtime_error("[qpp-dm-state] overlap error - other state "
                               "dimension not equal to this state dimension.");
    
    // Create rho and sigma matrices
    Eigen::MatrixXcd rho = state;
    Eigen::MatrixXcd sigma = Eigen::Map<const Eigen::MatrixXcd>(
        reinterpret_cast<const std::complex<double> *>(other.getTensor().data),
        other.getTensor().extents[0], other.getTensor().extents[1]);

    // For density matrices, fidelity F(rho,sigma) = tr(rho*sigma) + 2 *
    // sqrt(det(rho)*det(sigma))
    auto detprod = rho.determinant() * sigma.determinant();
    return (rho * sigma).trace().real() + 2.0 * std::sqrt(detprod.real());
  }

  std::complex<double>
  getAmplitude(const std::vector<int> &basisState) override {
    if (getNumQubits() != basisState.size())
      throw std::runtime_error(
          "[qpp-dm-state] getAmplitude with an invalid number of bits");
    
    // Convert basis state to index
    const std::size_t idx = std::accumulate(
        std::make_reverse_iterator(basisState.end()),
        std::make_reverse_iterator(basisState.begin()), 0ull,
        [](std::size_t acc, int bit) { return (acc << 1) + bit; });
    
    // Return diagonal element (probability amplitude for mixed states)
    return state(idx, idx);
  }

  Tensor getTensor(std::size_t tensorIdx = 0) const override {
    if (tensorIdx != 0)
      throw std::runtime_error("[qpp-dm-state] invalid tensor requested.");
    return Tensor{
        reinterpret_cast<void *>(const_cast<std::complex<double> *>(state.data())),
        std::vector<std::size_t>{static_cast<std::size_t>(state.rows()),
                                 static_cast<std::size_t>(state.cols())},
        getPrecision()};
  }

  std::vector<Tensor> getTensors() const override { return {getTensor()}; }
  std::size_t getNumTensors() const override { return 1; }

  std::complex<double>
  operator()(std::size_t tensorIdx,
             const std::vector<std::size_t> &indices) override {
    if (tensorIdx != 0)
      throw std::runtime_error("[qpp-dm-state] invalid tensor requested.");
    if (indices.size() != 2)
      throw std::runtime_error("[qpp-dm-state] invalid element extraction.");
    return state(indices[0], indices[1]);
  }

  std::unique_ptr<SimulationState>
  createFromSizeAndPtr(std::size_t size, void *ptr, std::size_t) override {
    return std::make_unique<qpp_dm_state>(
        Eigen::Map<qpp::cmat>(reinterpret_cast<std::complex<double> *>(ptr),
                              std::sqrt(size), std::sqrt(size)));
  }

  void dump(std::ostream &os) const override { os << state << "\n"; }

  precision getPrecision() const override {
    return cudaq::SimulationState::precision::fp64;
  }

  void destroyState() override {
    qpp::cmat k;
    state = k;
  }
  
  bool isArrayLike() const override { return false; }
  bool isDeviceData() const override { return false; }

  void toHost(std::complex<float> *, std::size_t) const override {
    throw std::runtime_error("qpp_dm_state::toHost(float) not implemented");
  }

  void toHost(std::complex<double> *ptr, std::size_t numElements) const override {
    std::copy(state.data(), state.data() + numElements, ptr);
  }
};

} // namespace cudaq::simulator::cpu

namespace cudaq::simulator::cpu::detail {

/// @brief Internal implementation class that encapsulates QPP details
template <typename ScalarType = double>
class density_matrix_impl {
public:
  /// @brief The density matrix state (rho)
  qpp::cmat state_;

  /// @brief Current state dimension
  std::size_t state_dimension_ = 0;

  /// @brief Number of qubits
  std::size_t num_qubits_ = 0;

  /// @brief Constructor
  density_matrix_impl() = default;

  /// @brief Destructor
  ~density_matrix_impl() = default;

  /// @brief Initialize state for given number of qubits
  void initialize_state(std::size_t num_qubits);

  /// @brief Allocate and set state to |0...0⟩⟨0...0|
  void allocate_zero_state(std::size_t num_qubits);

  /// @brief Expand state by adding qubits
  void expand_state_with_qubits(std::size_t num_new_qubits);

  /// @brief Apply a unitary gate to the density matrix
  void apply_gate(const std::vector<std::complex<double>> &matrix,
                  const std::vector<std::size_t> &controls,
                  const std::vector<std::size_t> &targets);

  /// @brief Apply Kraus operators (noise)
  void apply_kraus_ops(const std::vector<cudaq::kraus_op> &ops,
                       const std::vector<std::size_t> &targets);

  /// @brief Measure a qubit in Z basis
  bool measure_qubit(std::size_t qubit_idx, double random_value);

  /// @brief Reset a qubit to |0⟩
  void reset_qubit(std::size_t qubit_idx);

  /// @brief Get state dimension
  std::size_t get_state_dimension() const { return state_dimension_; }

  /// @brief Get number of qubits
  std::size_t get_num_qubits() const { return num_qubits_; }

  /// @brief Get the density matrix
  const qpp::cmat &get_state() const { return state_; }
  
  /// @brief Get mutable density matrix
  qpp::cmat &get_state_mut() { return state_; }
};

} // namespace cudaq::simulator::cpu::detail

