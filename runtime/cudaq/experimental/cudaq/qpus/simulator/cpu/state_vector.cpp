/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "state_vector.h"
#include "state_vector_impl.h"

#include <algorithm>
#include <iostream>
#include <sstream>

namespace cudaq::simulator::cpu::detail {

// ============================================================================
// state_vector_impl implementation
// ============================================================================

template <typename ScalarType>
void state_vector_impl<ScalarType>::initialize_state(std::size_t num_qubits) {
  allocate_zero_state(num_qubits);
}

template <typename ScalarType>
void state_vector_impl<ScalarType>::allocate_zero_state(std::size_t num_qubits) {
  num_qubits_ = num_qubits;
  state_dimension_ = 1ULL << num_qubits;
  
  // Initialize to |0...0⟩
  state_ = qpp::ket::Zero(state_dimension_);
  state_(0) = 1.0;
}

template <typename ScalarType>
void state_vector_impl<ScalarType>::expand_state_with_qubits(std::size_t num_new_qubits) {
  if (num_new_qubits == 0) return;

  std::size_t old_dim = state_dimension_;
  std::size_t new_dim = old_dim * (1ULL << num_new_qubits);
  
  // Tensor product: |ψ⟩ ⊗ |0⟩
  qpp::ket new_state = qpp::ket::Zero(new_dim);
  
  for (std::size_t i = 0; i < old_dim; ++i) {
    new_state(i * (1ULL << num_new_qubits)) = state_(i);
  }
  
  state_ = new_state;
  num_qubits_ += num_new_qubits;
  state_dimension_ = new_dim;
}

template <typename ScalarType>
void state_vector_impl<ScalarType>::apply_gate(
    const std::vector<std::complex<double>> &matrix,
    const std::vector<std::size_t> &controls,
    const std::vector<std::size_t> &targets) {
  
  // Convert to QPP format
  std::size_t target_dim = 1ULL << targets.size();
  Eigen::MatrixXcd gate_matrix(target_dim, target_dim);
  for (std::size_t i = 0; i < target_dim; ++i) {
    for (std::size_t j = 0; j < target_dim; ++j) {
      gate_matrix(i, j) = matrix[i * target_dim + j];
    }
  }

  // Apply controlled or uncontrolled gate
  if (controls.empty()) {
    state_ = qpp::apply(state_, gate_matrix, targets);
  } else {
    state_ = qpp::applyCTRL(state_, gate_matrix, controls, targets);
  }
}

template <typename ScalarType>
bool state_vector_impl<ScalarType>::measure_qubit(std::size_t qubit_idx,
                                                   double random_value) {
  // Compute probability of measuring |0⟩
  double prob_0 = 0.0;
  
  for (std::size_t i = 0; i < state_dimension_; ++i) {
    if (((i >> qubit_idx) & 1) == 0) {
      prob_0 += std::norm(state_(i));
    }
  }

  // Sample outcome
  int outcome = (random_value < prob_0) ? 0 : 1;

  // Collapse the state
  double norm = std::sqrt(outcome == 0 ? prob_0 : (1.0 - prob_0));
  for (std::size_t i = 0; i < state_dimension_; ++i) {
    if (((i >> qubit_idx) & 1) != outcome) {
      state_(i) = 0.0;
    } else {
      state_(i) /= norm;
    }
  }

  return outcome == 1;
}

template <typename ScalarType>
void state_vector_impl<ScalarType>::reset_qubit(std::size_t qubit_idx) {
  // Measure the qubit
  double dummy_random = 0.5;
  bool result = measure_qubit(qubit_idx, dummy_random);
  
  // If it's |1⟩, flip it to |0⟩
  if (result) {
    std::vector<std::complex<double>> x_gate = {0, 1, 1, 0};
    apply_gate(x_gate, {}, {qubit_idx});
  }
}

// Explicit template instantiation
template class state_vector_impl<double>;
template class state_vector_impl<float>;

} // namespace cudaq::simulator::cpu::detail

namespace cudaq::simulator::cpu {

// ============================================================================
// state_vector public interface implementation
// ============================================================================

template <typename ScalarType>
state_vector<ScalarType>::state_vector()
    : BaseType(), impl_(std::make_unique<detail::state_vector_impl<ScalarType>>()) {
  random_engine_.seed(random_device_());
}

template <typename ScalarType>
state_vector<ScalarType>::state_vector(const qpu_configuration &config)
    : BaseType(config), impl_(std::make_unique<detail::state_vector_impl<ScalarType>>()) {
  random_engine_.seed(random_device_());
}

template <typename ScalarType>
state_vector<ScalarType>::~state_vector() = default;

template <typename ScalarType>
void state_vector<ScalarType>::ensure_initialized() {
  if (!impl_) {
    impl_ = std::make_unique<detail::state_vector_impl<ScalarType>>();
  }
}

template <typename ScalarType>
void state_vector<ScalarType>::dump_state(std::ostream &os) {
  ensure_initialized();
  os << impl_->get_state() << "\n";
}

template <typename ScalarType>
cudaq::state state_vector<ScalarType>::get_state() {
  ensure_initialized();
  return cudaq::state(new qpp_state(qpp::ket(impl_->get_state())));
}

template <typename ScalarType>
cudaq::state state_vector<ScalarType>::get_state(const state_data &data) {
  auto internal_state = get_internal_state(data);
  return cudaq::state(internal_state->createFromData(data).release());
}

template <typename ScalarType>
std::unique_ptr<cudaq::SimulationState>
state_vector<ScalarType>::get_internal_state(const state_data &data) {
  // Create a dummy state to use the createFromData method
  qpp::ket dummy = qpp::ket::Zero(2);
  dummy(0) = 1.0;
  return std::make_unique<qpp_state>(std::move(dummy));
}

template <typename ScalarType>
simulation_precision state_vector<ScalarType>::get_precision() const {
  if constexpr (std::is_same_v<ScalarType, float>) {
    return simulation_precision::fp32;
  } else {
    return simulation_precision::fp64;
  }
}

template <typename ScalarType>
std::size_t state_vector<ScalarType>::allocateQudit(std::size_t numLevels) {
  return allocateQudits(1, numLevels)[0];
}

template <typename ScalarType>
std::vector<std::size_t>
state_vector<ScalarType>::allocateQudits(std::size_t numQudits,
                                         std::size_t numLevels) {
  if (numLevels != 2) {
    throw std::runtime_error(
        "state_vector simulator only supports qubits (numLevels=2)");
  }

  ensure_initialized();

  std::vector<std::size_t> allocated;
  for (std::size_t i = 0; i < numQudits; ++i) {
    std::size_t idx = next_qudit_idx_++;
    allocated_qudits_.push_back(idx);
    allocated.push_back(idx);
  }

  // Update state
  if (num_qubits_ == 0) {
    impl_->initialize_state(numQudits);
  } else {
    impl_->expand_state_with_qubits(numQudits);
  }

  num_qubits_ += numQudits;

  return allocated;
}

template <typename ScalarType>
void state_vector<ScalarType>::deallocate(std::size_t idx) {
  auto it = std::find(allocated_qudits_.begin(), allocated_qudits_.end(), idx);
  if (it != allocated_qudits_.end()) {
    allocated_qudits_.erase(it);
  }
}

template <typename ScalarType>
void state_vector<ScalarType>::deallocate(const std::vector<std::size_t> &idxs) {
  for (auto idx : idxs) {
    deallocate(idx);
  }
}

template <typename ScalarType>
void state_vector<ScalarType>::apply_impl(
    const std::vector<std::complex<double>> &matrixRowMajor,
    const std::vector<std::size_t> &controls,
    const std::vector<std::size_t> &targets,
    const traits::operation_metadata &metadata) {
  ensure_initialized();
  impl_->apply_gate(matrixRowMajor, controls, targets);
}

template <typename ScalarType>
void state_vector<ScalarType>::reset(std::size_t qidx) {
  ensure_initialized();
  impl_->reset_qubit(qidx);
}

template <typename ScalarType>
void state_vector<ScalarType>::apply_exp_pauli(
    double theta, const std::vector<std::size_t> &controls,
    const std::vector<std::size_t> &qubitIds, const cudaq::spin_op_term &term) {
  throw std::runtime_error(
      "apply_exp_pauli not yet implemented for state_vector simulator");
}

template <typename ScalarType>
std::size_t state_vector<ScalarType>::mz(std::size_t idx,
                                         const std::string regName) {
  ensure_initialized();
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  double random_value = dist(random_engine_);
  return impl_->measure_qubit(idx, random_value) ? 1 : 0;
}

template <typename ScalarType>
void state_vector<ScalarType>::set_random_seed(std::size_t seed) {
  random_engine_.seed(seed);
}

template <typename ScalarType>
sample_result state_vector<ScalarType>::sample_kernel(std::size_t shots) {
  ensure_initialized();
  
  // Get the state vector
  const auto &state = impl_->get_state();
  
  // Build list of qubit indices to measure (all qubits)
  std::vector<std::size_t> measured_bits;
  for (std::size_t i = 0; i < num_qubits_; ++i) {
    measured_bits.push_back(i);
  }
  
  // Use QPP's built-in sampling
  auto sample_result_qpp = qpp::sample(shots, state, measured_bits, 2);
  
  // Convert to CUDA-Q ExecutionResult
  ExecutionResult exec_result;
  std::stringstream bitstream;
  
  for (const auto &[result, count] : sample_result_qpp) {
    // Convert vector of bits to bitstring
    for (const auto &bit : result) {
      bitstream << bit;
    }
    
    auto bitstring = bitstream.str();
    exec_result.appendResult(bitstring, count);
    
    // Reset the stream
    bitstream.str("");
    bitstream.clear();
  }
  
  return sample_result(exec_result);
}

template <typename ScalarType>
void state_vector<ScalarType>::reset_state() {
  // Clear qubit tracking
  num_qubits_ = 0;
  allocated_qudits_.clear();
  next_qudit_idx_ = 0;
  
  // Reset the implementation (deallocates and prepares for new execution)
  impl_ = std::make_unique<detail::state_vector_impl<ScalarType>>();
}

// Explicit template instantiation
template class state_vector<double>;
template class state_vector<float>;

} // namespace cudaq::simulator::cpu

