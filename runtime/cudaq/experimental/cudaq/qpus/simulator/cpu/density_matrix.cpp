/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "density_matrix.h"
#include "density_matrix_impl.h"

#include <algorithm>
#include <iostream>
#include <sstream>

namespace cudaq::simulator::cpu::detail {

// ============================================================================
// density_matrix_impl implementation
// ============================================================================

template <typename ScalarType>
void density_matrix_impl<ScalarType>::initialize_state(std::size_t num_qubits) {
  allocate_zero_state(num_qubits);
}

template <typename ScalarType>
void density_matrix_impl<ScalarType>::allocate_zero_state(std::size_t num_qubits) {
  num_qubits_ = num_qubits;
  state_dimension_ = 1ULL << num_qubits;
  
  // Initialize to |0...0⟩⟨0...0|
  state_ = qpp::cmat::Zero(state_dimension_, state_dimension_);
  state_(0, 0) = 1.0;
}

template <typename ScalarType>
void density_matrix_impl<ScalarType>::expand_state_with_qubits(std::size_t num_new_qubits) {
  if (num_new_qubits == 0) return;

  std::size_t old_dim = state_dimension_;
  std::size_t new_dim = old_dim * (1ULL << num_new_qubits);
  
  // Tensor product: rho ⊗ |0⟩⟨0|
  qpp::cmat new_state = qpp::cmat::Zero(new_dim, new_dim);
  
  for (std::size_t i = 0; i < old_dim; ++i) {
    for (std::size_t j = 0; j < old_dim; ++j) {
      new_state(i * (1ULL << num_new_qubits), j * (1ULL << num_new_qubits)) = state_(i, j);
    }
  }
  
  state_ = new_state;
  num_qubits_ += num_new_qubits;
  state_dimension_ = new_dim;
}

template <typename ScalarType>
void density_matrix_impl<ScalarType>::apply_gate(
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
void density_matrix_impl<ScalarType>::apply_kraus_ops(
    const std::vector<cudaq::kraus_op> &ops,
    const std::vector<std::size_t> &targets) {
  
  // Apply Kraus operators: rho' = Σ_i K_i ρ K_i†
  qpp::cmat new_state = qpp::cmat::Zero(state_dimension_, state_dimension_);
  
  for (const auto &kraus_op : ops) {
    // Convert Kraus operator to Eigen matrix
    std::size_t op_dim = kraus_op.nRows;
    Eigen::MatrixXcd K(op_dim, op_dim);
    for (std::size_t i = 0; i < op_dim; ++i) {
      for (std::size_t j = 0; j < op_dim; ++j) {
        K(i, j) = kraus_op.data[i * op_dim + j];
      }
    }
    
    // Apply K ρ K†
    qpp::cmat K_rho_Kdag = qpp::apply(state_, K, targets);
    new_state += K_rho_Kdag;
  }
  
  state_ = new_state;
}

template <typename ScalarType>
bool density_matrix_impl<ScalarType>::measure_qubit(std::size_t qubit_idx,
                                                    double random_value) {
  // Compute probability of measuring |0⟩
  double prob_0 = 0.0;
  
  for (std::size_t i = 0; i < state_dimension_; ++i) {
    if (((i >> qubit_idx) & 1) == 0) {
      prob_0 += std::real(state_(i, i));
    }
  }

  // Sample outcome
  int outcome = (random_value < prob_0) ? 0 : 1;

  // Collapse the state
  qpp::cmat projector = qpp::cmat::Zero(state_dimension_, state_dimension_);
  for (std::size_t i = 0; i < state_dimension_; ++i) {
    if (((i >> qubit_idx) & 1) == outcome) {
      projector(i, i) = 1.0;
    }
  }

  // Apply: rho' = M ρ M† / P(outcome)
  double prob = (outcome == 0) ? prob_0 : (1.0 - prob_0);
  state_ = (projector * state_ * projector) / prob;

  return outcome == 1;
}

template <typename ScalarType>
void density_matrix_impl<ScalarType>::reset_qubit(std::size_t qubit_idx) {
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
template class density_matrix_impl<double>;
template class density_matrix_impl<float>;

} // namespace cudaq::simulator::cpu::detail

namespace cudaq::simulator::cpu {

// ============================================================================
// density_matrix public interface implementation
// ============================================================================

template <typename ScalarType>
density_matrix<ScalarType>::density_matrix()
    : BaseType(), impl_(std::make_unique<detail::density_matrix_impl<ScalarType>>()) {
  random_engine_.seed(random_device_());
}

template <typename ScalarType>
density_matrix<ScalarType>::density_matrix(const qpu_configuration &config)
    : BaseType(config), impl_(std::make_unique<detail::density_matrix_impl<ScalarType>>()) {
  random_engine_.seed(random_device_());
}

template <typename ScalarType>
density_matrix<ScalarType>::~density_matrix() = default;

template <typename ScalarType>
void density_matrix<ScalarType>::ensure_initialized() {
  if (!impl_) {
    impl_ = std::make_unique<detail::density_matrix_impl<ScalarType>>();
  }
}

template <typename ScalarType>
void density_matrix<ScalarType>::dump_state(std::ostream &os) {
  ensure_initialized();
  os << impl_->get_state() << "\n";
}

template <typename ScalarType>
cudaq::state density_matrix<ScalarType>::get_state() {
  ensure_initialized();
  return cudaq::state(new qpp_dm_state(qpp::cmat(impl_->get_state())));
}

template <typename ScalarType>
cudaq::state density_matrix<ScalarType>::get_state(const state_data &data) {
  auto internal_state = get_internal_state(data);
  return cudaq::state(internal_state->createFromData(data).release());
}

template <typename ScalarType>
std::unique_ptr<cudaq::SimulationState>
density_matrix<ScalarType>::get_internal_state(const state_data &data) {
  // Create a dummy state to use the createFromData method
  qpp::cmat dummy = qpp::cmat::Zero(2, 2);
  dummy(0, 0) = 1.0;
  return std::make_unique<qpp_dm_state>(std::move(dummy));
}

template <typename ScalarType>
simulation_precision density_matrix<ScalarType>::get_precision() const {
  if constexpr (std::is_same_v<ScalarType, float>) {
    return simulation_precision::fp32;
  } else {
    return simulation_precision::fp64;
  }
}

template <typename ScalarType>
std::size_t density_matrix<ScalarType>::allocateQudit(std::size_t numLevels) {
  return allocateQudits(1, numLevels)[0];
}

template <typename ScalarType>
std::vector<std::size_t>
density_matrix<ScalarType>::allocateQudits(std::size_t numQudits,
                                           std::size_t numLevels) {
  if (numLevels != 2) {
    throw std::runtime_error(
        "density_matrix simulator only supports qubits (numLevels=2)");
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
void density_matrix<ScalarType>::deallocate(std::size_t idx) {
  auto it = std::find(allocated_qudits_.begin(), allocated_qudits_.end(), idx);
  if (it != allocated_qudits_.end()) {
    allocated_qudits_.erase(it);
  }
}

template <typename ScalarType>
void density_matrix<ScalarType>::deallocate(const std::vector<std::size_t> &idxs) {
  for (auto idx : idxs) {
    deallocate(idx);
  }
}

template <typename ScalarType>
void density_matrix<ScalarType>::apply_impl(
    const std::vector<std::complex<double>> &matrixRowMajor,
    const std::vector<std::size_t> &controls,
    const std::vector<std::size_t> &targets,
    const traits::operation_metadata &metadata) {
  ensure_initialized();
  impl_->apply_gate(matrixRowMajor, controls, targets);

  // Apply noise if a noise model is set
  if (noise_model_ && !metadata.name.empty()) {
    printf("WE ARE APPLYING NOISE\n");
    auto channels = noise_model_->get_channels(metadata.name, targets, controls,
                                                metadata.parameters);
    for (const auto &channel : channels) {
      impl_->apply_kraus_ops(channel.get_ops(), targets);
    }
  }
}

template <typename ScalarType>
void density_matrix<ScalarType>::reset(std::size_t qidx) {
  ensure_initialized();
  impl_->reset_qubit(qidx);
}

template <typename ScalarType>
void density_matrix<ScalarType>::apply_exp_pauli(
    double theta, const std::vector<std::size_t> &controls,
    const std::vector<std::size_t> &qubitIds, const cudaq::spin_op_term &term) {
  throw std::runtime_error(
      "apply_exp_pauli not yet implemented for density_matrix simulator");
}

template <typename ScalarType>
std::size_t density_matrix<ScalarType>::mz(std::size_t idx,
                                           const std::string regName) {
  ensure_initialized();
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  double random_value = dist(random_engine_);
  return impl_->measure_qubit(idx, random_value) ? 1 : 0;
}

template <typename ScalarType>
void density_matrix<ScalarType>::set_random_seed(std::size_t seed) {
  random_engine_.seed(seed);
}

template <typename ScalarType>
void density_matrix<ScalarType>::apply_noise(
    const std::vector<cudaq::kraus_op> &ops,
    const std::vector<std::size_t> &qubits) {
  ensure_initialized();
  impl_->apply_kraus_ops(ops, qubits);
}

template <typename ScalarType>
sample_result density_matrix<ScalarType>::sample_kernel(std::size_t shots) {
  ensure_initialized();
  
  // Get the density matrix
  const auto &state = impl_->get_state();
  
  // Build list of qubit indices to measure (all qubits)
  std::vector<std::size_t> measured_bits;
  for (std::size_t i = 0; i < num_qubits_; ++i) {
    measured_bits.push_back(i);
  }
  
  // Use QPP's built-in sampling (works for both ket and density matrix)
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
void density_matrix<ScalarType>::reset_state() {
  // Clear qubit tracking
  num_qubits_ = 0;
  allocated_qudits_.clear();
  next_qudit_idx_ = 0;
  
  // Reset the implementation (deallocates and prepares for new execution)
  impl_ = std::make_unique<detail::density_matrix_impl<ScalarType>>();
}

// Explicit template instantiation
template class density_matrix<double>;
template class density_matrix<float>;

} // namespace cudaq::simulator::cpu

