/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "state_vector.h"
#include "cudaq/utils/type_traits.h"
#include "qpu.h"
#include "state_vector_impl.h"
#include "state_vector_state.h"

#include "cudaq/qis/state.h"
#include "cudaq/spin_op.h"

#include <cassert>
#include <cmath>
#include <stdexcept>

#define HANDLE_CUDA_ERROR(x)                                                   \
  {                                                                            \
    const auto err = x;                                                        \
    if (err != cudaSuccess) {                                                  \
      throw std::runtime_error(                                                \
          std::string("[gpu::state_vector] CUDA error: ") +                    \
          cudaGetErrorString(err) + " in " + __FUNCTION__ + " (line " +        \
          std::to_string(__LINE__) + ")");                                     \
    }                                                                          \
  }

#define HANDLE_CUSTATEVEC_ERROR(x)                                             \
  {                                                                            \
    const auto err = x;                                                        \
    if (err != CUSTATEVEC_STATUS_SUCCESS) {                                    \
      throw std::runtime_error(                                                \
          std::string("[gpu::state_vector] cuStateVec error: ") +              \
          custatevecGetErrorString(err) + " in " + __FUNCTION__ + " (line " +  \
          std::to_string(__LINE__) + ")");                                     \
    }                                                                          \
  }

namespace cudaq::simulator::gpu {

// ============================================================================
// state_vector_impl Implementation
// ============================================================================

namespace detail {

template <typename ScalarType>
state_vector_impl<ScalarType>::state_vector_impl() {
  if constexpr (std::is_same_v<ScalarType, float>) {
    compute_type_ = CUSTATEVEC_COMPUTE_32F;
    cuda_data_type_ = CUDA_C_32F;
  } else {
    compute_type_ = CUSTATEVEC_COMPUTE_64F;
    cuda_data_type_ = CUDA_C_64F;
  }

  // Initialize CUDA
  HANDLE_CUDA_ERROR(cudaFree(0));
}

template <typename ScalarType>
state_vector_impl<ScalarType>::~state_vector_impl() {
  deallocate();
}

template <typename ScalarType>
void state_vector_impl<ScalarType>::deallocate() {
  if (handle_) {
    HANDLE_CUSTATEVEC_ERROR(custatevecDestroy(handle_));
    handle_ = nullptr;
  }

  if (device_state_vector_ && owns_device_vector_) {
    HANDLE_CUDA_ERROR(cudaFree(device_state_vector_));
    device_state_vector_ = nullptr;
  }

  if (extra_workspace_) {
    HANDLE_CUDA_ERROR(cudaFree(extra_workspace_));
    extra_workspace_ = nullptr;
    extra_workspace_size_ = 0;
  }

  state_dimension_ = 0;
  num_qubits_ = 0;
}

template <typename ScalarType>
void state_vector_impl<ScalarType>::initialize_state(std::size_t num_qubits) {
  num_qubits_ = num_qubits;
  state_dimension_ = 1ULL << num_qubits;

  if (!handle_) {
    HANDLE_CUSTATEVEC_ERROR(custatevecCreate(&handle_));
  }
}

template <typename ScalarType>
void state_vector_impl<ScalarType>::allocate_zero_state(
    std::size_t num_qubits) {
  initialize_state(num_qubits);

  if (!device_state_vector_) {
    HANDLE_CUDA_ERROR(cudaMalloc(&device_state_vector_,
                                 state_dimension_ * sizeof(CudaDataType)));
    owns_device_vector_ = true;
  }

  // Initialize to |0...0⟩ state
  if constexpr (std::is_same_v<ScalarType, float>) {
    initialize_device_state_vector_f32(device_state_vector_, state_dimension_);
  } else {
    initialize_device_state_vector_f64(device_state_vector_, state_dimension_);
  }

  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
}

template <typename ScalarType>
void state_vector_impl<ScalarType>::allocate_from_data(std::size_t num_qubits,
                                                       const void *host_data,
                                                       std::size_t data_size) {
  initialize_state(num_qubits);

  if (!device_state_vector_) {
    HANDLE_CUDA_ERROR(cudaMalloc(&device_state_vector_,
                                 state_dimension_ * sizeof(CudaDataType)));
    owns_device_vector_ = true;
  }

  // Copy from host to device
  HANDLE_CUDA_ERROR(cudaMemcpy(device_state_vector_, host_data,
                               data_size * sizeof(CudaDataType),
                               cudaMemcpyHostToDevice));
}

template <typename ScalarType>
void state_vector_impl<ScalarType>::apply_gate(
    const std::vector<std::complex<double>> &matrix,
    const std::vector<int> &controls, const std::vector<int> &targets) {

  // Convert matrix to appropriate type
  std::vector<CudaDataType> matrix_cuda;
  if constexpr (std::is_same_v<ScalarType, float>) {
    matrix_cuda.reserve(matrix.size());
    for (const auto &elem : matrix) {
      matrix_cuda.push_back(make_cuFloatComplex(elem.real(), elem.imag()));
    }
  } else {
    matrix_cuda.reserve(matrix.size());
    for (const auto &elem : matrix) {
      matrix_cuda.push_back(make_cuDoubleComplex(elem.real(), elem.imag()));
    }
  }

  // Apply the matrix using cuStateVec
  HANDLE_CUSTATEVEC_ERROR(custatevecApplyMatrix(
      handle_, device_state_vector_, cuda_data_type_, num_qubits_,
      matrix_cuda.data(), cuda_data_type_, CUSTATEVEC_MATRIX_LAYOUT_ROW,
      /*adjoint=*/0, targets.data(), targets.size(), controls.data(),
      /*controlBitValues=*/nullptr, controls.size(),
      /*computeType=*/compute_type_, extra_workspace_, extra_workspace_size_));
}

template <typename ScalarType>
void state_vector_impl<ScalarType>::apply_pauli_rotation(
    double theta, const std::vector<int> &controls,
    const std::vector<int> &targets,
    const std::vector<custatevecPauli_t> &paulis) {

  HANDLE_CUSTATEVEC_ERROR(custatevecApplyPauliRotation(
      handle_, device_state_vector_, cuda_data_type_, num_qubits_, theta,
      paulis.data(), targets.data(), targets.size(), controls.data(),
      /*controlBitValues=*/nullptr, controls.size()));
}

template <typename ScalarType>
bool state_vector_impl<ScalarType>::measure_qubit(std::size_t qubit_idx,
                                                  double random_value) {
  const int basis_bits[] = {static_cast<int>(qubit_idx)};
  int parity;

  HANDLE_CUSTATEVEC_ERROR(custatevecMeasureOnZBasis(
      handle_, device_state_vector_, cuda_data_type_, num_qubits_, &parity,
      basis_bits, /*nBits=*/1, random_value,
      CUSTATEVEC_COLLAPSE_NORMALIZE_AND_ZERO));

  return parity == 1;
}

template <typename ScalarType>
void state_vector_impl<ScalarType>::reset_qubit(std::size_t qubit_idx,
                                                double random_value) {
  bool measured = measure_qubit(qubit_idx, random_value);
  // If measured 1, apply X to flip back to 0
  if (measured) {
    // Apply X gate
    std::vector<std::complex<double>> x_matrix = {
        {0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}};
    std::vector<int> targets = {static_cast<int>(qubit_idx)};
    std::vector<int> controls;
    apply_gate(x_matrix, controls, targets);
  }
}

template <typename ScalarType>
void state_vector_impl<ScalarType>::copy_to_host(void *host_ptr,
                                                 std::size_t size) const {
  HANDLE_CUDA_ERROR(cudaMemcpy(host_ptr, device_state_vector_,
                               size * sizeof(CudaDataType),
                               cudaMemcpyDeviceToHost));
}

template <typename ScalarType>
void state_vector_impl<ScalarType>::copy_from_host(const void *host_ptr,
                                                   std::size_t size) {
  HANDLE_CUDA_ERROR(cudaMemcpy(device_state_vector_, host_ptr,
                               size * sizeof(CudaDataType),
                               cudaMemcpyHostToDevice));
}

template <typename ScalarType>
void state_vector_impl<ScalarType>::synchronize() {
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
}

template <typename ScalarType>
void state_vector_impl<ScalarType>::expand_state_with_qubits(
    std::size_t num_new_qubits, const void *new_qubit_state) {
  std::size_t previous_dimension = state_dimension_;
  std::size_t new_qubit_dimension = 1ULL << num_new_qubits;

  num_qubits_ += num_new_qubits;
  state_dimension_ = 1ULL << num_qubits_;

  // Allocate temporary state for new qubits
  void *temp_new_qubits;
  HANDLE_CUDA_ERROR(
      cudaMalloc(&temp_new_qubits, new_qubit_dimension * sizeof(CudaDataType)));

  if (new_qubit_state == nullptr) {
    // Initialize to |0...0⟩
    if constexpr (std::is_same_v<ScalarType, float>) {
      initialize_device_state_vector_f32(temp_new_qubits, new_qubit_dimension);
    } else {
      initialize_device_state_vector_f64(temp_new_qubits, new_qubit_dimension);
    }
  } else {
    // Copy provided state
    HANDLE_CUDA_ERROR(cudaMemcpy(temp_new_qubits, new_qubit_state,
                                 new_qubit_dimension * sizeof(CudaDataType),
                                 cudaMemcpyHostToDevice));
  }

  // Allocate new state vector for result
  void *new_device_state_vector;
  HANDLE_CUDA_ERROR(cudaMalloc(&new_device_state_vector,
                               state_dimension_ * sizeof(CudaDataType)));
  HANDLE_CUDA_ERROR(cudaMemset(new_device_state_vector, 0,
                               state_dimension_ * sizeof(CudaDataType)));

  // Compute Kronecker product
  constexpr int32_t threads_per_block = 256;
  uint32_t n_blocks = 4; // For 2D grid

  if constexpr (std::is_same_v<ScalarType, float>) {
    kronprod_f32(n_blocks, threads_per_block, previous_dimension,
                 device_state_vector_, new_qubit_dimension, temp_new_qubits,
                 new_device_state_vector);
  } else {
    kronprod_f64(n_blocks, threads_per_block, previous_dimension,
                 device_state_vector_, new_qubit_dimension, temp_new_qubits,
                 new_device_state_vector);
  }

  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

  // Free old state and temp state
  HANDLE_CUDA_ERROR(cudaFree(device_state_vector_));
  HANDLE_CUDA_ERROR(cudaFree(temp_new_qubits));

  device_state_vector_ = new_device_state_vector;
}

template <typename ScalarType>
void state_vector_impl<ScalarType>::expand_state_by_one_qubit() {
  std::size_t previous_dimension = state_dimension_;

  num_qubits_++;
  state_dimension_ = 1ULL << num_qubits_;

  // Allocate new state vector
  void *new_device_state_vector;
  HANDLE_CUDA_ERROR(cudaMalloc(&new_device_state_vector,
                               state_dimension_ * sizeof(CudaDataType)));

  // Copy old state to first half, zero out second half
  constexpr int32_t threads_per_block = 256;
  uint32_t n_blocks =
      (state_dimension_ + threads_per_block - 1) / threads_per_block;

  if constexpr (std::is_same_v<ScalarType, float>) {
    set_first_n_elements_f32(n_blocks, threads_per_block,
                             new_device_state_vector, device_state_vector_,
                             previous_dimension, state_dimension_);
  } else {
    set_first_n_elements_f64(n_blocks, threads_per_block,
                             new_device_state_vector, device_state_vector_,
                             previous_dimension, state_dimension_);
  }

  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

  // Free old state
  HANDLE_CUDA_ERROR(cudaFree(device_state_vector_));
  device_state_vector_ = new_device_state_vector;
}
template <typename ScalarType>
sample_result state_vector_impl<ScalarType>::sample_kernel(std::size_t shots) {

  // Perform sampling on final state using cuStateVec
  std::size_t nShots = shots;
  std::vector<int32_t> bitOrdering(num_qubits_);
  for (int i = 0; i < num_qubits_; ++i)
    bitOrdering[i] = i;
  std::vector<int32_t> bitString(num_qubits_);
  std::unordered_map<std::string, std::size_t> counts;

  std::mt19937_64 gen(std::random_device{}());
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  for (std::size_t shot = 0; shot < nShots; ++shot) {
    double randnum = dist(gen);
    custatevecBatchMeasure(handle_, device_state_vector_, CUDA_C_64F,
                           num_qubits_, bitString.data(), bitOrdering.data(),
                           num_qubits_, randnum, CUSTATEVEC_COLLAPSE_NONE);
    // convert bitstring to human-readable string
    std::ostringstream bits;
    for (int q = num_qubits_ - 1; q >= 0; --q)
      bits << bitString[q];
    counts[bits.str()]++;
  }

  // sampling = false;

  // Construct and return sample_result
  return sample_result(ExecutionResult{counts});
}

// Explicit template instantiations
template class state_vector_impl<float>;
template class state_vector_impl<double>;

} // namespace detail

// ============================================================================
// state_vector Public Interface
// ============================================================================

template <typename ScalarType>
state_vector<ScalarType>::state_vector()
    : impl_(std::make_unique<detail::state_vector_impl<ScalarType>>()),
      random_engine_(random_device_()) {}

template <typename ScalarType>
state_vector<ScalarType>::state_vector(const qpu_configuration &config)
    : state_vector::BaseType(config),
      impl_(std::make_unique<detail::state_vector_impl<ScalarType>>()),
      random_engine_(random_device_()) {

  // Handle configuration options
  if (auto maybe_random_seed =
          extract_config<std::size_t>(m_configuration, "random_seed")) {
    set_random_seed(maybe_random_seed.value());
  }
}

template <typename ScalarType>
state_vector<ScalarType>::~state_vector() = default;

template <typename ScalarType>
void state_vector<ScalarType>::ensure_initialized() {
  if (num_qubits_ == 0) {
    // Start with a default single qubit
    impl_->allocate_zero_state(1);
    num_qubits_ = 1;
  }
}

template <typename ScalarType>
std::size_t state_vector<ScalarType>::get_state_dimension() const {
  return 1ULL << num_qubits_;
}

template <typename ScalarType>
void state_vector<ScalarType>::dump_state(std::ostream &os) {
  if (num_qubits_ == 0) {
    os << "No qubits allocated\n";
    return;
  }

  std::size_t dim = get_state_dimension();
  std::vector<std::complex<ScalarType>> state(dim);
  impl_->copy_to_host(state.data(), dim);

  for (std::size_t i = 0; i < dim; ++i) {
    os << state[i] << "\n";
  }
}

template <typename ScalarType>
cudaq::state state_vector<ScalarType>::get_state() {
  if (num_qubits_ == 0) {
    throw std::runtime_error(
        "[gpu::state_vector] Cannot get state: no qubits allocated");
  }

  return cudaq::state(new gpu_state_vector<ScalarType>(
      get_state_dimension(), impl_->get_device_pointer(), false));
}

template <typename ScalarType>
cudaq::state state_vector<ScalarType>::get_state(const state_data &data) {
  auto internal_state = get_internal_state(data);
  return cudaq::state(internal_state.release());
}

template <typename ScalarType>
std::unique_ptr<cudaq::SimulationState>
state_vector<ScalarType>::get_internal_state(const state_data &data) {
  return std::make_unique<gpu_state_vector<ScalarType>>(0, nullptr, true)
      ->createFromData(data);
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
  if (numLevels != 2) {
    throw std::runtime_error(
        "[gpu::state_vector] Only qubits (numLevels=2) are supported");
  }

  std::size_t new_idx = next_qudit_idx_++;
  allocated_qudits_.push_back(new_idx);
  qudit_levels_.push_back(numLevels);
  num_qubits_++;

  // Reallocate state with additional qubit
  if (impl_->get_num_qubits() == 0) {
    impl_->allocate_zero_state(num_qubits_);
  } else {
    // Expand existing state by one qubit in |0⟩ state
    impl_->expand_state_by_one_qubit();
  }

  return new_idx;
}

template <typename ScalarType>
std::vector<std::size_t>
state_vector<ScalarType>::allocateQudits(std::size_t numQudits,
                                         std::size_t numLevels) {
  if (numLevels != 2) {
    throw std::runtime_error(
        "[gpu::state_vector] Only qubits (numLevels=2) are supported");
  }

  std::vector<std::size_t> indices;
  indices.reserve(numQudits);

  // If this is the first allocation, do it efficiently
  if (num_qubits_ == 0) {
    for (std::size_t i = 0; i < numQudits; ++i) {
      std::size_t idx = next_qudit_idx_++;
      allocated_qudits_.push_back(idx);
      qudit_levels_.push_back(numLevels);
      indices.push_back(idx);
    }
    num_qubits_ = numQudits;
    impl_->allocate_zero_state(num_qubits_);
  } else {
    // Allocate one at a time (can be optimized later)
    for (std::size_t i = 0; i < numQudits; ++i) {
      indices.push_back(allocateQudit(numLevels));
    }
  }

  return indices;
}

template <typename ScalarType>
std::vector<std::size_t> state_vector<ScalarType>::allocateQudits(
    std::size_t numQudits, std::size_t numLevels, const void *state,
    simulation_precision precision) {
  if (numLevels != 2) {
    throw std::runtime_error(
        "[gpu::state_vector] Only qubits (numLevels=2) are supported");
  }

  // Check precision matches
  if constexpr (std::is_same_v<ScalarType, float>) {
    if (precision != simulation_precision::fp32) {
      throw std::runtime_error(
          "[gpu::state_vector] Precision mismatch: simulator is FP32");
    }
  } else {
    if (precision != simulation_precision::fp64) {
      throw std::runtime_error(
          "[gpu::state_vector] Precision mismatch: simulator is FP64");
    }
  }

  std::vector<std::size_t> indices;
  indices.reserve(numQudits);

  for (std::size_t i = 0; i < numQudits; ++i) {
    std::size_t idx = next_qudit_idx_++;
    allocated_qudits_.push_back(idx);
    qudit_levels_.push_back(numLevels);
    indices.push_back(idx);
  }

  std::size_t old_num_qubits = num_qubits_;
  num_qubits_ += numQudits;

  if (old_num_qubits == 0 && impl_->get_num_qubits() == 0) {
    // First allocation with provided state
    impl_->allocate_from_data(numQudits, state, 1ULL << numQudits);
  } else {
    // Expand existing state with new qubits from provided state
    impl_->expand_state_with_qubits(numQudits, state);
  }

  return indices;
}

template <typename ScalarType>
std::vector<std::size_t>
state_vector<ScalarType>::allocateQudits(std::size_t numQudits,
                                         std::size_t numLevels,
                                         const SimulationState *state) {
  if (numLevels != 2) {
    throw std::runtime_error(
        "[gpu::state_vector] Only qubits (numLevels=2) are supported");
  }

  if (!state) {
    throw std::runtime_error(
        "[gpu::state_vector] Null SimulationState provided");
  }

  // Check that the state has the right number of qubits
  if (state->getNumQubits() != numQudits) {
    throw std::runtime_error(
        "[gpu::state_vector] SimulationState qubit count mismatch");
  }

  // Check precision matches
  if constexpr (std::is_same_v<ScalarType, float>) {
    if (state->getPrecision() != SimulationState::precision::fp32) {
      throw std::runtime_error(
          "[gpu::state_vector] Precision mismatch: simulator is FP32");
    }
  } else {
    if (state->getPrecision() != SimulationState::precision::fp64) {
      throw std::runtime_error(
          "[gpu::state_vector] Precision mismatch: simulator is FP64");
    }
  }

  std::vector<std::size_t> indices;
  indices.reserve(numQudits);

  for (std::size_t i = 0; i < numQudits; ++i) {
    std::size_t idx = next_qudit_idx_++;
    allocated_qudits_.push_back(idx);
    qudit_levels_.push_back(numLevels);
    indices.push_back(idx);
  }

  std::size_t old_num_qubits = num_qubits_;
  num_qubits_ += numQudits;

  // Get tensor from the state
  auto tensor = state->getTensor(0);
  void *state_data = tensor.data;

  if (old_num_qubits == 0 && impl_->get_num_qubits() == 0) {
    // First allocation - if state is on device, copy device-to-device
    if (state->isDeviceData()) {
      // Allocate and copy from device
      impl_->initialize_state(numQudits);
      void *temp_device;
      HANDLE_CUDA_ERROR(cudaMalloc(&temp_device, tensor.get_num_elements() *
                                                     tensor.element_size()));
      HANDLE_CUDA_ERROR(
          cudaMemcpy(temp_device, state_data,
                     tensor.get_num_elements() * tensor.element_size(),
                     cudaMemcpyDeviceToDevice));
      impl_->copy_from_host(temp_device, tensor.get_num_elements());
      HANDLE_CUDA_ERROR(cudaFree(temp_device));
    } else {
      // Copy from host
      impl_->allocate_from_data(numQudits, state_data,
                                tensor.get_num_elements());
    }
  } else {
    // Expand existing state
    if (state->isDeviceData()) {
      // TODO: Support device-to-device expansion
      throw std::runtime_error(
          "[gpu::state_vector] Device-to-device state expansion not yet "
          "implemented");
    } else {
      impl_->expand_state_with_qubits(numQudits, state_data);
    }
  }

  return indices;
}

template <typename ScalarType>
void state_vector<ScalarType>::deallocate(std::size_t idx) {
  // Deallocation in state vector simulators requires partial trace
  // which is computationally expensive and changes the state representation
  // from pure to mixed. For now, this is not supported.
  // throw std::runtime_error(
  //     "[gpu::state_vector] Qubit deallocation not supported. "
  //     "State vector simulators cannot deallocate individual qubits without "
  //     "performing a partial trace (which produces a density matrix). "
  //     "Consider resetting the entire simulator or using a density matrix "
  //     "simulator for this operation.");
  return;
}

template <typename ScalarType>
void state_vector<ScalarType>::deallocate(
    const std::vector<std::size_t> &idxs) {
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

  if (num_qubits_ == 0) {
    throw std::runtime_error(
        "[gpu::state_vector] Cannot apply gate: no qubits allocated");
  }

  // Convert size_t to int for cuStateVec
  std::vector<int> controls_int(controls.begin(), controls.end());
  std::vector<int> targets_int(targets.begin(), targets.end());

  impl_->apply_gate(matrixRowMajor, controls_int, targets_int);
}

// Note: applyControlRegion and applyAdjointRegion are now implemented in the
// base simulator trait, so we don't need to provide them here

template <typename ScalarType>
void state_vector<ScalarType>::reset(std::size_t qidx) {
  if (num_qubits_ == 0) {
    throw std::runtime_error(
        "[gpu::state_vector] Cannot reset: no qubits allocated");
  }

  std::uniform_real_distribution<double> dist(0.0, 1.0);
  double rand_val = dist(random_engine_);
  impl_->reset_qubit(qidx, rand_val);
}

template <typename ScalarType>
void state_vector<ScalarType>::apply_exp_pauli(
    double theta, const std::vector<std::size_t> &controls,
    const std::vector<std::size_t> &qubitIds, const cudaq::spin_op_term &term) {

  if (num_qubits_ == 0) {
    throw std::runtime_error(
        "[gpu::state_vector] Cannot apply exp_pauli: no qubits allocated");
  }

  // Convert controls and targets to int
  std::vector<int> controls_int(controls.begin(), controls.end());
  std::vector<int> targets_int(qubitIds.begin(), qubitIds.end());

  // Convert Pauli operators
  std::vector<custatevecPauli_t> paulis;
  for (const auto &op : term) {
    auto pauli = op.as_pauli();
    if (pauli == cudaq::pauli::I)
      paulis.push_back(CUSTATEVEC_PAULI_I);
    else if (pauli == cudaq::pauli::X)
      paulis.push_back(CUSTATEVEC_PAULI_X);
    else if (pauli == cudaq::pauli::Y)
      paulis.push_back(CUSTATEVEC_PAULI_Y);
    else if (pauli == cudaq::pauli::Z)
      paulis.push_back(CUSTATEVEC_PAULI_Z);
  }

  impl_->apply_pauli_rotation(theta, controls_int, targets_int, paulis);
}

template <typename ScalarType>
std::size_t state_vector<ScalarType>::mz(std::size_t idx,
                                         const std::string regName) {
  if (num_qubits_ == 0) {
    throw std::runtime_error(
        "[gpu::state_vector] Cannot measure: no qubits allocated");
  }

  std::uniform_real_distribution<double> dist(0.0, 1.0);
  double rand_val = dist(random_engine_);
  bool result = impl_->measure_qubit(idx, rand_val);
  return result ? 1 : 0;
}

template <typename ScalarType>
void state_vector<ScalarType>::set_random_seed(std::size_t seed) {
  random_engine_ = std::mt19937(seed);
}

template <typename ScalarType>
sample_result state_vector<ScalarType>::sample_kernel(std::size_t shots) {
  return impl_->sample_kernel(shots);
}

// Template method implementations need to be in header or explicitly
// instantiated Explicit template instantiations
template class state_vector<float>;
template class state_vector<double>;

} // namespace cudaq::simulator::gpu
