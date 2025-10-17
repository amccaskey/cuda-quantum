/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cuComplex.h"
#include "cuda_runtime.h"
#include "cudaq/spin_op.h"
#include "custatevec.h"
#include "device_launch_parameters.h"
#define CUDAQ_NO_STD20
#include "state_vector_impl.h"
#undef CUDAQ_NO_STD20
#include <algorithm>
#include <bitset>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <thrust/complex.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/inner_product.h>

#define HANDLE_ERROR(x)                                                        \
  {                                                                            \
    const auto err = x;                                                        \
    if (err != CUSTATEVEC_STATUS_SUCCESS) {                                    \
      throw std::runtime_error(                                                \
          std::string("[custatevec] ") + custatevecGetErrorString(err) +       \
          " in " + __FUNCTION__ + " (line " + std::to_string(__LINE__) + ")"); \
    }                                                                          \
  }

#define HANDLE_CUDA_ERROR(x)                                                   \
  {                                                                            \
    const auto err = x;                                                        \
    if (err != cudaSuccess) {                                                  \
      throw std::runtime_error(                                                \
          std::string("[cuda] ") + cudaGetErrorString(err) + " in " +          \
          __FUNCTION__ + " (line " + std::to_string(__LINE__) + ")");          \
    }                                                                          \
  }

namespace cudaq::simulator::gpu {

// Utility kernels
template <typename CudaDataType>
__global__ void cudaInitializeDeviceStateVector(CudaDataType *sv, int64_t dim) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i == 0) {
    sv[i].x = 1.0;
    sv[i].y = 0.0;
  } else if (i < dim) {
    sv[i].x = 0.0;
    sv[i].y = 0.0;
  }
}

template <typename CudaDataType>
__global__ void cudaSetFirstNElements(CudaDataType *sv,
                                      const CudaDataType *__restrict__ sv2,
                                      int64_t N, int64_t totalSize) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < totalSize) {
    if (i < N) {
      sv[i].x = sv2[i].x;
      sv[i].y = sv2[i].y;
    } else {
      sv[i].x = 0.0;
      sv[i].y = 0.0;
    }
  }
}

// Custom functor for thrust inner product
template <typename ScalarType>
struct AdotConjB {
  __host__ __device__ thrust::complex<ScalarType>
  operator()(thrust::complex<ScalarType> a, thrust::complex<ScalarType> b) {
    return a * thrust::conj(b);
  };
};

// Helper function for inner product computation
template <typename ScalarType>
std::complex<ScalarType> innerProduct(void *devicePtr, void *otherPtr,
                                      std::size_t size,
                                      bool createDeviceAlloc) {
  auto *castedDevicePtr =
      reinterpret_cast<thrust::complex<ScalarType> *>(devicePtr);
  thrust::device_ptr<thrust::complex<ScalarType>> thrustDevPtrABegin(
      castedDevicePtr);
  thrust::device_ptr<thrust::complex<ScalarType>> thrustDevPtrAEnd(
      castedDevicePtr + size);
  thrust::device_ptr<thrust::complex<ScalarType>> thrustDevPtrBBegin;

  if (createDeviceAlloc) {
    auto *castedOtherPtr =
        reinterpret_cast<std::complex<ScalarType> *>(otherPtr);
    std::vector<std::complex<ScalarType>> dataAsVec(castedOtherPtr,
                                                    castedOtherPtr + size);
    thrust::device_vector<thrust::complex<ScalarType>> otherDevPtr(dataAsVec);
    thrustDevPtrBBegin = otherDevPtr.data();
  } else {
    auto *castedOtherPtr =
        reinterpret_cast<thrust::complex<ScalarType> *>(otherPtr);
    thrustDevPtrBBegin =
        thrust::device_ptr<thrust::complex<ScalarType>>(castedOtherPtr);
  }

  thrust::complex<ScalarType> result = thrust::inner_product(
      thrustDevPtrABegin, thrustDevPtrAEnd, thrustDevPtrBBegin,
      thrust::complex<ScalarType>(0.0),
      thrust::plus<thrust::complex<ScalarType>>(), AdotConjB<ScalarType>());

  return std::complex<ScalarType>(result.real(), result.imag());
}

//==============================================================================
// CuStateVecSimulationState Implementation
//==============================================================================

template <typename ScalarType>
CuStateVecSimulationState<ScalarType>::CuStateVecSimulationState(
    std::size_t s, void *ptr, custatevecHandle_t h)
    : size(s), devicePtr(ptr), handle(h) {}

template <typename ScalarType>
CuStateVecSimulationState<ScalarType>::CuStateVecSimulationState(
    std::size_t s, void *ptr, custatevecHandle_t h, bool owns)
    : size(s), devicePtr(ptr), ownsDevicePtr(owns), handle(h) {}

template <typename ScalarType>
CuStateVecSimulationState<ScalarType>::~CuStateVecSimulationState() {
  if (ownsDevicePtr && devicePtr) {
    checkAndSetDevice();
    cudaFree(devicePtr);
  }
}

template <typename ScalarType>
void CuStateVecSimulationState<ScalarType>::checkAndSetDevice() const {
  int dev = 0;
  HANDLE_CUDA_ERROR(cudaGetDevice(&dev));
  auto currentDevice = deviceFromPointer(devicePtr);
  if (dev != currentDevice)
    HANDLE_CUDA_ERROR(cudaSetDevice(currentDevice));
}

template <typename ScalarType>
void CuStateVecSimulationState<ScalarType>::extractValues(
    std::complex<ScalarType> *value, std::size_t start, std::size_t end) const {
  checkAndSetDevice();
  HANDLE_CUDA_ERROR(cudaMemcpy(
      value, reinterpret_cast<std::complex<ScalarType> *>(devicePtr) + start,
      (end - start) * sizeof(std::complex<ScalarType>),
      cudaMemcpyDeviceToHost));
}

template <typename ScalarType>
bool CuStateVecSimulationState<ScalarType>::isDevicePointer(void *ptr) const {
  cudaPointerAttributes attributes;
  HANDLE_CUDA_ERROR(cudaPointerGetAttributes(&attributes, ptr));
  return attributes.type > 1;
}

template <typename ScalarType>
int CuStateVecSimulationState<ScalarType>::deviceFromPointer(void *ptr) const {
  cudaPointerAttributes attributes;
  HANDLE_CUDA_ERROR(cudaPointerGetAttributes(&attributes, ptr));
  return attributes.device;
}

template <typename ScalarType>
auto CuStateVecSimulationState<ScalarType>::maybeCopyToDevice(std::size_t size,
                                                              void *dataPtr) {
  if (isDevicePointer(dataPtr))
    return dataPtr;

  std::complex<ScalarType> *ptr = nullptr;
  HANDLE_CUDA_ERROR(
      cudaMalloc((void **)&ptr, size * sizeof(std::complex<ScalarType>)));
  HANDLE_CUDA_ERROR(cudaMemcpy(ptr, dataPtr,
                               size * sizeof(std::complex<ScalarType>),
                               cudaMemcpyHostToDevice));
  return reinterpret_cast<void *>(ptr);
}

template <typename ScalarType>
std::size_t CuStateVecSimulationState<ScalarType>::getNumQubits() const {
  return std::log2(size);
}

template <typename ScalarType>
std::complex<double> CuStateVecSimulationState<ScalarType>::overlap(
    const cudaq::SimulationState &other) {
  if (getTensor().extents != other.getTensor().extents)
    throw std::runtime_error("[custatevec-state] overlap error - other state "
                             "dimension not equal to this state dimension.");

  if (other.getPrecision() != getPrecision()) {
    throw std::runtime_error(
        "[custatevec-state] overlap error - precision mismatch.");
  }

  int currentDev;
  cudaGetDevice(&currentDev);
  auto dataDev = deviceFromPointer(devicePtr);
  if (currentDev != dataDev)
    cudaSetDevice(dataDev);

  if (isDevicePointer(other.getTensor().data)) {
    if (deviceFromPointer(devicePtr) !=
        deviceFromPointer(other.getTensor().data))
      throw std::runtime_error(
          "overlap requested for device pointers on separate GPU devices.");

    auto cmplx = innerProduct<ScalarType>(devicePtr, other.getTensor().data,
                                          size, false);
    return std::abs(cmplx);
  } else {
    auto cmplx =
        innerProduct<ScalarType>(devicePtr, other.getTensor().data, size, true);
    return std::abs(cmplx);
  }
}

template <typename ScalarType>
std::complex<double> CuStateVecSimulationState<ScalarType>::getAmplitude(
    const std::vector<int> &basisState) {
  if (getNumQubits() != basisState.size())
    throw std::runtime_error("[custatevec-state] getAmplitude with an invalid "
                             "number of bits in the basis state");

  if (std::any_of(basisState.begin(), basisState.end(),
                  [](int x) { return x != 0 && x != 1; }))
    throw std::runtime_error(
        "[custatevec-state] getAmplitude with an invalid basis state: only "
        "qubit state (0 or 1) is supported.");

  const std::size_t idx = std::accumulate(
      std::make_reverse_iterator(basisState.end()),
      std::make_reverse_iterator(basisState.begin()), 0ull,
      [](std::size_t acc, int bit) { return (acc << 1) + bit; });

  std::complex<ScalarType> value;
  extractValues(&value, idx, idx + 1);
  return {value.real(), value.imag()};
}

template <typename ScalarType>
void CuStateVecSimulationState<ScalarType>::dump(std::ostream &os) const {
  std::vector<std::complex<ScalarType>> tmp(size);
  HANDLE_CUDA_ERROR(cudaMemcpy(tmp.data(), devicePtr,
                               size * sizeof(std::complex<ScalarType>),
                               cudaMemcpyDeviceToHost));

  for (std::size_t i = 0; i < size; ++i) {
    os << "|" << std::bitset<32>(i).to_string().substr(32 - getNumQubits())
       << ">: " << tmp[i] << "\n";
  }
}

template <typename ScalarType>
bool CuStateVecSimulationState<ScalarType>::isDeviceData() const {
  return true;
}

template <typename ScalarType>
typename CuStateVecSimulationState<ScalarType>::precision
CuStateVecSimulationState<ScalarType>::getPrecision() const {
  if constexpr (std::is_same_v<ScalarType, float>)
    return cudaq::SimulationState::precision::fp32;
  return cudaq::SimulationState::precision::fp64;
}

template <typename ScalarType>
std::unique_ptr<SimulationState>
CuStateVecSimulationState<ScalarType>::createFromSizeAndPtr(std::size_t size,
                                                            void *ptr,
                                                            std::size_t type) {
  bool weOwnTheData = type < 2;
  ptr = maybeCopyToDevice(size, ptr);
  return std::make_unique<CuStateVecSimulationState<ScalarType>>(
      size, ptr, handle, weOwnTheData);
}

template <typename ScalarType>
typename SimulationState::Tensor
CuStateVecSimulationState<ScalarType>::getTensor(std::size_t tensorIdx) const {
  if (tensorIdx != 0)
    throw std::runtime_error("[cusv-state] invalid tensor requested.");
  return Tensor{devicePtr, std::vector<std::size_t>{size}, getPrecision()};
}

template <typename ScalarType>
std::vector<typename SimulationState::Tensor>
CuStateVecSimulationState<ScalarType>::getTensors() const {
  return {getTensor()};
}

template <typename ScalarType>
std::size_t CuStateVecSimulationState<ScalarType>::getNumTensors() const {
  return 1;
}

template <typename ScalarType>
std::complex<double> CuStateVecSimulationState<ScalarType>::operator()(
    std::size_t tensorIdx, const std::vector<std::size_t> &indices) {
  if (tensorIdx != 0)
    throw std::runtime_error("[cusv-state] invalid tensor requested.");

  if (indices.size() != 1)
    throw std::runtime_error("[cusv-state] invalid element extraction.");

  auto idx = indices[0];
  std::complex<ScalarType> value;
  extractValues(&value, idx, idx + 1);
  return {value.real(), value.imag()};
}

template <typename ScalarType>
void CuStateVecSimulationState<ScalarType>::toHost(
    std::complex<double> *userData, std::size_t numElements) const {
  if constexpr (std::is_same_v<ScalarType, float>)
    throw std::runtime_error("simulation precision is FP32 but toHost "
                             "requested with FP64 host buffer.");

  if (numElements != size)
    throw std::runtime_error("[custatevec-state] provided toHost pointer has "
                             "invalid number of elements specified.");

  extractValues(reinterpret_cast<std::complex<ScalarType> *>(userData), 0,
                size);
}

template <typename ScalarType>
void CuStateVecSimulationState<ScalarType>::toHost(
    std::complex<float> *userData, std::size_t numElements) const {
  if constexpr (std::is_same_v<ScalarType, double>)
    throw std::runtime_error("simulation precision is FP64 but toHost "
                             "requested with FP32 host buffer.");

  if (numElements != size)
    throw std::runtime_error("[custatevec-state] provided toHost pointer has "
                             "invalid number of elements specified.");

  extractValues(reinterpret_cast<std::complex<ScalarType> *>(userData), 0,
                size);
}

template <typename ScalarType>
void CuStateVecSimulationState<ScalarType>::destroyState() {
  if (!ownsDevicePtr)
    return;

  checkAndSetDevice();
  HANDLE_CUDA_ERROR(cudaFree(devicePtr));
  devicePtr = nullptr;
}

//==============================================================================
// state_vector::Impl Implementation
//==============================================================================

state_vector::Impl::Impl() : randomEngine(randomDevice()) {
  HANDLE_CUDA_ERROR(cudaFree(0));

  if constexpr (std::is_same_v<ScalarType, float>) {
    cuStateVecComputeType = CUSTATEVEC_COMPUTE_32F;
    cuStateVecCudaDataType = CUDA_C_32F;
  }
}

state_vector::Impl::~Impl() {
  if (handle) {
    custatevecDestroy(handle);
  }
  if (deviceStateVector && ownsDeviceVector) {
    cudaFree(deviceStateVector);
  }
  if (extraWorkspace) {
    cudaFree(extraWorkspace);
  }
}
sample_result state_vector::Impl::sample(std::size_t shots,
                                         const std::string &kernel_name,
                                         const std::function<void()> &wrapped) {

  // /// @brief Sample the multi-qubit state.
  // cudaq::ExecutionResult sample(const std::vector<std::size_t> &measuredBits,
  //                               const int shots) override {
  // call the kernel
  disableDeallocate = true;
  wrapped();
  disableDeallocate = false;
  std::vector<std::size_t> measuredBits;
  for (std::size_t i = 0; i < nQubitsAllocated; i++)
    measuredBits.push_back(i);
  for (auto &m : measuredBits)
    deallocate(m);

  double expVal = 0.0;
  // cudaq::CountsDictionary counts;
  std::vector<custatevecPauli_t> z_pauli;
  std::vector<int> measuredBits32;
  for (auto m : measuredBits) {
    measuredBits32.push_back(m);
    z_pauli.push_back(CUSTATEVEC_PAULI_Z);
  }

  if (shots < 1) {
    // Just compute the expected value on <Z...Z>
    const uint32_t nBasisBitsArray[] = {(uint32_t)measuredBits.size()};
    const int *basisBitsArray[] = {measuredBits32.data()};
    const custatevecPauli_t *pauliArray[] = {z_pauli.data()};
    double expectationValues[1];
    HANDLE_ERROR(custatevecComputeExpectationsOnPauliBasis(
        handle, deviceStateVector, cuStateVecCudaDataType, nQubitsAllocated,
        expectationValues, pauliArray, 1, basisBitsArray, nBasisBitsArray));
    expVal = expectationValues[0];
    return cudaq::ExecutionResult{expVal};
  }

  // Grab some random seed values and create the sampler
  auto randomValues_ = randomValues(shots, 1.0);
  custatevecSamplerDescriptor_t sampler;
  HANDLE_ERROR(custatevecSamplerCreate(
      handle, deviceStateVector, cuStateVecCudaDataType, measuredBits.size(),
      &sampler, shots, &extraWorkspaceSizeInBytes));
  // allocate external workspace if necessary
  if (extraWorkspaceSizeInBytes > 0)
    HANDLE_CUDA_ERROR(cudaMalloc(&extraWorkspace, extraWorkspaceSizeInBytes));

  // Run the sampling preprocess step.
  HANDLE_ERROR(custatevecSamplerPreprocess(handle, sampler, extraWorkspace,
                                           extraWorkspaceSizeInBytes));

  // Sample!
  std::vector<custatevecIndex_t> bitstrings0(shots);
  HANDLE_ERROR(custatevecSamplerSample(
      handle, sampler, bitstrings0.data(), measuredBits32.data(),
      measuredBits32.size(), randomValues_.data(), shots,
      CUSTATEVEC_SAMPLER_OUTPUT_ASCENDING_ORDER));

  if (extraWorkspace) {
    HANDLE_CUDA_ERROR(cudaFree(extraWorkspace));
    extraWorkspace = nullptr;
  }

  std::vector<std::string> sequentialData;
  sequentialData.reserve(shots);

  cudaq::ExecutionResult counts;

  // We've sampled, convert the results to our ExecutionResult counts
  for (int i = 0; i < shots; ++i) {
    auto bitstring = std::bitset<64>(bitstrings0[i])
                         .to_string()
                         .erase(0, 64 - measuredBits.size());
    std::reverse(bitstring.begin(), bitstring.end());
    counts.appendResult(bitstring, 1);
    sequentialData.push_back(std::move(bitstring));
  }

  // Compute the expectation value from the counts
  for (auto &kv : counts.counts) {
    auto par = cudaq::sample_result::has_even_parity(kv.first);
    auto p = kv.second / (double)shots;
    if (!par) {
      p = -p;
    }
    expVal += p;
  }

  counts.expectationValue = expVal;

  HANDLE_ERROR(custatevecSamplerDestroy(sampler));

  sample_result result;
  result.append(counts);
  return result;
}

void state_vector::Impl::ensureStateAllocated(std::size_t numQubits) {
  if (nQubitsAllocated >= numQubits)
    return;

  std::size_t newStateDim = 1UL << numQubits;

  if (!handle) {
    HANDLE_ERROR(custatevecCreate(&handle));
  }

  if (!deviceStateVector) {
    HANDLE_CUDA_ERROR(cudaMalloc((void **)&deviceStateVector,
                                 newStateDim * sizeof(CudaDataType)));
    ownsDeviceVector = true;

    constexpr int32_t threads_per_block = 256;
    uint32_t n_blocks =
        (newStateDim + threads_per_block - 1) / threads_per_block;
    cudaInitializeDeviceStateVector<CudaDataType>
        <<<n_blocks, threads_per_block>>>(
            reinterpret_cast<CudaDataType *>(deviceStateVector), newStateDim);
    HANDLE_CUDA_ERROR(cudaGetLastError());
  } else {
    void *newDeviceStateVector;
    HANDLE_CUDA_ERROR(cudaMalloc((void **)&newDeviceStateVector,
                                 newStateDim * sizeof(CudaDataType)));

    constexpr int32_t threads_per_block = 256;
    uint32_t n_blocks =
        (newStateDim + threads_per_block - 1) / threads_per_block;
    cudaSetFirstNElements<CudaDataType><<<n_blocks, threads_per_block>>>(
        reinterpret_cast<CudaDataType *>(newDeviceStateVector),
        reinterpret_cast<CudaDataType *>(deviceStateVector), stateDimension,
        newStateDim);
    HANDLE_CUDA_ERROR(cudaGetLastError());

    if (ownsDeviceVector) {
      HANDLE_CUDA_ERROR(cudaFree(deviceStateVector));
    }
    deviceStateVector = newDeviceStateVector;
    ownsDeviceVector = true;
  }

  nQubitsAllocated = numQubits;
  stateDimension = newStateDim;
}

custatevecPauli_t
state_vector::Impl::pauliStringToEnum(const std::string_view type) {
  if (type == "rx" || type == "x")
    return CUSTATEVEC_PAULI_X;
  else if (type == "ry" || type == "y")
    return CUSTATEVEC_PAULI_Y;
  else if (type == "rz" || type == "z")
    return CUSTATEVEC_PAULI_Z;
  else if (type == "i")
    return CUSTATEVEC_PAULI_I;
  throw std::invalid_argument("Invalid Pauli type: " + std::string(type));
}

std::vector<double> state_vector::Impl::randomValues(uint64_t num_samples,
                                                     double max_value) {
  std::vector<double> rs;
  rs.reserve(num_samples);
  std::uniform_real_distribution<double> distr(0.0, max_value);
  for (uint64_t i = 0; i < num_samples; ++i) {
    rs.emplace_back(distr(randomEngine));
  }
  std::sort(rs.begin(), rs.end());
  return rs;
}

void state_vector::Impl::applyGateMatrix(const std::vector<DataType> &matrix,
                                         const std::vector<int> &controls,
                                         const std::vector<int> &targets) {
  HANDLE_ERROR(custatevecApplyMatrixGetWorkspaceSize(
      handle, cuStateVecCudaDataType, nQubitsAllocated, matrix.data(),
      cuStateVecCudaDataType, CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, targets.size(),
      controls.size(), cuStateVecComputeType, &extraWorkspaceSizeInBytes));

  if (extraWorkspaceSizeInBytes > 0) {
    HANDLE_CUDA_ERROR(cudaMalloc(&extraWorkspace, extraWorkspaceSizeInBytes));
  }

  HANDLE_ERROR(custatevecApplyMatrix(
      handle, deviceStateVector, cuStateVecCudaDataType, nQubitsAllocated,
      matrix.data(), cuStateVecCudaDataType, CUSTATEVEC_MATRIX_LAYOUT_ROW, 0,
      targets.data(), targets.size(),
      controls.empty() ? nullptr : controls.data(), nullptr, controls.size(),
      cuStateVecComputeType, extraWorkspace, extraWorkspaceSizeInBytes));

  if (extraWorkspace) {
    HANDLE_CUDA_ERROR(cudaFree(extraWorkspace));
    extraWorkspace = nullptr;
    extraWorkspaceSizeInBytes = 0;
  }
}

void state_vector::Impl::dump_state(std::ostream &os) {
  if (!deviceStateVector)
    return;

  std::vector<DataType> hostData(stateDimension);
  HANDLE_CUDA_ERROR(cudaMemcpy(hostData.data(), deviceStateVector,
                               stateDimension * sizeof(DataType),
                               cudaMemcpyDeviceToHost));

  for (std::size_t i = 0; i < stateDimension; ++i) {
    os << "|" << std::bitset<32>(i).to_string().substr(32 - nQubitsAllocated)
       << ">: " << hostData[i] << "\n";
  }
}

std::unique_ptr<cudaq::SimulationState>
state_vector::Impl::get_internal_state() {
  if (!deviceStateVector) {
    throw std::runtime_error("No quantum state allocated");
  }

  return std::make_unique<CuStateVecSimulationState<ScalarType>>(
      stateDimension, deviceStateVector, handle, false);
}

simulation_precision state_vector::Impl::get_precision() const {
  if constexpr (std::is_same_v<ScalarType, float>)
    return simulation_precision::fp32;
  return simulation_precision::fp64;
}

std::size_t state_vector::Impl::allocateQudit(std::size_t numLevels) {
  if (numLevels != 2) {
    throw std::invalid_argument("Only qubits (numLevels=2) are supported");
  }

  ensureStateAllocated(nQubitsAllocated + 1);
  return nQubitsAllocated - 1;
}

std::vector<std::size_t>
state_vector::Impl::allocateQudits(std::size_t numQudits, std::size_t numLevels,
                                   const void *state,
                                   simulation_precision precision) {
  if (numLevels != 2) {
    throw std::invalid_argument("Only qubits (numLevels=2) are supported");
  }

  std::vector<std::size_t> indices;
  indices.reserve(numQudits);

  std::size_t startingQubit = nQubitsAllocated;
  ensureStateAllocated(nQubitsAllocated + numQudits);

  for (std::size_t i = 0; i < numQudits; ++i) {
    indices.push_back(startingQubit + i);
  }

  if (state) {
    std::size_t stateSize = 1UL << numQudits;
    HANDLE_CUDA_ERROR(cudaMemcpy(deviceStateVector, state,
                                 stateSize * sizeof(CudaDataType),
                                 cudaMemcpyHostToDevice));
  }

  return indices;
}

std::vector<std::size_t>
state_vector::Impl::allocateQudits(std::size_t numQudits, std::size_t numLevels,
                                   const SimulationState *state) {
  if (numLevels != 2) {
    throw std::invalid_argument("Only qubits (numLevels=2) are supported");
  }

  std::vector<std::size_t> indices;
  indices.reserve(numQudits);

  std::size_t startingQubit = nQubitsAllocated;
  ensureStateAllocated(nQubitsAllocated + numQudits);

  for (std::size_t i = 0; i < numQudits; ++i) {
    indices.push_back(startingQubit + i);
  }

  if (state && state->isDeviceData()) {
    auto tensor = state->getTensor();
    HANDLE_CUDA_ERROR(cudaMemcpy(deviceStateVector, tensor.data,
                                 tensor.extents[0] * sizeof(CudaDataType),
                                 cudaMemcpyDeviceToDevice));
  }

  return indices;
}

std::vector<std::size_t>
state_vector::Impl::allocateQudits(std::size_t numQudits,
                                   std::size_t numLevels) {
  return allocateQudits(numQudits, numLevels, nullptr,
                        simulation_precision::fp64);
}

void state_vector::Impl::deallocate(std::size_t idx) {
  if (disableDeallocate)
    return;
  // For now, just reduce the count - actual deallocation would require
  // more complex state manipulation
  if (idx == nQubitsAllocated - 1) {
    nQubitsAllocated--;
    stateDimension = 1UL << nQubitsAllocated;
  }
}

void state_vector::Impl::deallocate(const std::vector<std::size_t> &idxs) {
  for (auto idx : idxs) {
    deallocate(idx);
  }
}

void state_vector::Impl::apply(
    const std::vector<std::complex<double>> &matrixRowMajor,
    const std::vector<std::size_t> &controls,
    const std::vector<std::size_t> &targets,
    const traits::operation_metadata &metadata) {
  std::vector<int> controls32, targets32;
  std::transform(controls.begin(), controls.end(),
                 std::back_inserter(controls32),
                 [](std::size_t idx) { return static_cast<int>(idx); });
  std::transform(targets.begin(), targets.end(), std::back_inserter(targets32),
                 [](std::size_t idx) { return static_cast<int>(idx); });

  applyGateMatrix(matrixRowMajor, controls32, targets32);
}

void state_vector::Impl::applyControlRegion(
    const std::vector<std::size_t> &controls,
    const std::function<void()> &wrapped) {
  // For now, just execute the wrapped function
  // In a full implementation, would need to track control context
  wrapped();
}

void state_vector::Impl::applyAdjointRegion(
    const std::function<void()> &wrapped) {
  // For now, just execute the wrapped function
  // In a full implementation, would need to track adjoint context
  wrapped();
}

void state_vector::Impl::reset(std::size_t qidx) {
  std::size_t measureResult = mz(qidx);
  if (measureResult == 1) {
    // Apply X gate to flip back to |0>
    std::vector<std::complex<double>> xMatrix = {
        {0, 0}, {1, 0}, {1, 0}, {0, 0}};
    apply(xMatrix, {}, {qidx}, traits::operation_metadata("x"));
  }
}

void state_vector::Impl::apply_exp_pauli(
    double theta, const std::vector<std::size_t> &controls,
    const std::vector<std::size_t> &qubitIds, const cudaq::spin_op_term &term) {
  std::vector<int> controls32, targets32;
  std::transform(controls.begin(), controls.end(),
                 std::back_inserter(controls32),
                 [](std::size_t idx) { return static_cast<int>(idx); });

  std::vector<custatevecPauli_t> paulis;
  std::size_t idx = 0;
  for (const auto &op : term) {
    auto pauli = op.as_pauli();
    if (pauli == cudaq::pauli::I)
      paulis.push_back(CUSTATEVEC_PAULI_I);
    else if (pauli == cudaq::pauli::X)
      paulis.push_back(CUSTATEVEC_PAULI_X);
    else if (pauli == cudaq::pauli::Y)
      paulis.push_back(CUSTATEVEC_PAULI_Y);
    else
      paulis.push_back(CUSTATEVEC_PAULI_Z);

    targets32.push_back(qubitIds[idx++]);
  }

  HANDLE_ERROR(custatevecApplyPauliRotation(
      handle, deviceStateVector, cuStateVecCudaDataType, nQubitsAllocated,
      theta, paulis.data(), targets32.data(), targets32.size(),
      controls32.data(), nullptr, controls32.size()));
}

std::size_t state_vector::Impl::mz(std::size_t idx, const std::string regName) {
  const int basisBits[] = {static_cast<int>(idx)};
  int parity;
  double rand = randomValues(1, 1.0)[0];

  HANDLE_ERROR(custatevecMeasureOnZBasis(
      handle, deviceStateVector, cuStateVecCudaDataType, nQubitsAllocated,
      &parity, basisBits, 1, rand, CUSTATEVEC_COLLAPSE_NORMALIZE_AND_ZERO));

  return static_cast<std::size_t>(parity);
}

// Template instantiations
template class CuStateVecSimulationState<float>;
template class CuStateVecSimulationState<double>;

} // namespace cudaq::simulator::gpu
