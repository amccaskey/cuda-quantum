/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include "state_vector.h"

#include "cuComplex.h"
#include "cuda_runtime.h"
#include "custatevec.h"
#include <complex>
#include <memory>
#include <random>
#include <vector>

namespace cudaq::simulator::gpu {

template <typename ScalarType>
class CuStateVecSimulationState : public cudaq::SimulationState {
private:
  std::size_t size = 0;
  void *devicePtr = nullptr;
  bool ownsDevicePtr = true;
  custatevecHandle_t handle = nullptr;

  void checkAndSetDevice() const;
  void extractValues(std::complex<ScalarType> *value, std::size_t start,
                     std::size_t end) const;
  bool isDevicePointer(void *ptr) const;
  int deviceFromPointer(void *ptr) const;
  auto maybeCopyToDevice(std::size_t size, void *dataPtr);

public:
  CuStateVecSimulationState(std::size_t s, void *ptr, custatevecHandle_t h);
  CuStateVecSimulationState(std::size_t s, void *ptr, custatevecHandle_t h,
                            bool owns);
  ~CuStateVecSimulationState() override;

  std::size_t getNumQubits() const override;
  std::complex<double> overlap(const cudaq::SimulationState &other) override;
  std::complex<double>
  getAmplitude(const std::vector<int> &basisState) override;
  void dump(std::ostream &os) const override;
  bool isDeviceData() const override;
  precision getPrecision() const override;

  std::unique_ptr<SimulationState>
  createFromSizeAndPtr(std::size_t size, void *ptr, std::size_t type) override;
  Tensor getTensor(std::size_t tensorIdx = 0) const override;
  std::vector<Tensor> getTensors() const override;
  std::size_t getNumTensors() const override;

  std::complex<double>
  operator()(std::size_t tensorIdx,
             const std::vector<std::size_t> &indices) override;
  void toHost(std::complex<double> *userData,
              std::size_t numElements) const override;
  void toHost(std::complex<float> *userData,
              std::size_t numElements) const override;
  void destroyState() override;

  const void *getDevicePointer() const { return devicePtr; }
  custatevecHandle_t getHandle() const { return handle; }
};

class state_vector::Impl {
public:
  using ScalarType = double;
  using DataType = std::complex<ScalarType>;
  using CudaDataType = cuDoubleComplex;

private:
  bool disableDeallocate = false;
  void *deviceStateVector = nullptr;
  custatevecHandle_t handle = nullptr;
  void *extraWorkspace = nullptr;
  size_t extraWorkspaceSizeInBytes = 0;

  std::size_t nQubitsAllocated = 0;
  std::size_t stateDimension = 0;
  custatevecComputeType_t cuStateVecComputeType = CUSTATEVEC_COMPUTE_64F;
  cudaDataType_t cuStateVecCudaDataType = CUDA_C_64F;

  std::random_device randomDevice;
  std::mt19937 randomEngine;
  bool ownsDeviceVector = true;

  void ensureStateAllocated(std::size_t numQubits);
  void applyGateMatrix(const std::vector<DataType> &matrix,
                       const std::vector<int> &controls,
                       const std::vector<int> &targets);
  custatevecPauli_t pauliStringToEnum(const std::string_view type);
  std::vector<double> randomValues(uint64_t num_samples, double max_value);

public:
  Impl();
  ~Impl();
  sample_result sample(std::size_t num_shots, const std::string &kernel_name,
                       const std::function<void()> &wrapped);
  void dump_state(std::ostream &os);
  // cudaq::state get_state();
  // cudaq::state get_state(const state_data &data);
  std::unique_ptr<cudaq::SimulationState> get_internal_state();
  simulation_precision get_precision() const;

  std::size_t allocateQudit(std::size_t numLevels = 2);
  std::vector<std::size_t> allocateQudits(std::size_t numQudits,
                                          std::size_t numLevels,
                                          const void *state,
                                          simulation_precision precision);
  std::vector<std::size_t> allocateQudits(std::size_t numQudits,
                                          std::size_t numLevels,
                                          const SimulationState *state);
  std::vector<std::size_t> allocateQudits(std::size_t numQudits,
                                          std::size_t numLevels = 2);

  void deallocate(std::size_t idx);
  void deallocate(const std::vector<std::size_t> &idxs);

  void apply(const std::vector<std::complex<double>> &matrixRowMajor,
             const std::vector<std::size_t> &controls,
             const std::vector<std::size_t> &targets,
             const traits::operation_metadata &metadata);

  void applyControlRegion(const std::vector<std::size_t> &controls,
                          const std::function<void()> &wrapped);
  void applyAdjointRegion(const std::function<void()> &wrapped);

  void reset(std::size_t qidx);
  void apply_exp_pauli(double theta, const std::vector<std::size_t> &controls,
                       const std::vector<std::size_t> &qubitIds,
                       const cudaq::spin_op_term &term);

  std::size_t mz(std::size_t idx, const std::string regName = "");
};

} // namespace cudaq::simulator::gpu
