/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include <unordered_map>

#include "cutensornet.h"
#include "tensornet_state.h"
#include "tensornet_utils.h"
#include "timing_utils.h"

#include "common/SimulationState.h"

namespace nvqir {

class MPSSimulationState : public cudaq::SimulationState {

public:
  MPSSimulationState(TensorNetState *inState,
                     const std::vector<MPSTensor> &mpsTensors);

  MPSSimulationState(const MPSSimulationState &) = delete;
  MPSSimulationState &operator=(const MPSSimulationState &) = delete;
  MPSSimulationState(MPSSimulationState &&) noexcept = default;
  MPSSimulationState &operator=(MPSSimulationState &&) noexcept = default;

  virtual ~MPSSimulationState();

  std::complex<double> overlap(const cudaq::SimulationState &other) override;

  std::complex<double>
  getAmplitude(const std::vector<int> &basisState) override;
  std::size_t getNumQubits() const override;
  void dump(std::ostream &) const override {}
  cudaq::SimulationState::precision getPrecision() const override {
    return cudaq::SimulationState::precision::fp64;
  }

  Tensor getTensor(std::size_t tensorIdx = 0) const override;

  std::vector<Tensor> getTensors() const override;

  std::size_t getNumTensors() const override;

  void destroyState() override;

protected:
  void deallocate();
  void deallocateBackendStructures();
  std::complex<double>
  computeOverlap(const std::vector<MPSTensor> &m_mpsTensors,
                 const std::vector<MPSTensor> &mpsOtherTensors);

  TensorNetState *state = nullptr;
  std::vector<MPSTensor> m_mpsTensors;
  cutensornetNetworkDescriptor_t m_tnDescr;
  cutensornetContractionOptimizerConfig_t m_tnConfig;
  cutensornetContractionOptimizerInfo_t m_tnPath;
  cutensornetContractionPlan_t m_tnPlan;
  ScratchDeviceMem m_scratchPad;
  void *m_dOverlap{nullptr};
  bool m_allSet{false};
};

} // namespace nvqir