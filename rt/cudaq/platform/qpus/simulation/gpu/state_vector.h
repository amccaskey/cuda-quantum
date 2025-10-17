/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cudaq/host_config.h"
#include "cudaq/platform/qpu.h"

#include "cudaq/platform/traits/sampling.h"
#include "cudaq/platform/traits/simulator.h"

#include <stdio.h>

namespace cudaq::simulator::gpu {
class state_vector : public qpu<state_vector, traits::simulator<state_vector>,
                                traits::sample_trait<state_vector>> {
public:
  std::string name() const { return "gpu::state_vector"; }

  sample_result sample(std::size_t num_shots, const std::string &kernel_name,
                       const std::function<void()> &wrapped);

  state_vector();
  ~state_vector();

  void dump_state(std::ostream &os);
  cudaq::state get_state();
  cudaq::state get_state(const state_data &data);
  std::unique_ptr<cudaq::SimulationState>
  get_internal_state(const state_data &data);
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

  class Impl;

private:
  std::unique_ptr<Impl> pImpl;
};
} // namespace cudaq::simulator::gpu
