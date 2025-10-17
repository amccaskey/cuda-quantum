/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include <complex>
#include <concepts>

#include "cudaq/qis/state.h"
#include "cudaq/spin_op.h"

#include "utils.h"
#include "../type_traits.h"

namespace cudaq::traits {

template <typename Derived>
class simulator {
public:
  void dump_state(std::ostream &os) {
    return crtp_cast<Derived>(this)->dump_state(os);
  }

  cudaq::state get_state() { return crtp_cast<Derived>(this)->get_state(); }

  cudaq::state get_state(const state_data &data) {
    return crtp_cast<Derived>(this)->get_state(data);
  }

  std::unique_ptr<cudaq::SimulationState>
  get_internal_state(const state_data &data) {
    return crtp_cast<Derived>(this)->get_internal_state(data);
  }

  simulation_precision get_precision() const {
    return crtp_cast<const Derived>(this)->get_precision();
  }

  std::size_t allocateQudit(std::size_t numLevels = 2) {
    return crtp_cast<Derived>(this)->allocateQudit(numLevels);
  }

  std::vector<std::size_t> allocateQudits(std::size_t numQudits,
                                          std::size_t numLevels,
                                          const void *state,
                                          simulation_precision precision) {
    return crtp_cast<Derived>(this)->allocateQudits(numQudits, numLevels, state,
                                                    precision);
  }

  std::vector<std::size_t> allocateQudits(std::size_t numQudits,
                                          std::size_t numLevels,
                                          const SimulationState *state) {
    return crtp_cast<Derived>(this)->allocateQudits(numQudits, numLevels,
                                                    state);
  }

  std::vector<std::size_t> allocateQudits(std::size_t numQudits,
                                          std::size_t numLevels = 2) {
    return crtp_cast<Derived>(this)->allocateQudits(numQudits, numLevels);
  }

  void deallocate(std::size_t idx) {
    return crtp_cast<Derived>(this)->deallocate(idx);
  }

  void deallocate(const std::vector<std::size_t> &idxs) {
    return crtp_cast<Derived>(this)->deallocate(idxs);
  }

  void apply(const std::vector<std::complex<double>> &matrixRowMajor,
             const std::vector<std::size_t> &controls,
             const std::vector<std::size_t> &targets,
             const operation_metadata &metadata) {
    return crtp_cast<Derived>(this)->apply(matrixRowMajor, controls, targets,
                                           metadata);
  }

  void apply(const std::vector<std::complex<double>> &matrixRowMajor,
             const std::vector<std::size_t> &controls,
             const std::vector<std::size_t> &targets) {
    operation_metadata metadata("custom_op");
    apply(matrixRowMajor, controls, targets, metadata);
  }

  void applyControlRegion(const std::vector<std::size_t> &controls,
                          const std::function<void()> &wrapped) {
    return crtp_cast<Derived>(this)->applyControlRegion(controls, wrapped);
  }

  void applyAdjointRegion(const std::function<void()> &wrapped) {
    return crtp_cast<Derived>(this)->applyAdjointRegion(wrapped);
  }

  void reset(std::size_t qidx) { return crtp_cast<Derived>(this)->reset(qidx); }

  void apply_exp_pauli(double theta, const std::vector<std::size_t> &controls,
                       const std::vector<std::size_t> &qubitIds,
                       const cudaq::spin_op_term &term) {
    return crtp_cast<Derived>(this)->apply_exp_pauli(theta, controls, qubitIds,
                                                     term);
  }

  std::size_t mz(std::size_t idx, const std::string regName = "") {
    return crtp_cast<Derived>(this)->mz(idx, regName);
  }

  std::vector<std::size_t> mz(const std::vector<std::size_t> &qubits) {
    std::vector<std::size_t> ret;
    for (auto &q : qubits)
      ret.push_back(mz(q, ""));
    return ret;
  }
};
} // namespace cudaq::traits

namespace cudaq {
#ifndef CUDAQ_NO_STD20

template <typename T>
concept SimulatorQPU = requires {
  // Type requirement: T must derive from sample_trait<T> (CRTP pattern)
  requires std::derived_from<std::decay_t<T>, traits::simulator<T>>;
};
#endif

template <typename T>
bool is_simulator(T &&t) {
  return std::is_base_of_v<traits::simulator<std::decay_t<T>>, std::decay_t<T>>;
}

struct kernel_simulator_api {
  std::function<std::size_t()> q_allocator;
  std::function<void(std::size_t)> q_deallocator;
  std::function<void(const std::vector<std::complex<double>> &,
                     const std::vector<std::size_t> &,
                     const std::vector<std::size_t> &,
                     const traits::operation_metadata &)>
      q_applicator;
};

static std::unique_ptr<kernel_simulator_api> m_kernel_api;

template <typename T>
void set_kernel_api(traits::simulator<T> &simulator) {
  printf("Setting the allocator\n");
  m_kernel_api = std::make_unique<kernel_simulator_api>(
      [&]() -> std::size_t { return simulator.allocateQudit(2); },
      [&](std::size_t idx) { simulator.deallocate(idx); },
      [&](const std::vector<std::complex<double>> &matrixRowMajor,
          const std::vector<std::size_t> &controls,
          const std::vector<std::size_t> &targets,
          const traits::operation_metadata &metadata) {
        simulator.apply(matrixRowMajor, controls, targets, metadata);
      });
}

} // namespace cudaq
