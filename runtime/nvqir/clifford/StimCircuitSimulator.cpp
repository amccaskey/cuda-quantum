/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/
#if (defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wreturn-type"
#pragma GCC diagnostic ignored "-Wcast-qual"
#endif

#include "CircuitSimulator.h"
#include "Gates.h"
#include "stim.h"
#include <iostream>
#include <set>

namespace nvqir {

/// @brief The QppCircuitSimulator implements the CircuitSimulator
/// base class to provide a simulator delegating to the Q++ library from
/// https://github.com/softwareqinc/qpp.
class StimCircuitSimulator : public nvqir::CircuitSimulatorBase<double> {
protected:

  /// @brief Grow the state vector by one qubit.
  void addQubitToState() override {
   
  }

  /// @brief Override the default sized allocation of qubits
  /// here to be a bit more efficient than the default implementation
  void addQubitsToState(std::size_t count) override {
   
    return;
  }

  /// @brief Reset the qubit state.
  void deallocateStateImpl() override {
  }

  void applyGate(const GateApplicationTask &task) override {
  }

  /// @brief Set the current state back to the |0> state.
  void setToZeroState() override {
  }

  /// @brief Measure the qubit and return the result. Collapse the
  /// state vector.
  bool measureQubit(const std::size_t qubitIdx) override {
    
  }

public:
  StimCircuitSimulator() = default;
  virtual ~StimCircuitSimulator() = default;

  bool canHandleObserve() override {
   return false;
  }

  /// @brief Reset the qubit
  /// @param qubitIdx
  void resetQubit(const std::size_t qubitIdx) override {
    flushGateQueue();
  }

  /// @brief Sample the multi-qubit state.
  cudaq::ExecutionResult sample(const std::vector<std::size_t> &measuredBits,
                                const int shots) override {
  }

  cudaq::State getStateData() override {
    flushGateQueue();

  }

  std::string name() const override { return "clifford"; }
  NVQIR_SIMULATOR_CLONE_IMPL(StimCircuitSimulator)
};

} // namespace nvqir

#ifndef __NVQIR_QPP_TOGGLE_CREATE
/// Register this Simulator with NVQIR.
NVQIR_REGISTER_SIMULATOR(nvqir::StimCircuitSimulator, clifford)
#endif
