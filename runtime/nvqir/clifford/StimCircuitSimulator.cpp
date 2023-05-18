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
#pragma GCC diagnostic ignored "-Wdeprecated-copy"
#endif

#include "CircuitSimulator.h"
#include "Gates.h"
#include "stim.h"
#include <iostream>
#include <set>

namespace nvqir {

class StimCircuitSimulator : public nvqir::CircuitSimulatorBase<double> {
protected:
  stim::Circuit circuit;

  /// @brief Grow the state vector by one qubit.
  void addQubitToState() override {
    // do nothing
  }

  /// @brief Override the default sized allocation of qubits
  /// here to be a bit more efficient than the default implementation
  void addQubitsToState(std::size_t count) override {
    // do nothing
  }

  /// @brief Reset the qubit state.
  void deallocateStateImpl() override {}

  void applyGate(const GateApplicationTask &task) override {
    if (task.controls.size() > 1)
      throw std::runtime_error("Non-clifford gate, too many controls.");

    std::vector<uint32_t> targets;
    std::transform(task.targets.begin(), task.targets.end(),
                   std::back_inserter(targets),
                   [](auto &&el) { return static_cast<uint32_t>(el); });
    if (task.operationName == "h") {
      circuit.safe_append_u("H", targets);
      return;
    }

    if (task.operationName == "x") {
      if (!task.controls.empty())
        targets.insert(targets.begin(), task.controls[0]);

      circuit.safe_append_u("CX", targets);
      return;
    }
  }

  /// @brief Set the current state back to the |0> state.
  void setToZeroState() override {}

  /// @brief Measure the qubit and return the result. Collapse the
  /// state vector.
  bool measureQubit(const std::size_t qubitIdx) override {
    // do nothing
  }

public:
  StimCircuitSimulator() = default;
  virtual ~StimCircuitSimulator() = default;

  bool canHandleObserve() override { return false; }

  /// @brief Reset the qubit
  /// @param qubitIdx
  void resetQubit(const std::size_t qubitIdx) override { flushGateQueue(); }

  /// @brief Sample the multi-qubit state.
  cudaq::ExecutionResult sample(const std::vector<std::size_t> &measuredBits,
                                const int shots) override {
    auto localCircuit = circuit;
    for (auto b : measuredBits)
      localCircuit.safe_append_u("MZ", {static_cast<uint32_t>(b)});

    cudaq::ExecutionResult counts;
    std::random_device dev;
    std::mt19937_64 rng(dev());
    std::size_t numSamples = static_cast<std::size_t>(shots);
    auto ref = stim::TableauSimulator::reference_sample_circuit(localCircuit);
    auto r = stim::sample_batch_measurements(localCircuit, ref, numSamples, rng,
                                             true);
    auto numMz = localCircuit.count_measurements();
    for (int i = 0; i < shots; i++) {
      std::string str = "";
      for (decltype(numMz) j = 0; j < numMz; j++)
        str += std::to_string(r[i][j]);
      counts.appendResult(str, 1);
    }

    return counts;
  }

  cudaq::State getStateData() override { flushGateQueue(); }

  std::string name() const override { return "clifford"; }
  NVQIR_SIMULATOR_CLONE_IMPL(StimCircuitSimulator)
};

} // namespace nvqir

#ifndef __NVQIR_QPP_TOGGLE_CREATE
/// Register this Simulator with NVQIR.
NVQIR_REGISTER_SIMULATOR(nvqir::StimCircuitSimulator, clifford)
#endif
