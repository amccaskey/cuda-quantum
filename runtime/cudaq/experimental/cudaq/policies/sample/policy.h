/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "sample_result.h"

#include "cudaq/qis/noise_model.h"
#include "cudaq/traits/local.h"

#include <cstddef>
#include <future>
#include <memory>

namespace cudaq {

// ============================================================================
// Sample Policy
// ============================================================================

/// Execution policy for sampling quantum circuit measurements
///
/// This policy instructs the QPU to execute the quantum kernel multiple times
/// (specified by `shots`) and return the measurement outcome statistics.
///
/// Example:
/// ```cpp
/// auto result = cudaq::launch(qpu, sample_policy{.shots = 1000}, kernel);
/// std::cout << "00: " << result.probability("00") << "\n";
/// ```
///
/// Example with noise:
/// ```cpp
/// cudaq::noise_model noise;
/// noise.add_all_qubit_channel<cudaq::gates::x<>>(cudaq::bit_flip_channel(.01));
/// auto result = cudaq::launch(qpu, sample_policy{.shots = 1000, .noise = &noise}, kernel);
/// ```
struct sample_policy {
  using result_type = sample_result;
  using execution_policy_tag = void;

  /// Number of times to execute and measure the circuit
  std::size_t shots = 1000;
  
  /// Optional noise model to apply during execution
  /// If nullptr, no noise is applied (ideal execution)
  const noise_model *noise = nullptr;

  /// Hidden friend for ADL-based dispatch
  /// This function is only found when sample_policy is passed to launch()
  template <typename QPU, typename QuantumKernel, typename... Args>
  friend auto launch_impl(QPU &qpu, const sample_policy &policy,
                          QuantumKernel &&kernel, Args &&...args)
      -> result_type {
    // Delegate to the QPU's launch method
    // The QPU must implement: launch(const sample_policy&, Kernel, Args...)
    return qpu.launch(policy, std::forward<QuantumKernel>(kernel),
                      std::forward<Args>(args)...);
  }
};

namespace async {
struct sample_policy {
  using result_type = std::future<sample_result>;
  using execution_policy_tag = void;

  /// Number of times to execute and measure the circuit
  std::size_t shots = 1000;
  
  /// Optional noise model to apply during execution
  const noise_model *noise = nullptr;

  /// We only allow async sampling like this (returning std::future) on LocalQPUs
  template <traits::LocalQPU QPU, typename QuantumKernel, typename... Args>
  friend auto launch_impl(QPU &qpu, const sample_policy &policy,
                          QuantumKernel &&kernel, Args &&...args)
      -> result_type {
    return std::async(std::launch::async, [&, shots = policy.shots, noise = policy.noise]() {
      QPU local_qpu;
      return local_qpu.launch(cudaq::sample_policy{.shots = shots, .noise = noise},
                              std::forward<QuantumKernel>(kernel),
                              std::forward<Args>(args)...);
    });
  }
};
} // namespace async
} // namespace cudaq
