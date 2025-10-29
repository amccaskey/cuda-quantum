/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "sample_result.h"

#include "cudaq/traits/local.h"

#include <cstddef>
#include <future>

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
struct sample_policy {
  using result_type = sample_result;
  using execution_policy_tag = void;

  /// Number of times to execute and measure the circuit
  std::size_t shots = 1000;

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

  /// We only allow async sampling like this (returning std::future) on LocalQPUs
  template <traits::LocalQPU QPU, typename QuantumKernel, typename... Args>
  friend auto launch_impl(QPU &qpu, const sample_policy &policy,
                          QuantumKernel &&kernel, Args &&...args)
      -> result_type {
    return std::async(std::launch::async, [&, shots = policy.shots]() {
      QPU local_qpu;
      return local_qpu.launch(cudaq::sample_policy{.shots = shots},
                              std::forward<QuantumKernel>(kernel),
                              std::forward<Args>(args)...);
    });
  }
};
} // namespace async
} // namespace cudaq
