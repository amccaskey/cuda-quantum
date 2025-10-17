/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once
// FIXME remove this
#include <stdio.h>

#include "../base_policy.h"

#include "cudaq/platform/traits/sampling.h"
#include "cudaq/platform/traits/simulator.h"

namespace cudaq {
namespace details {
template <typename QuantumKernel, typename... Args>
concept IsValidSampleKernel =
    std::invocable<QuantumKernel, Args...> &&
    std::same_as<std::invoke_result_t<QuantumKernel, Args...>, void>;
} // namespace details

// Policy types for behavior configuration
struct sample_policy {
  using result_type = sample_result;
  std::optional<std::size_t> shots = 1000;
  // Hidden friend - only found via ADL, not normal lookup
  template <SamplingQPU QPU, typename QuantumKernel, typename... Args>
  friend auto launch_impl(QPU &qpu, const sample_policy &policy,
                          QuantumKernel &&kernel, Args &&...args)
      -> result_type {
    // Perform required runtime checks on kernel code to
    // assert that this is a valid kernel to sample
    printf("QPU %s ", qpu.name().c_str());
    auto kernelName = ""; // get kernel name

    if (cudaq::is_simulator(qpu)) {
      return qpu.sample(policy.shots.value_or(100), kernelName, [&]() {
        cudaq::set_kernel_api(qpu);
        kernel(args...);
      });
    }

    // This QPU is not local simulation, and we assume 
    // it can handle all sample requests on its own
    return qpu.sample(policy.shots.value_or(100), kernelName,
                      [&]() { kernel(args...); });
  }
};

} // namespace cudaq
