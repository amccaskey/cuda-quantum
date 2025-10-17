#pragma once

#include "base_policy.h"

#include "cudaq/utils/bit_table.h"

namespace cudaq {

struct explicit_measurements_policy {
  using result_type = bit_table;
  using execution_policy_tag = void;

  std::optional<std::size_t> shots = 1000;

  // Hidden friend for explicit measurements execution
  template <typename QuantumKernel, typename... Args>
  friend auto launch_impl(const explicit_measurements_policy &policy,
                          QuantumKernel &&kernel, Args &&...args)
      -> result_type {

    static_assert(std::is_void_v<std::invoke_result_t<QuantumKernel, Args...>>,
                  "Explicit measurements kernels must return void");

    auto &current_qpu = cudaq::get_qpu();
    if (auto *sampler = current_qpu.as<sample_trait>()) {
      auto kernelName = cudaq::getKernelName(kernel);
      auto quakeCode = cudaq::get_quake_by_name(kernelName, false);
      return sampler->sample_explicit_measurements(
          policy.shots.value(), quakeCode,
          [&, ... args = std::forward<Args>(args)]() mutable {
            kernel(std::forward<Args>(args)...);
          });
    }

    throw std::runtime_error(
        "current target does not support the explicit measurement policy.");
    return bit_table();
  }
};

} // namespace cudaq
