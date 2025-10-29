/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/qpu.h"
#include <concepts>
#include <type_traits>

namespace cudaq::traits {

// ============================================================================
// Local Trait (CRTP)
// ============================================================================

/// Trait for QPUs that execute locally (in-process)
///
/// This trait provides a generic launch interface that delegates to the
/// derived QPU's policy-specific launch implementations. It uses compile-time
/// checking to ensure the QPU supports the requested execution policy.
///
/// Example:
/// ```cpp
/// class my_local_qpu : public qpu<my_local_qpu,
///                                  local_trait<my_local_qpu>,
///                                  simulator<my_local_qpu>> {
/// public:
///   // Implement policy-specific launchers
///   template<typename... Args>
///   sample_result launch(const sample_policy& p, auto&& k, Args&&... args) {
///     // Implementation
///   }
/// };
/// ```
template <typename Derived>
class local_trait {
public:
  /// Generic launch method that delegates to derived class
  /// Uses SFINAE/requires to check if the derived class supports the policy
  template <typename Policy, typename QuantumKernel, typename... Args>
  auto launch(Policy &&policy, QuantumKernel &&kernel, Args &&...args) {
    // Check if derived class has a launch method for this policy
    if constexpr (requires(Derived &d) {
                    d.launch(std::forward<Policy>(policy),
                             std::forward<QuantumKernel>(kernel),
                             std::forward<Args>(args)...);
                  }) {
      return static_cast<Derived *>(this)->launch(
          std::forward<Policy>(policy), std::forward<QuantumKernel>(kernel),
          std::forward<Args>(args)...);
    } else {
      static_assert(
          always_false_v<Policy>,
          "\n\nThis local QPU does not support the given execution policy.\n"
          "Hint: Implement a 'launch(const PolicyType&, QuantumKernel&&, "
          "Args&&...)' method in your QPU class.\n");
    }
  }
};

// ============================================================================
// Local QPU Concept
// ============================================================================

/// Concept to check if a type is a local QPU
template <typename T>
concept LocalQPU =
    std::derived_from<std::decay_t<T>, traits::local_trait<std::decay_t<T>>>;



} // namespace cudaq::traits

