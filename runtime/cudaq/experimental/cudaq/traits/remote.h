/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/utils/job.h"
#include "cudaq/policies/sample/policy.h"

#include <concepts>
#include <type_traits>
#include <string> 

namespace cudaq::traits {

/// CRTP trait for remote QPU execution
///
/// QPUs that execute remotely (cloud services, hardware backends) should
/// inherit from this trait. Remote execution always returns a job handle
/// that users can poll or wait on for results.
///
/// The remote_trait generalizes job submission across all execution policies.
/// Each policy's result_type is wrapped in a job<> for asynchronous retrieval.
///
/// Example:
/// ```cpp
/// class phantom : public qpu<phantom, remote_trait<phantom>> {
///   // Must implement these for each supported policy:
///   std::string submit(const sample_policy& policy, auto&& kernel, auto&&... args);
///   job_status get_job_status(const std::string& job_id);
///   sample_result retrieve_results(const std::string& job_id);
/// };
/// ```
template <typename Derived>
class remote_trait {
public:
  /// Launch implementation for remote sampling
  /// Always returns a job<sample_result> handle
  template <typename QuantumKernel, typename... Args>
  auto launch(const sample_policy &policy, QuantumKernel &&kernel,
              Args &&...args) {
    auto *derived = static_cast<Derived *>(this);

    // Submit job and get job ID
    std::string job_id =
        derived->submit(policy, std::forward<QuantumKernel>(kernel),
                       std::forward<Args>(args)...);

    // Create job handle with closures for status and retrieval
    auto retrieve_fn = [derived, job_id]() {
      return derived->retrieve_results(job_id);
    };

    auto status_fn = [derived, job_id]() {
      return derived->get_job_status(job_id);
    };

    // Poll interval: use 500ms default for now
    // Could be made configurable per-provider
    return job<sample_result>(job_id, retrieve_fn, status_fn, 
                              std::chrono::milliseconds(500));
  }

  /// Future: Support for other execution policies
  /// Each policy type can be specialized to return job<Policy::result_type>
  ///
  /// Example for observe_policy:
  /// ```cpp
  /// auto launch(const observe_policy& policy, ...) {
  ///   std::string job_id = derived->submit(policy, ...);
  ///   return job<observe_result>(job_id, retrieve_fn, status_fn);
  /// }
  /// ```
};

/// Concept for remote QPUs
template <typename T>
concept RemoteQPU =
    std::derived_from<std::decay_t<T>, traits::remote_trait<std::decay_t<T>>>;

} // namespace cudaq::traits
