/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/policies/sample/policy.h"
#include "cudaq/policies/sample/sample_result.h"
#include "cudaq/qpu.h"
#include "cudaq/traits/remote.h"
#include "cudaq/utils/job.h"

#include <atomic>
#include <map>
#include <mutex>
#include <sstream>
#include <string>

namespace cudaq::mock {

/// @brief Phantom - A mock remote quantum execution backend for testing
///
/// Phantom is a test-only QPU that simulates remote quantum execution without
/// requiring an actual remote service. It provides job-based asynchronous
/// execution for testing the remote execution infrastructure.
///
/// **NOTE**: This is a test utility and should NOT be used in production code.
///
/// Example usage in tests:
/// ```cpp
/// cudaq::mock::phantom qpu("http://localhost:5000");
/// auto job = cudaq::launch(qpu, cudaq::sample_policy{.shots = 1000},
/// my_kernel);
///
/// // Poll for completion
/// while (!job.is_complete()) {
///   std::this_thread::sleep_for(std::chrono::seconds(1));
/// }
/// auto results = job.get();
/// ```
class phantom : public qpu<phantom, traits::remote_trait<phantom>> {
  using BaseType = qpu<phantom, traits::remote_trait<phantom>>;

private:
  /// Remote service endpoint (e.g., URL, address)
  std::string endpoint_ = "";

  /// Authentication token (if required)
  std::string auth_token_ = "";

  /// Job storage for testing/development
  struct job_data {
    job_status status;
    sample_result result;
    std::string kernel_name;
    std::size_t shots;
  };

  std::map<std::string, job_data> jobs_;
  std::mutex jobs_mutex_;
  std::atomic<std::size_t> job_counter_{0};

  // Shared pointers for safe closure capture
  std::shared_ptr<std::map<std::string, job_data>> jobs_ptr_;
  std::shared_ptr<std::mutex> jobs_mutex_ptr_;

public:
  /// @brief Default constructor
  phantom() : BaseType() {}

  /// @brief Constructor with endpoint
  phantom(const std::string &endpoint) : BaseType(), endpoint_(endpoint) {}

  /// @brief Constructor with configuration
  phantom(const qpu_configuration &config) : BaseType(config) {}

  /// @brief Get the name of this QPU
  std::string name() const { return "phantom"; }

  /// @brief Get the endpoint
  std::string endpoint() const {
    return extract_config<std::string>(m_configuration, "endpoint")
        .value_or("local");
  }

  // ============================================================================
  // ADL Launch Implementation - Hidden Friend Pattern
  // ============================================================================

  /// @brief Launch implementation for sample_policy - returns job handle
  /// This is found via ADL when cudaq::launch() is called
  template <typename QuantumKernel, typename... Args>
  friend auto launch_impl(phantom &qpu, const sample_policy &policy,
                          QuantumKernel &&kernel, Args &&...args) {

    // Submit job and get job ID
    std::string job_id = qpu.submit(policy, std::forward<QuantumKernel>(kernel),
                                    std::forward<Args>(args)...);

    // For job closures, we need shared access to the job storage
    // In production, these would make HTTP/gRPC calls to endpoint
    // For now, share the internal job map
    auto jobs_ptr = qpu.get_jobs_ptr();
    auto jobs_mutex_ptr = qpu.get_jobs_mutex_ptr();

    // Create job handle with closures for status and retrieval
    auto retrieve_fn = [jobs_ptr, jobs_mutex_ptr, job_id]() {
      std::lock_guard<std::mutex> lock(*jobs_mutex_ptr);
      auto it = jobs_ptr->find(job_id);
      if (it == jobs_ptr->end() || it->second.status != job_status::completed) {
        throw std::runtime_error("Job not completed: " + job_id);
      }
      return it->second.result;
    };

    auto status_fn = [jobs_ptr, jobs_mutex_ptr, job_id]() {
      std::lock_guard<std::mutex> lock(*jobs_mutex_ptr);
      auto it = jobs_ptr->find(job_id);
      if (it == jobs_ptr->end()) {
        throw std::runtime_error("Unknown job ID: " + job_id);
      }
      return it->second.status;
    };

    return job<sample_result>(job_id, retrieve_fn, status_fn,
                              std::chrono::milliseconds(500));
  }

  // ============================================================================
  // Remote Trait Interface - Required Methods
  // ============================================================================

  /// @brief Submit a sampling job to the remote service
  template <typename QuantumKernel, typename... Args>
  std::string submit(const sample_policy &policy, QuantumKernel &&kernel,
                     Args &&...args) {
    if (auto maybe_endpoint =
            extract_config<std::string>(m_configuration, "endpoint")) {
      endpoint_ = maybe_endpoint.value();
    }

    if (auto maybe_auth_token =
            extract_config<std::string>(m_configuration, "auth_token")) {
      auth_token_ = maybe_auth_token.value();
    }
    
    // Generate unique job ID
    std::stringstream ss;
    ss << "phantom-" << job_counter_++;
    std::string job_id = ss.str();

    // In a real implementation, this would:
    // 1. Serialize the quantum kernel to IR
    // 2. Send HTTP/gRPC request to endpoint_
    // 3. Receive job ID from remote service

    // For now, store job locally for testing
    // Get shared pointers before spawning any async execution
    auto jobs_ptr = get_jobs_ptr();
    auto jobs_mutex_ptr = get_jobs_mutex_ptr();

    {
      std::lock_guard<std::mutex> lock(*jobs_mutex_ptr);
      (*jobs_ptr)[job_id] =
          job_data{.status = job_status::queued,
                   .result = sample_result(),
                   .kernel_name = "kernel", // Would extract real name
                   .shots = policy.shots};
    }

    // Simulate immediate execution for testing
    // In production, the remote service handles this
    simulate_execution(job_id, std::forward<QuantumKernel>(kernel),
                       policy.shots, jobs_ptr, jobs_mutex_ptr,
                       std::forward<Args>(args)...);

    return job_id;
  }

  /// @brief Get the status of a submitted job
  job_status get_job_status(const std::string &job_id) {
    auto jobs_ptr = get_jobs_ptr();
    auto jobs_mutex_ptr = get_jobs_mutex_ptr();
    std::lock_guard<std::mutex> lock(*jobs_mutex_ptr);

    auto it = jobs_ptr->find(job_id);
    if (it == jobs_ptr->end()) {
      throw std::runtime_error("Unknown job ID: " + job_id);
    }

    return it->second.status;
  }

  /// @brief Retrieve results for a completed job
  sample_result retrieve_results(const std::string &job_id) {
    auto jobs_ptr = get_jobs_ptr();
    auto jobs_mutex_ptr = get_jobs_mutex_ptr();
    std::lock_guard<std::mutex> lock(*jobs_mutex_ptr);

    auto it = jobs_ptr->find(job_id);
    if (it == jobs_ptr->end()) {
      throw std::runtime_error("Unknown job ID: " + job_id);
    }

    if (it->second.status != job_status::completed) {
      throw std::runtime_error("Job not completed: " + job_id);
    }

    return it->second.result;
  }

  /// @brief Get shared pointer to jobs map (for closure capture)
  std::shared_ptr<std::map<std::string, job_data>> get_jobs_ptr() {
    if (!jobs_ptr_) {
      jobs_ptr_ = std::make_shared<std::map<std::string, job_data>>();
      // Move existing jobs into shared storage
      *jobs_ptr_ = std::move(jobs_);
    }
    return jobs_ptr_;
  }

  /// @brief Get shared pointer to jobs mutex (for closure capture)
  std::shared_ptr<std::mutex> get_jobs_mutex_ptr() {
    if (!jobs_mutex_ptr_) {
      jobs_mutex_ptr_ = std::make_shared<std::mutex>();
    }
    return jobs_mutex_ptr_;
  }

private:
  /// @brief Simulate remote execution (for testing)
  /// In production, this happens on the remote service
  template <typename QuantumKernel, typename... Args>
  void simulate_execution(
      const std::string &job_id, QuantumKernel &&kernel, std::size_t shots,
      std::shared_ptr<std::map<std::string, job_data>> jobs_ptr,
      std::shared_ptr<std::mutex> jobs_mutex_ptr, Args &&...args) {
    // For testing: immediately mark as running then completed
    // In production: remote service processes the job

    // Simulate simple measurement results
    // For a single-qubit Hadamard, we'd expect ~50/50 distribution
    ExecutionResult exec_result;
    exec_result.appendResult("0", shots / 2);
    exec_result.appendResult("1", shots - (shots / 2));
    sample_result result(exec_result);

    {
      std::lock_guard<std::mutex> lock(*jobs_mutex_ptr);
      (*jobs_ptr)[job_id].status = job_status::running;
    }

    // Simulate processing delay
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    {
      std::lock_guard<std::mutex> lock(*jobs_mutex_ptr);
      (*jobs_ptr)[job_id].status = job_status::completed;
      (*jobs_ptr)[job_id].result = result;
    }
  }
};

} // namespace cudaq::mock

