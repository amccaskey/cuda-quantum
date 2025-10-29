/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <stdexcept>
#include <thread>

namespace cudaq {

/// Job status enumeration
enum class job_status {
  queued,     ///< Job is queued for execution
  running,    ///< Job is currently executing
  completed,  ///< Job has completed successfully
  failed,     ///< Job execution failed
  cancelled   ///< Job was cancelled
};

/// Convert job_status to string
inline std::string to_string(job_status status) {
  switch (status) {
    case job_status::queued: return "queued";
    case job_status::running: return "running";
    case job_status::completed: return "completed";
    case job_status::failed: return "failed";
    case job_status::cancelled: return "cancelled";
    default: return "unknown";
  }
}

/// Job handle for asynchronous remote quantum execution
///
/// A job represents an asynchronous quantum computation that may be
/// executing on a remote QPU. Users can poll for completion, wait for
/// results, or retrieve results once available.
///
/// Example:
/// ```cpp
/// auto job = launch(phantom_qpu, sample_policy{.shots = 1000}, kernel);
/// 
/// // Option 1: Poll manually
/// while (!job.is_complete()) {
///   std::this_thread::sleep_for(std::chrono::seconds(1));
/// }
/// auto result = job.get();
///
/// // Option 2: Block until complete
/// auto result = job.get();  // Blocks internally
/// ```
template <typename Result>
class job {
public:
  /// Callback type for job completion
  using completion_callback = std::function<void(const Result&)>;

  job() = default;

  /// Construct job with job ID and provider-specific retrieval logic
  template <typename RetrieveFn, typename StatusFn>
  job(std::string job_id, RetrieveFn retrieve_fn, StatusFn status_fn,
      std::chrono::milliseconds poll_interval = std::chrono::milliseconds(500))
      : job_id_(std::move(job_id)),
        poll_interval_(poll_interval),
        retrieve_fn_(std::make_shared<std::function<Result()>>(retrieve_fn)),
        status_fn_(std::make_shared<std::function<job_status()>>(status_fn)) {}

  /// Get the job ID
  std::string id() const { return job_id_; }

  /// Check if job is complete (non-blocking)
  bool is_complete() const {
    if (!status_fn_) {
      throw std::runtime_error("Job not initialized");
    }
    auto status = (*status_fn_)();
    return status == job_status::completed || 
           status == job_status::failed || 
           status == job_status::cancelled;
  }

  /// Get current job status (non-blocking)
  job_status status() const {
    if (!status_fn_) {
      throw std::runtime_error("Job not initialized");
    }
    return (*status_fn_)();
  }

  /// Wait for job completion and return result (blocking)
  Result get() {
    if (result_cached_) {
      return cached_result_;
    }

    if (!retrieve_fn_) {
      throw std::runtime_error("Job not initialized");
    }

    // Poll until complete
    while (!is_complete()) {
      std::this_thread::sleep_for(poll_interval_);
    }

    // Retrieve result
    cached_result_ = (*retrieve_fn_)();
    result_cached_ = true;
    return cached_result_;
  }

  /// Wait for job completion with timeout
  Result get(std::chrono::milliseconds timeout) {
    if (result_cached_) {
      return cached_result_;
    }

    if (!retrieve_fn_) {
      throw std::runtime_error("Job not initialized");
    }

    auto start = std::chrono::steady_clock::now();
    
    // Poll until complete or timeout
    while (!is_complete()) {
      auto elapsed = std::chrono::steady_clock::now() - start;
      if (elapsed >= timeout) {
        throw std::runtime_error("Job timeout: " + job_id_);
      }
      std::this_thread::sleep_for(poll_interval_);
    }

    // Retrieve result
    cached_result_ = (*retrieve_fn_)();
    result_cached_ = true;
    return cached_result_;
  }

  /// Check if result is cached
  bool has_result() const { return result_cached_; }

  /// Set completion callback (called when job completes)
  void on_complete(completion_callback callback) {
    callback_ = std::move(callback);
  }

private:
  std::string job_id_;
  std::chrono::milliseconds poll_interval_{500};
  bool result_cached_ = false;
  Result cached_result_;
  completion_callback callback_;
  
  // Type-erased functions for provider-specific logic
  std::shared_ptr<std::function<Result()>> retrieve_fn_;
  std::shared_ptr<std::function<job_status()>> status_fn_;
};

} // namespace cudaq

