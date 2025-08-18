/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "batch.h"
#include <any>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cudaq::remote {

// Forward declarations of types and classes
class Job;
class Task;

// A type-erased container for kernel arguments.
// This allows for different kernel signatures in the same batch.
using ArgPack = std::vector<std::any>;

/// @brief Represents a single kernel execution within a remote job.
///
/// Each task has a unique ID and holds the results for a specific kernel
/// execution. This allows for individual result retrieval from a batch job.
class Task {
private:
  std::string task_id;
  // `SampleResult`, `ObserveResult`, etc. Should probably
  // just make a class for this instead of using any or a variant.
  std::any result_data;
  bool is_successful = true;

public:
  Task(const std::string &id) : task_id(id) {}

  const std::string &get_id() const { return task_id; }
  bool is_success() const { return is_successful; }

  template <typename T>
  T get_result() const {
    return std::any_cast<T>(result_data);
  }

  // Simplified result setter.
  void set_result(std::any result) { result_data = result; }
  void set_failed(const std::string &error_message) {
    is_successful = false;
    // Store away any errors
  }
};

/// @brief Represents a single remote batch submission.
///
/// The Job object manages the entire batch of tasks. It is created upon
/// asynchronous submission and is the primary interface for querying the
/// job status and retrieving results.
class Job {
private:
  std::string job_id;
  std::string status = "submitted";
  std::vector<Task> tasks;

public:
  Job(const std::string &id) : job_id(id) {}

  /// @brief Get the unique ID of the job.
  const std::string &get_id() const { return job_id; }

  /// @brief Check if the job has completed.
  /// @note For this prototype, it always returns true to simplify the example.
  bool is_complete() const { return true; }

  /// @brief Get the current status of the job.
  const std::string &get_status() const { return status; }

  /// @brief Get the list of tasks associated with this job.
  const std::vector<Task> &get_tasks() const { return tasks; }

  // For prototype purposes, methods to populate the job with tasks.
  void add_task(Task &&task) { tasks.push_back(std::move(task)); }
  void set_status(const std::string &new_status) { status = new_status; }
};

/// @brief Submits a batch of kernels for remote sampling.
/// @param kernels The cudaq::batch of kernels to execute.
/// @param args A vector of argument packs, one for each kernel.
/// @param shots The number of shots to run.
/// @return A Job handle for the remote execution.
Job sample(const cudaq::batch &kernels, const std::vector<ArgPack> &args,
           int shots);

/// @brief Submits a batch of kernels for remote observation.
/// TODO: Handle potentially many different observables for each kernel, and
///       accept the operator type instead of the string dump.
/// @param kernels The cudaq::batch of kernels to execute.
/// @param args A vector of argument packs, one for each kernel.
/// @param observable The observable to measure.
/// @return A Job handle for the remote execution.
Job observe(const cudaq::batch &kernels, const std::vector<ArgPack> &args,
            const std::string &observable);

} // namespace cudaq::remote
