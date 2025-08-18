/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "remote.h"

#include <iostream>
#include <map>
#include <random>
#include <sstream>

namespace cudaq::remote {

// A simple utility to generate a unique ID.
std::string generate_id() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 15);

  std::stringstream ss;
  ss << std::hex;
  for (int i = 0; i < 32; i++) {
    ss << dis(gen);
    if (i == 7 || i == 11 || i == 15 || i == 19) {
      ss << "-";
    }
  }
  return ss.str();
}

// This function should really make an API call to a remote backend.
// For now, we just simulate the process.
Job sample(const cudaq::batch &kernels, const std::vector<ArgPack> &args,
           int shots) {

  if (kernels.size() != args.size()) {
    std::cerr << "Error: The number of argument packs must match the number of "
                 "kernels."
              << std::endl;
    return Job("error-job");
  }

  std::cout << "Submitting batch job for sampling with " << kernels.size()
            << " kernels and " << shots << " shots." << std::endl;

  Job job(generate_id());
  job.set_status("submitted");

  // Simulate creating a task for each kernel.
  const auto &kernel_handles = kernels.get_kernels();
  for (size_t i = 0; i < kernel_handles.size(); ++i) {
    Task task(generate_id());
    // Just simulate a simple result for the prototype.
    std::map<std::string, int> dummy_counts;
    dummy_counts["00"] = shots / 2;
    dummy_counts["11"] = shots / 2;
    task.set_result(dummy_counts);

    job.add_task(std::move(task));
  }
  job.set_status("completed");

  return job;
}

Job observe(const cudaq::batch &kernels, const std::vector<ArgPack> &args,
            const std::string &observable) {
  std::cout << "Submitting batch job for observation with " << kernels.size()
            << " kernels and observable " << observable << "." << std::endl;

  Job job(generate_id());
  job.set_status("completed");

  const auto &kernel_handles = kernels.get_kernels();
  for (size_t i = 0; i < kernel_handles.size(); ++i) {
    Task task(generate_id());
    // Just pass back some concrete expectation values.
    task.set_result(0.5);
    job.add_task(std::move(task));
  }

  return job;
}

} // namespace cudaq::remote
