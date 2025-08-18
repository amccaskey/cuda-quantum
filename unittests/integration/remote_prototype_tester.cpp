/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <any>
#include <gtest/gtest.h>
#include <map>
#include <string>
#include <vector>

#include <cudaq.h>

// A simple mock kernel for testing purposes.
auto __qpu__ mock_kernel_1 = [](double angle) {};
auto __qpu__ mock_kernel_2 = []() {};

TEST(RemoteTester, idFormat) {
  auto id = cudaq::remote::generate_id();
  ASSERT_EQ(id.length(), 36) << "id should have a length of 36.";
}

TEST(RemoteTester, JobAndTaskCreation) {
  cudaq::remote::Task task("task-123");
  ASSERT_EQ(task.get_id(), "task-123") << "Task ID should be set correctly.";
  ASSERT_TRUE(task.is_success()) << "New task should be successful by default.";

  cudaq::remote::Job job("job-abc");
  ASSERT_EQ(job.get_id(), "job-abc") << "Job ID should be set correctly.";
  ASSERT_EQ(job.get_status(), "submitted")
      << "New job status should be 'submitted'.";
  ASSERT_TRUE(job.get_tasks().empty()) << "New job should have no tasks.";
}

TEST(RemoteTester, SampleApiErrorHandling) {
  // Create a batch with two kernels.
  auto kernels = cudaq::batch(mock_kernel_1, mock_kernel_2);

  // Provide only one argument pack, which should cause an error.
  std::vector<cudaq::remote::ArgPack> args = {{std::any(0.1)}};

  // The function should return a Job with an "error-job" ID.
  auto job = cudaq::remote::sample(kernels, args, 100);
  ASSERT_EQ(job.get_id(), "error-job")
      << "sample() should return an error job on argument mismatch.";
}

TEST(RemoteTester, SampleApiSuccess) {
  // Create a batch with two kernels.
  auto kernels = cudaq::batch(mock_kernel_1, mock_kernel_2);
  std::vector<cudaq::remote::ArgPack> args = {{std::any(0.1)}, {}};

  // Execute the sample function.
  auto job = cudaq::remote::sample(kernels, args, 1000);

  // Verify the job properties.
  ASSERT_NE(job.get_id(), "error-job")
      << "Job ID should be a valid id on success.";
  ASSERT_EQ(job.get_status(), "completed")
      << "Job status should be 'completed' in this prototype.";
  ASSERT_TRUE(job.is_complete()) << "is_complete() should return true.";
  ASSERT_EQ(job.get_tasks().size(), 2) << "The job should contain two tasks.";

  // Verify the simulated results of the first task.
  const auto &task_1 = job.get_tasks()[0];
  ASSERT_TRUE(task_1.is_success()) << "Task 1 should be successful.";
  auto counts_1 = task_1.get_result<std::map<std::string, int>>();
  ASSERT_EQ(counts_1.size(), 2) << "Simulated counts should have 2 entries.";
  ASSERT_EQ(counts_1.at("00"), 500) << "Simulated count for 00 should be 500.";
  ASSERT_EQ(counts_1.at("11"), 500) << "Simulated count for 11 should be 500.";

  // Verify the simulated results of the second task.
  const auto &task_2 = job.get_tasks()[1];
  ASSERT_TRUE(task_2.is_success()) << "Task 2 should be successful.";
  auto counts_2 = task_2.get_result<std::map<std::string, int>>();
  ASSERT_EQ(counts_2.size(), 2) << "Simulated counts should have 2 entries.";
  ASSERT_EQ(counts_2.at("00"), 500) << "Simulated count for 00 should be 500.";
  ASSERT_EQ(counts_2.at("11"), 500) << "Simulated count for 11 should be 500.";
}

TEST(RemoteTester, ObserveApiSuccess) {
  // Create a batch with a single kernel.
  auto kernels = cudaq::batch(mock_kernel_2);
  std::vector<cudaq::remote::ArgPack> args = {{}};

  // Execute the observe function.
  auto job = cudaq::remote::observe(kernels, args, "Z0");

  // Verify the job properties.
  ASSERT_NE(job.get_id(), "error-job")
      << "Job ID should be a valid id on success.";
  ASSERT_EQ(job.get_status(), "completed")
      << "Job status should be 'completed'.";
  ASSERT_EQ(job.get_tasks().size(), 1) << "The job should contain one task.";

  // Verify the simulated result of the task.
  const auto &task = job.get_tasks()[0];
  ASSERT_TRUE(task.is_success()) << "Task should be successful.";
  double expectation_value = task.get_result<double>();
  ASSERT_EQ(expectation_value, 0.5)
      << "Simulated expectation value should be 0.5.";
}
