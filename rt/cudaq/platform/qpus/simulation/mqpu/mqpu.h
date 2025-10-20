/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include <future>

#include "cudaq/platform/qpu.h"

#include "cudaq/platform/qpus/simulation/gpu/state_vector.h"
#include "cudaq/platform/traits/batch.h"

namespace cudaq::simulator::mqpu {
namespace details {
int get_gpu_device_count();
void set_gpu_device(int);
} // namespace details

template <typename SimulatorType>
class base_mqpu : public qpu<base_mqpu<SimulatorType>,
                             traits::batch_trait<base_mqpu<SimulatorType>>> {
protected:
  int num_qpus;
  int num_gpus; // Actual number of available GPUs
  std::queue<int> available_qpus;
  std::mutex pool_mutex;
  std::condition_variable qpu_available;

  int acquire_qpu() {
    std::unique_lock<std::mutex> lock(pool_mutex);
    qpu_available.wait(lock, [this] { return !available_qpus.empty(); });
    auto idx = available_qpus.front();
    available_qpus.pop();
    return idx;
  }

  void release_qpu(int idx) {
    std::lock_guard<std::mutex> lock(pool_mutex);
    available_qpus.push(idx);
    qpu_available.notify_one();
  }

  int get_gpu_device_count() { return details::get_gpu_device_count(); }

  // Map QPU index to actual GPU device ID
  int qpu_to_gpu_device(int qpu_idx) { return qpu_idx % num_gpus; }

  void internal_config(int qpu_count = 0) {
    num_gpus = get_gpu_device_count();
    if (num_gpus == 0)
      throw std::runtime_error("gpus for simulation not detected.");

    // Allow more simulators than GPUs, but default to number of GPUs
    num_qpus = qpu_count == 0 ? num_gpus : qpu_count;

    for (int i = 0; i < num_qpus; ++i)
      available_qpus.push(i);
  }

public:
  using base_type = qpu<base_mqpu<SimulatorType>,
                        traits::batch_trait<base_mqpu<SimulatorType>>>;
  void configure(const heterogeneous_map &config) {}

  base_mqpu(const heterogeneous_map &config) : base_type(config) {
    internal_config();
  }
  base_mqpu(int qpu_count = 0) { internal_config(qpu_count); }

  int get_num_qpus() { return num_qpus; }

  template <typename ExecutionPolicy, typename QuantumKernel, typename... Args>
  auto execute_batch(ExecutionPolicy &&policy, QuantumKernel &&kernel,
                     const std::vector<std::tuple<Args...>> &task_args)
      -> std::vector<
          typename std::remove_cvref_t<ExecutionPolicy>::result_type> {
    std::vector<
        std::future<typename std::remove_cvref_t<ExecutionPolicy>::result_type>>
        futures;

    for (const auto &args : task_args) {
      futures.push_back(std::async(std::launch::async, [&, args]() {
        auto qpu_idx = acquire_qpu();
        auto gpu_device_id = qpu_to_gpu_device(qpu_idx); // Map to actual GPU
        details::set_gpu_device(gpu_device_id);
        SimulatorType qpu;
        auto result = std::apply(
            [&](auto &&...captured_args) -> sample_result {
              // ADL, so nice...
              return launch_impl(qpu, policy, kernel, captured_args...);
            },
            args);

        release_qpu(qpu_idx);
        return result;
      }));
    }

    std::vector<sample_result> results;
    for (auto &future : futures)
      results.push_back(future.get());

    return results;
  }

  // Deduction helper that accepts initializer list
  template <typename ExecutionPolicy, typename QuantumKernel, typename... Args>
  auto execute_batch(ExecutionPolicy &&policy, QuantumKernel &&kernel,
                     std::initializer_list<std::tuple<Args...>> task_args)
      -> std::vector<typename ExecutionPolicy::result_type> {
    return execute_batch(std::forward<ExecutionPolicy>(policy),
                         std::forward<QuantumKernel>(kernel),
                         std::vector<std::tuple<Args...>>(task_args));
  }
};

// Enumerate all simulators that can be used MQPU
class state_vector : public base_mqpu<cudaq::simulator::gpu::state_vector> {
public:
  using base_mqpu::base_mqpu;
};
// Add others...

} // namespace cudaq::simulator::mqpu
