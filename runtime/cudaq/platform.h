/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/builder/kernel_builder.h"
#include "cudaq/platform/platform.h"
#include <cxxabi.h>

namespace cudaq {
using QuantumTask = std::function<void()>;

struct quantum_platform {

  static std::string demangle(char const *mangled) {
    auto ptr = std::unique_ptr<char, decltype(&std::free)>{
        abi::__cxa_demangle(mangled, nullptr, nullptr, nullptr), std::free};
    return {ptr.get()};
  }

  std::size_t get_num_qpus() { return cudaq::get_num_qpus(); }
  std::size_t num_qpus() { return cudaq::get_num_qpus(); }

  bool is_remote(std::size_t qpu_id = 0) { return get_qpu(qpu_id).is_remote(); }
  RemoteCapabilities get_remote_capabilities(std::size_t qpu_id = 0) {
    return get_qpu(qpu_id).get_remote_capabilities();
  }
  bool is_simulator(std::size_t qpu_id = 0) {
    return get_qpu(qpu_id).is_simulator();
  }
  bool is_remote_simulator(std::size_t qpu_id = 0) {
    return get_qpu(qpu_id).get_remote_capabilities().isRemoteSimulator;
  }
  bool is_emulator(std::size_t qpu_id = 0) {
    return get_qpu(qpu_id).is_emulator();
  }
  bool supports_task_distribution(std::size_t qpu_id = 0) {
    return get_qpu(qpu_id).supports_task_distribution();
  }
  bool supports_explicit_measurements(std::size_t qpu_id = 0) {
    return get_qpu(qpu_id).supports_explicit_measurements();
  }
  std::optional<std::size_t> get_shots() { return std::nullopt; }
  void enqueueAsyncTask(std::size_t qpu_id, QuantumTask &&task) {
    get_qpu(qpu_id).enqueue_task(std::move(task));
  }

  void set_exec_ctx(ExecutionContext *ctx) { get_qpu().set_exec_ctx(ctx); }
  void reset_exec_ctx() { get_qpu().reset_exec_ctx(); }

  qpu &get(std::size_t qpu_id = 0) { return get_qpu(qpu_id); }
  void set_noise(const noise_model *nm) {
    if (auto *valid = get_qpu().as<noise_trait>())
      valid->set_noise(*nm);
  }
  void reset_noise() {
    if (auto *valid = get_qpu().as<noise_trait>())
      valid->reset_noise();
  }

  const noise_model *get_noise() {
    if (auto *valid = get_qpu().as<noise_trait>())
      return valid->get_noise();
    return nullptr;
  }
  void launchVQE(const std::string kernelName, const void *kernelArgs,
                 cudaq::gradient *gradient, const cudaq::spin_op &H,
                 cudaq::optimizer &optimizer, const int n_params,
                 const std::size_t shots) {
    if (auto *valid = get_qpu().as<mlir_launch_trait>())
      return valid->launch_vqe(kernelName, kernelArgs, gradient, H, optimizer,
                               n_params, shots);

    throw std::runtime_error("not a valid qpu, no launch_vqe method");
    return;
  }
};

// quantum_platform *getQuantumPlatformInternal();

/// @brief Return the quantum platform provided by the linked platform library
/// @return
static quantum_platform m_platform{};
inline quantum_platform &get_platform() { return m_platform; }

/// @brief Return the number of QPUs (at runtime)
inline std::size_t platform_num_qpus() { return get_platform().get_num_qpus(); }

/// @brief Return true if the quantum platform is remote.
inline bool is_remote_platform() { return get_platform().is_remote(); }

/// @brief Return true if the quantum platform is a remote simulator.
inline bool is_remote_simulator_platform() {
  return get_platform().is_remote_simulator();
}

/// @brief Return true if the quantum platform is emulated.
inline bool is_emulated_platform() { return get_platform().is_emulator(); }

} // namespace cudaq
