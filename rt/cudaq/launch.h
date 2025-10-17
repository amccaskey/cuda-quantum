/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include <utility>

#include "cudaq/platform/config.h"

namespace cudaq {

// Fallback for direct kernel execution - found by ADL when no policy matches
template <typename QPU, typename QuantumKernel, typename... Args>
  requires std::invocable<QuantumKernel, Args...>
auto launch_impl(QPU &, QuantumKernel &&kernel, Args &&...args)
    -> std::invoke_result_t<QuantumKernel, Args...> {
  return kernel(std::forward<Args>(args)...);
}

// The main launch function that takes a user QPU as input
template <typename QPU, typename ExecutionPolicy, typename QuantumKernel,
          typename... Args>
auto launch(QPU &qpu, ExecutionPolicy &&policy, QuantumKernel &&kernel,
            Args &&...args) {
  return launch_impl(qpu, std::forward<ExecutionPolicy>(policy), kernel,
                     std::forward<Args>(args)...);
}

// The main launch function, allows user expression of QPU template type
template <typename QPU, typename ExecutionPolicy, typename QuantumKernel,
          typename... Args>
auto launch(ExecutionPolicy &&policy, QuantumKernel &&kernel, Args &&...args) {
  QPU qpu;
  return launch_impl(qpu, std::forward<ExecutionPolicy>(policy),
                     std::forward<Args>(args)...);
}

// Default launcher, will use compiler specified QPU 
template <typename ExecutionPolicy, typename QuantumKernel, typename... Args>
auto launch(ExecutionPolicy &&policy, QuantumKernel &&kernel, Args &&...args) {
  return launch<config::default_qpu>(std::forward<ExecutionPolicy>(policy),
                                     std::forward<Args>(args)...);
}
} // namespace cudaq

// Bring in all the default policies
#include "policies/policies.h"
