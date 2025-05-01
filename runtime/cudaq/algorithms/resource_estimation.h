/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/ExecutionContext.h"
#include "common/Resources.h"
#include "cudaq/platformv2/platform.h"

namespace cudaq {

/// @brief Given any CUDA-Q kernel and its associated runtime arguments,
/// return the resources that this kernel will use. This does not execute the
/// circuit simulation, it only traces the quantum operation calls and returns
/// a `resources` type that allows the programmer to query the number and types
/// of operations in the kernel.
template <typename QuantumKernel, typename... Args>
auto estimate_resources(QuantumKernel &&kernel, Args &&...args) {
  ExecutionContext context("tracer");
  auto &platform = v2::get_qpu();
  platform.set_execution_context(&context);
  kernel(args...);
  platform.reset_execution_context();
  return cudaq::Resources::compute(context.kernelTrace);
}

} // namespace cudaq
