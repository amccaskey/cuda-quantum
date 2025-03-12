/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once
#include "cudaq/platform/qpu.h"
#include ""
namespace cudaq {
class DriverQPU : public QPU {
public:
  DriverQPU() = default;

  cudaq::KernelThunkResultType
  launchKernel(const std::string &name, cudaq::KernelThunkType kernelFunc,
               void *args, std::uint64_t argsSize, std::uint64_t resultOffset,
               const std::vector<void *> &rawArgs) override {}

  void setExecutionContext(cudaq::ExecutionContext *context) override {}

  void resetExecutionContext() override {}
};

} // namespace cudaq