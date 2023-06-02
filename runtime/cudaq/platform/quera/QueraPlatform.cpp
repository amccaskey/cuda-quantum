/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/ExecutionContext.h"
#include "common/Logger.h"
#include "cudaq/platform/qpu.h"
#include "cudaq/platform/quantum_platform.h"
#include "cudaq/qis/qubit_qis.h"
#include "cudaq/spin_op.h"
#include <fstream>

namespace {

class QueraPlatform : public cudaq::quantum_platform {
public:
  QueraPlatform() {}
  void launchKernel(std::string kernelName, void (*kernelFunc)(void *),
                    void *kernelArgs, std::uint64_t voidStarSize,
                    std::uint64_t resultOffset) override {
    cudaq::info("[quera] Launching kernel.");
    // Here I have kernelFunc == nullptr, but I do have
    // kernelArgs with everything I need.
  }
};
} // namespace

CUDAQ_REGISTER_PLATFORM(QueraPlatform, quera)
