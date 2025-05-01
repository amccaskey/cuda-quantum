/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "qpu.h"

namespace cudaq::v2 {

namespace config {

/// @brief platform_config is an extension point for CUDA-Q
/// platform developers. Subtypes should implement configure_qpus
/// in order to configure the platform based on the target and
/// provided user information. Configure the platform implies
/// setting the platform_qpus reference.
class platform_config : public extension_point<platform_config> {
public:
  virtual void configure_qpus(std::vector<std::unique_ptr<qpu_handle>> &,
                              const platform_metadata &) = 0;
};
} // namespace config

/// @brief Initialize the platform from the given
/// target config file name.
/// @param targetConfigName
void initialize(const std::string &targetConfigName,
                const std::string &options);

void reset_platform(const std::string &cfg);

// Return the QPU at the given index
qpu_handle &get_qpu(std::size_t idx);

// Return the current active QPU
qpu_handle &get_qpu();

std::size_t get_num_qpus();

// Set the active qpu to the one at given index
void set_qpu(std::size_t idx);

void override_current_qpu(qpu_handle *q);
void reset_override_qpu();

} // namespace cudaq::v2

namespace cudaq {
extern "C" {

cudaq::KernelThunkResultType altLaunchKernel(const char *kernelName,
                                             cudaq::KernelThunkType kernelFunc,
                                             void *kernelArgs,
                                             std::uint64_t argsSize,
                                             std::uint64_t resultOffset);

cudaq::KernelThunkResultType
streamlinedLaunchKernel(const char *kernelName,
                        const std::vector<void *> &rawArgs);

cudaq::KernelThunkResultType
hybridLaunchKernel(const char *kernelName, cudaq::KernelThunkType kernel,
                   void *args, std::uint64_t argsSize,
                   std::uint64_t resultOffset,
                   const std::vector<void *> &rawArgs);
}
} // namespace cudaq
