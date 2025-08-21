/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "qpu.h"
#include <map>

namespace cudaq {

/// @brief Initialize the platform from the given
/// target config file name.
/// @param targetConfigName
void initialize(const std::string &targetConfigName,
                const std::string &options);

// Return the QPU at the given index
qpu &get_qpu(std::size_t idx);
qpu *get_qpu_ptr();

// Return a user-specified qpu
qpu &create_qpu(const std::string &qpu_name, const heterogeneous_map &options);

// push an acquired qpu onto the current execution stack, 
// this will make the top of the stack the preferred qpu 
// for all further public API calls
void push_qpu(qpu *qpuToPushOnStack);

// remove a qpu from the stack. 
void pop_qpu();

// Return the current active QPU
qpu &get_qpu();

// return the number of qpus 

std::size_t get_num_qpus();

// In a multi-qpu setting, 
// set the active qpu to the one at given index
void set_qpu(std::size_t idx);

} // namespace cudaq

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
