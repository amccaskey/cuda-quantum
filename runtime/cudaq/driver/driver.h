/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "channel.h"

#include <cstddef>
#include <numeric>

namespace cudaq::driver {

// FIXME what about
// query info on QPU architecture
// get device, query device info
// do we need a - class device {};

/// Initialize the CUDA-Q Driver API based on the
/// current user-selected target. The target defines
/// the QPU architecture, including the classical devices
/// and communication channels present.
void initialize(const config::TargetConfig &config);

/// Allocate data of the given number of size bytes on the
/// user-specified classical device. Return a device_ptr.
device_ptr malloc(std::size_t size, std::size_t deviceId = host_qpu_channel_id);

/// Free the memory held by the given device_ptr.
void free(device_ptr &d);

// Copy the given src data into the QPU device data element.
void memcpy(device_ptr &dest, const void *src);

// Copy the data on QPU device to the given host pointer dest
void memcpy(void *dest, device_ptr &src);

/// Run any target-specific Quake compilation passes.
/// Returns a handle to the remotely JIT-ed code
handle compile_kernel(const std::string &quake);

/// Launch the kernel remotely held at the given handle, with
/// the given runtime arguments.
error_code launch_kernel(handle kernelHandle, device_ptr args);

// need to be able to get the result

// sample, observe, run?

} // namespace cudaq::driver
