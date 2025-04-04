/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "device_ptr.h"

#include <cstddef>
#include <numeric>

namespace cudaq {
namespace config {
class TargetConfig;
}
namespace driver {

// FIXME what about
// query info on QPU architecture
// get device, query device info
// do we need a - class device {};

/// Initialize the CUDA-Q Driver API based on the
/// current user-selected target. The target defines
/// the QPU architecture, including the classical devices
/// and communication channels present.
void initialize(const config::TargetConfig &config);

// Allocate data of the given number of size bytes on the
// controller.
device_ptr malloc(std::size_t size);

/// Allocate data of the given number of size bytes on the
/// user-specified classical device. Return a device_ptr.
device_ptr malloc(std::size_t size, std::size_t deviceId);

/// Free the memory held by the given device_ptr.
void free(device_ptr &d);

// Copy the given src data into the QPU device data element.
void memcpy(device_ptr &dest, const void *src);

// Copy the data on QPU device to the given host pointer dest
void memcpy(void *dest, device_ptr &src);

/// Run any target-specific Quake compilation passes.
/// Returns a handle to the remotely JIT-ed code
handle load_kernel(const std::string &quake);

/// Launch the kernel remotely held at the given handle, with
/// the given runtime arguments.
launch_result launch_kernel(handle kernelHandle, device_ptr args);

} // namespace driver
} // namespace cudaq