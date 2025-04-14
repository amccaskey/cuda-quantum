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

namespace cudaq {
namespace config {
class TargetConfig;
}
namespace driver {

/// @brief Initialize the CUDA-Q Driver API based on the
/// current user-selected target. The target defines
/// the QPU architecture, including the classical devices
/// and communication channels present.
void initialize(const config::TargetConfig &config);

/// @brief Allocate data of the given number of size bytes on the
/// controller.
device_ptr malloc(std::size_t size);

template <typename T>
device_ptr malloc() {
  return malloc(sizeof(T));
}
/// @brief Allocate data of the given number of size bytes on the
/// user-specified classical device. Return a device_ptr.
device_ptr malloc(std::size_t size, std::size_t deviceId);

/// @brief Allocate and set the data with given value. Return a device_ptr.
template <typename T>
device_ptr malloc_set(T t, std::size_t device = -1) {
  auto ret = malloc(sizeof(T), device);
  memcpy(ret, &t);
  return ret;
}

/// @brief Free the memory held by the given device_ptr.
void free(device_ptr &d);

/// @brief Copy the given src data into the QPU device data element.
void memcpy(device_ptr &dest, const void *src);

/// @brief Copy the data on QPU device to the given host pointer dest
void memcpy(void *dest, device_ptr &src);

/// @brief Copy the data from the given device_ptr to a host-side value.
template <typename T>
T memcpy(device_ptr &src) {
  T t;
  memcpy(static_cast<void *>(&t), src);
  return t;
}

/// @brief Run any target-specific Quake compilation passes.
/// Returns a handle to the remotely JIT-ed code
handle load_kernel(const std::string &quake);

/// @brief Launch the kernel remotely held at the given handle, with
/// the given runtime arguments.
launch_result launch_kernel(handle kernelHandle, device_ptr args);

/// @brief shutdown the driver API. This should
/// kick of the disconnection of all channels.
void shutdown();

} // namespace driver

} // namespace cudaq

#include "device.h"
