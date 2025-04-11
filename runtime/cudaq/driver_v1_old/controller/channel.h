/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "common/Logger.h"

#include "cudaq/driver/device_ptr.h"
#include "cudaq/utils/extension_point.h"

#include <string>
#include <vector>

namespace cudaq::config {
class TargetConfig;
}

namespace cudaq::driver {

// The communication channel from host to QPU control is special,
// give it a unique ID none of the others will get
static constexpr std::size_t host_qpu_channel_id =
    std::numeric_limits<std::size_t>::max();

/// @brief The memory_manager exposes a simple interface for
/// data movement across devices (device_ptrs). Implementations of this provide
/// concrete mechanisms for allocating and deallocating data,
/// and copying values to and from.
class memory_manager {
public:
  /// @brief Connect to the process / memory space where
  /// the data should live.
  virtual void connect(std::size_t assignedID,
                       const config::TargetConfig &config) = 0;

  /// @brief Allocate data of given size on given device and
  /// return the device_ptr handle.
  virtual device_ptr malloc(std::size_t size, std::size_t devId) = 0;

  /// @brief Free the data at the given device_ptr.
  virtual void free(device_ptr &d) = 0;

  /// @brief Copy data to the given device_ptr.
  virtual void memcpy(device_ptr &dest, const void *src) = 0;

  /// @brief Copy data from the given device_ptr.
  virtual void memcpy(void *dest, device_ptr &src) = 0;
};

/// Next, we differentiate between device channels and the
/// host-to-controller channel. We do so because the latter is
/// more specialized for loading and launching quantum kernels
/// over the channel. We do not need to launch quantum kernels
/// over device channels. Moreover, device channels need to
/// convey some mechanism for loading and launching classical
/// callbacks, which is not something germane to the controller.

/// @brief The device_channel implements the data marshaling
/// API to provide an abstraction for data movement across
/// concrete communication channels from the QPU controller.
/// device_channels are also responsible for one-time load / JIT
/// of classical callbacks and invoking those callbacks.
class device_channel : public memory_manager,
                       public extension_point<device_channel> {
public:
  device_channel() = default;
  virtual ~device_channel() = default;

  /// @brief Load the callback of given name with the given MLIR FuncOp code.
  virtual void load_callback(const std::string &funcName,
                             const std::string &unmarshallerCode) = 0;

  /// @brief Launch the callback with given thunk arguments.
  virtual launch_result launch_callback(const std::string &funcName,
                                        device_ptr &argsHandle) = 0;
};

/// @brief The controller_channel implements the data marshaling API
/// to model data movement from host to QPU control. Moreover, it
/// exposes an API for loading and launching CUDA-Q quantum kernels, and
/// retrieving available callback function names.
class controller_channel : public memory_manager,
                           public extension_point<controller_channel> {
public:
  controller_channel() = default;
  virtual ~controller_channel() = default;

  /// @brief Return the callback function names that the given kernel may
  /// invoke.
  virtual std::vector<std::string> get_callbacks(handle kernelHandle) = 0;

  /// @brief Load the provided quantum kernel (target-specific JIT compilation)
  virtual handle load_kernel(const std::string &quake) const = 0;

  /// @brief Launch the quantum kernel with given thunk args.
  virtual launch_result launch_kernel(handle kernelHandle,
                                      device_ptr &argsHandle) const = 0;
};

} // namespace cudaq::driver
