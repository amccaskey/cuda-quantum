/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once
#include "device_ptr.h"

#include "common/ThunkInterface.h"
#include "cudaq/utils/extension_point.h"

namespace cudaq {
namespace config {
class TargetConfig;
}
namespace driver {

class control_dispatcher : public extension_point<control_dispatcher> {
public:
  /// @brief Connect to the process / memory space where
  /// the data should live.
  virtual void connect(const config::TargetConfig &config) = 0;
  virtual void disconnect() = 0;

  /// @brief Allocate data of given size on given device and
  /// return the device_ptr handle.
  virtual device_ptr malloc(std::size_t size, std::size_t devId) = 0;

  /// @brief Free the data at the given device_ptr.
  virtual void free(device_ptr &d) = 0;

  /// @brief Copy data to the given device_ptr.
  virtual void send(device_ptr &dest, const void *src) = 0;

  /// @brief Copy data from the given device_ptr.
  virtual void recv(void *dest, device_ptr &src) = 0;

  /// @brief Load the provided quantum kernel (target-specific JIT compilation)
  virtual handle load_kernel(const std::string &quake) const = 0;

  /// @brief Launch the quantum kernel with given thunk args.
  virtual launch_result launch_kernel(handle kernelHandle,
                                      device_ptr &argsHandle) const = 0;
};

class channel : public extension_point<channel> {
protected:
  /// @brief The channel device ID.
  std::size_t device_id = 0;

  /// @brief Locations to libraries containing symbols that 
  /// are required for classical callbacks. 
  std::vector<std::string> symbol_locations;

public:
  channel() = default;
  virtual ~channel() = default;

  /// @brief Connect to the process / memory space where
  /// the data should live.
  virtual void connect(std::size_t assignedID,
                       const config::TargetConfig &config) = 0;
  virtual void disconnect() {}

  virtual bool runs_on_separate_process() { return false; }

  /// @brief Allocate data of given size on given device and
  /// return the device_ptr handle.
  virtual device_ptr malloc(std::size_t size) = 0;

  /// @brief Free the data at the given device_ptr.
  virtual void free(device_ptr &d) = 0;

  /// @brief Copy data to the given device_ptr.
  virtual void send(device_ptr &dest, const void *src) = 0;

  /// @brief Copy data from the given device_ptr.
  virtual void recv(void *dest, device_ptr &src) = 0;

  /// @brief Load the callback of given name with the given MLIR FuncOp code.
  virtual void load_callback(const std::string &funcName,
                             const std::string &unmarshallerCode) = 0;

  virtual void load_callback(
      const std::string &funcName,
      KernelThunkResultType (*shmemUnmarshallerFunc)(void *, bool)) = 0;

  /// @brief Launch the callback with given thunk arguments.
  virtual launch_result launch_callback(const std::string &funcName,
                                        const device_ptr &argsHandle) = 0;
};
} // namespace driver
} // namespace cudaq
