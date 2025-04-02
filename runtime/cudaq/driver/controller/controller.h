/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cudaq/driver/channel.h"
#include "cudaq/driver/controller/quake_compiler.h"
#include "cudaq/utils/extension_point.h"

#include <map>

/// The Controller Interface provides an extension point
/// for QPU device side control. By assumption, it sits in
/// a separate process / memory space from host-side code.
/// The controller receives requests from the host-side driver
/// to allocate / deallocate memory, and to load and launch
/// quantum kernels. The Controller has knowledge of the available
/// classical device communication channels and can perform automatic
/// data marshalling across devices.
///
/// The design here provides an extension point class that is
/// meant to be subtyped for the specific host-control communcation
/// type, e.g. rpc, pcie, etc.

namespace cudaq::driver {

class controller : public extension_point<controller> {
protected:
  std::vector<std::unique_ptr<channel>> communication_channels;
  std::unique_ptr<quake_compiler> compiler;
  std::map<intptr_t, device_ptr> memory_pool;

public:
  virtual void initialize(int argc, char **argv) = 0;
  virtual bool should_stop() = 0;
  virtual void connect(const std::string &cfg);

  // Allocate memory on devId. Return a unique handle
  virtual std::size_t malloc(std::size_t size, std::size_t devId);
  // Free allocated memory
  virtual void free(std::size_t handle);

  virtual void memcpy_to(std::size_t handle, std::vector<char> &data,
                         std::size_t size);
  virtual std::vector<char> memcpy_from(std::size_t handle, std::size_t size);

  virtual launch_result launch_callback(std::size_t deviceId,
                                        const std::string &funcName,
                                        std::size_t argsHandle);
  // load the kernel into controller memory,
  // can perform target-specific compilation.
  virtual handle load_kernel(const std::string &quake);

  // launch and return the result data
  virtual std::vector<char> launch_kernel(std::size_t kernelHandle,
                                          std::size_t argsHandle);
};

void initialize(const std::string &controllerType, int argc, char **argv);
bool should_stop();


} // namespace cudaq::driver