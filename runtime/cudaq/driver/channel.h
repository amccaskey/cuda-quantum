/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "common/Logger.h"
#include "cudaq/utils/extension_point.h"

#include <string>
#include <vector>

namespace cudaq::config {
class TargetConfig;
}

namespace cudaq::driver {

enum class memcpy_kind { host_to_driver, driver_to_host };

// The communication channel from host to QPU control is special,
// give it a unique ID none of the others will get
static constexpr std::size_t host_qpu_channel_id =
    std::numeric_limits<std::size_t>::max();

struct device_ptr {
  // The pointer to the data
  void *data = nullptr;
  // The size in bytes of the data
  std::size_t size;
  // The device ID the data resides on
  std::size_t deviceId = -1;
};

// Handle to a remote resource
using handle = std::size_t;
using error_code = std::size_t;

class channel : public extension_point<channel> {
public:
  channel() = default;
  virtual ~channel() = default;

  virtual void connect(std::size_t assignedID,
                       const config::TargetConfig &config) = 0;

  virtual device_ptr malloc(std::size_t size, std::size_t devId) = 0;
  virtual void free(device_ptr &d) = 0;
  virtual void free(std::size_t argsHandle) = 0;

  virtual void memcpy(device_ptr &dest, const void *src) = 0;
  virtual void memcpy(void *dest, device_ptr &src) = 0;

  // memcpy a logical grouping of data, return a handle on that (remote) data
  virtual handle memcpy(std::vector<device_ptr> &args,
                        std::vector<const void *> srcs) = 0;

  virtual error_code launch_callback(const std::string &funcName,
                                     handle argsHandle) const = 0;

  // Register the given quake code on the other side of the channel,
  // return a handle to that kernel.
  virtual handle register_compiled(const std::string &quake) const = 0;

  virtual error_code launch_kernel(handle kernelHandle,
                                   handle argsHandle) const = 0;
};

} // namespace cudaq::driver
