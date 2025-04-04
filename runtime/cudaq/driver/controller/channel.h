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

class data_marshaller {
public:
  virtual void connect(std::size_t assignedID,
                       const config::TargetConfig &config) = 0;
  virtual device_ptr malloc(std::size_t size, std::size_t devId) = 0;
  virtual void free(device_ptr &d) = 0;

  virtual void memcpy(device_ptr &dest, const void *src) = 0;
  virtual void memcpy(void *dest, device_ptr &src) = 0;
};

class device_channel : public data_marshaller, public extension_point<device_channel> {
public:
  device_channel() = default;
  virtual ~device_channel() = default;
  virtual void load_callback(const std::string &funcName,
                             const std::string &unmarshallerCode) {}
  virtual launch_result launch_callback(const std::string &funcName,
                                        device_ptr &argsHandle) {
    throw std::runtime_error("launch callback not supported on this device_channel.");
    return {};
  }
};

class controller_channel : public data_marshaller,
                           public extension_point<controller_channel> {
public:
  controller_channel() = default;
  virtual ~controller_channel() = default;

  virtual handle load_kernel(const std::string &quake) const = 0;
  virtual launch_result launch_kernel(handle kernelHandle,
                                      device_ptr &argsHandle) const = 0;
};

} // namespace cudaq::driver
