/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Support/TargetConfig.h"
#include "cudaq/driver/channel.h"

INSTANTIATE_REGISTRY_NO_ARGS(cudaq::driver::channel)

namespace cudaq::driver {

/// @brief The
class shared_memory : public channel {
public:
  using channel::channel;

  void connect(std::size_t assignedID,
               const config::TargetConfig &config) override {
    cudaq::info("shared_memory channel connected.");
  }

  device_ptr malloc(std::size_t size, std::size_t devId) override { return {}; }

  void free(device_ptr &d) override {}
  void free(std::size_t argsHandle) override {}

  void memcpy(device_ptr &arg, const void *src) override {}
  void memcpy(void *dst, device_ptr &src) override {}
  // memcpy a logical grouping of data, return a handle on that (remote) data
  std::size_t memcpy(std::vector<device_ptr> &args,
                     std::vector<const void *> srcs) override {
    return 0;
  }

  error_code launch_callback(const std::string &funcName,
                             std::size_t argsHandle) const override {
    return 0;
  }

  std::size_t register_compiled(const std::string &quake) const override {
    return 0;
  }

  error_code launch_kernel(std::size_t kernelHandle,
                           std::size_t argsHandle) const override {
    return 0;
  }

  CUDAQ_EXTENSION_CREATOR_FUNCTION(channel, shared_memory);
};

CUDAQ_REGISTER_EXTENSION_TYPE(shared_memory)

} // namespace cudaq::driver
