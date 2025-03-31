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
INSTANTIATE_REGISTRY_NO_ARGS(cudaq::driver::controller_channel)

namespace cudaq::driver {

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

  CUDAQ_EXTENSION_CREATOR_FUNCTION(channel, shared_memory);
};

CUDAQ_REGISTER_EXTENSION_TYPE(shared_memory)

} // namespace cudaq::driver
