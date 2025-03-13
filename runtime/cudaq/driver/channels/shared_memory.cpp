/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/driver/channel.h"

namespace cudaq::driver {

/// @brief The
class shared_memory : public channel {
public:
  void connect(std::size_t assignedID) const override {}

  data malloc(std::size_t size) const override {}
  void free(data &d) const override {}

  void memcpy(data &arg, const void *src, std::size_t size) const override {}

  // memcpy a logical grouping of data, return a handle on that (remote) data
  std::size_t memcpy(std::vector<data> &args, std::vector<const void *> srcs,
                     std::vector<std::size_t> size) const override {}

  CUDAQ_EXTENSION_CREATOR_FUNCTION(channel, shared_memory);
};

CUDAQ_REGISTER_TYPE(shared_memory)

} // namespace cudaq::driver

INSTANTIATE_REGISTRY_NO_ARGS(cudaq::driver::channel)