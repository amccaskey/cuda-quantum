/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cudaq/utils/extension_point.h"

#include <string>
#include <vector>

namespace cudaq::driver {

class data {
  void *data = nullptr;
  std::size_t size;
  std::size_t deviceId = -1;
};

class channel : public extension_point<channel> {
public:
  virtual void connect(std::size_t assignedID) const = 0;

  virtual data malloc(std::size_t size) const = 0;
  virtual void free(data &d) const = 0;
  virtual void free(std::size_t argsHandle) const = 0;

  virtual void memcpy(data &arg, const void *src, std::size_t size) const = 0;
  // memcpy a logical grouping of data, return a handle on that (remote) data
  virtual std::size_t memcpy(std::vector<data> &args,
                             std::vector<const void *> srcs,
                             std::vector<std::size_t> size) const = 0;

  virtual std::size_t launch_callback(const std::string &funcName,
                                      std::size_t argsHandle) const = 0;

  virtual std::size_t upload_compiled(const std::string &quake) const = 0;
};

} // namespace cudaq::driver
