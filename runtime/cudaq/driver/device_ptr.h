/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include <cstddef>
#include <string> 

namespace cudaq::driver {

// Handle to a remote resource
using handle = std::size_t;
using error_code = std::size_t;

struct device_ptr {
  // The pointer to the data
  void *data = nullptr;
  // The size in bytes of the data
  std::size_t size;
  // The device ID the data resides on
  std::size_t deviceId = -1;
};

struct launch_result {
  device_ptr result;
  error_code error;
  std::string msg;
};

} // namespace cudaq::driver
