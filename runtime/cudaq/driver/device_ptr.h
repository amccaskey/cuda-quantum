/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace cudaq::driver {

// Handle to a remote resource
using handle = std::size_t;
using error_code = std::size_t;

struct device_ptr {
  // The pointer to the data
  std::size_t handle = -1;
  // The size in bytes of the data
  std::size_t size;
  // The device ID the data resides on
  std::size_t deviceId = -1;

  template <typename T>
  operator T *() {
    // dummy function
    return nullptr;
  }
};

struct launch_result {
  std::vector<char> data;
  std::optional<std::string> error;
};

} // namespace cudaq::driver

namespace cudaq {
using device_ptr = driver::device_ptr;
}
