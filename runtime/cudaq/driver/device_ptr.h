/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace cudaq::driver {

/// @brief Handle to a remote resource
using handle = std::size_t;

/// @brief Error code type for quantum operations
using error_code = std::size_t;

/// @brief Represents a pointer to memory allocated on a quantum processing
/// unit (controller or classical device)
/// @details Encapsulates device memory management details including location
/// and size
struct device_ptr {
  /// @brief Opaque handle to device memory block
  std::size_t handle = std::numeric_limits<std::size_t>::max();

  /// @brief Size of allocated memory in bytes
  std::size_t size;

  /// @brief Physical device identifier
  std::size_t deviceId = std::numeric_limits<std::size_t>::max();

  /// @brief Type conversion operator for device memory access
  /// @tparam T Target data type for pointer conversion
  /// @return nullptr (dummy implementation for template compatibility)
  /// @note Current implementation returns null - actual device memory access
  ///       requires platform-specific handling
  template <typename T>
  operator T *() {
    return nullptr;
  }
};

/// @brief Result container for quantum kernel execution
/// @details Stores both successful computation results and error information
struct launch_result {
  /// @brief Raw byte stream containing execution results
  std::vector<char> data;

  /// @brief Optional error message container
  std::optional<std::string> error;
};

} // namespace cudaq::driver

// For language exposure
namespace cudaq {
/// @brief Alias for quantum device memory pointer handle
using device_ptr = driver::device_ptr;
} // namespace cudaq
