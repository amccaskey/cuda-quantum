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

/// @brief Structure representing CUDA launch parameters for kernel execution.
///
/// Contains configuration for block and grid dimensions used in CUDA kernels.
struct cuda_launch_parameters {
  /// @brief Array holding the dimensions of each thread block (x, y, z).
  const std::size_t blockDim[3];

  /// @brief Array holding the dimensions of the grid (x, y, z).
  const std::size_t gridDim[3];

  /// @brief Default constructor initializing all dimensions to zero.
  ///
  /// Sets both blockDim and gridDim arrays to {0, 0, 0}.
  cuda_launch_parameters() : blockDim{0, 0, 0}, gridDim{0, 0, 0} {}

  /// @brief Constructor initializing block and grid dimensions with specified
  /// values.
  ///
  /// @param block The size of the block in the x-dimension; y and z are
  /// defaulted to 1.
  /// @param grid The size of the grid in the x-dimension; y and z are defaulted
  /// to 1.
  cuda_launch_parameters(std::size_t block, std::size_t grid)
      : blockDim{block, 1, 1}, gridDim{grid, 1, 1} {}

  /// @brief Checks if the launch parameters are valid.
  ///
  /// @return true if the first element of blockDim is zero, indicating invalid
  /// parameters.
  bool has_parameters() const { return blockDim[0] == 0; }
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
