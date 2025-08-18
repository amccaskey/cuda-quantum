/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <functional>
#include <vector>

namespace cudaq {

// Forward declaration of the kernel concept
namespace details {
// A simplified handle to a kernel.
struct kernel_handle {
  std::function<void()> functor;
};

// Helper to get a kernel_handle from a QPU lambda.
template <typename Kernel>
kernel_handle get_handle(Kernel &&kernel) {
  return kernel_handle{std::forward<Kernel>(kernel)};
}
} // namespace details

/// @class batch
/// @brief A container for a collection of kernels.
///
/// This class provides a way to group multiple quantum kernels into a single
/// unit for batch submission. It does not contain any execution data
/// (arguments, shots, etc.), which allows the same batch to be reused with
/// different parameter sets.
class batch {
private:
  std::vector<details::kernel_handle> kernels;

public:
  /// @brief Construct a batch from a list of kernels.
  /// @tparam Kernels The types of the kernels to add.
  /// @param k The kernels to add to the batch.
  template <typename... Kernels>
  batch(Kernels &&...k) : kernels{details::get_handle(k)...} {}

  /// @brief Get the number of kernels in the batch.
  size_t size() const { return kernels.size(); }

  /// @brief Get the internal kernel handles.
  const std::vector<details::kernel_handle> &get_kernels() const {
    return kernels;
  }
};

} // namespace cudaq
