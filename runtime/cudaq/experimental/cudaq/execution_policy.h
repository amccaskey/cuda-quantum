/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <concepts>
#include <type_traits>

namespace cudaq {

// ============================================================================
// Execution Policy Concept
// ============================================================================

/// Concept defining what constitutes an execution policy
///
/// An execution policy must define either:
/// - A `result_type` type alias, or
/// - A templated `result_type<T>` type alias, or
/// - An `execution_policy_tag` type alias
///
/// Execution policies encapsulate the intent of how a quantum kernel should
/// be executed and what type of result should be returned.
namespace detail {

template <typename T>
concept ExecutionPolicy = requires {
  typename T::result_type;
} || requires {
  // Allow result_type as template, e.g. for std::vector<Res>
  typename T::template result_type<int>;
} || requires {
  typename T::execution_policy_tag;
};

} // namespace detail

} // namespace cudaq

