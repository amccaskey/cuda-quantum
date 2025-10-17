/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include <concepts> 
#include <optional> 

namespace cudaq {

namespace details {
// Enhanced concept that handles both fixed and templated result_type
template <typename T>
concept ExecutionPolicy = requires {
  // Check for either a fixed result_type or a templated result_type alias
  typename T::result_type;
} || requires {
  // Check for templated result_type alias
  typename T::template result_type<int>; // Use int as a test type
} || requires {
  // Alternative: check for a policy_tag to identify policies
  typename T::execution_policy_tag;
};

template <typename QuantumKernel, typename... Args>
concept IsValidLaunchFirstArg =
    std::invocable<QuantumKernel, Args...> || ExecutionPolicy<QuantumKernel>;
} // namespace details
} // namespace cudaq
