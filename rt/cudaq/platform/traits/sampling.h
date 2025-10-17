/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include <concepts>
#include <type_traits>
#include <functional> 

#include "cudaq/policies/sample/sample_result.h"

namespace cudaq {

namespace traits {
template <typename Derived>
class sample_trait {
public:
  // need num shots, kernel name (from this can get quake code),
  // unified wrapped functor for kernel (captured args).
  sample_result sample(std::size_t num_shots, const std::string &kernel_name,
                       const std::function<void()> &wrapped_kernel) {
    return static_cast<Derived>(this).sample(num_shots, kernel_name,
                                             wrapped_kernel);
  }
};
} // namespace traits

#ifndef CUDAQ_NO_STD20

template <typename T>
concept SamplingQPU = requires {
  // Type requirement: T must derive from sample_trait<T> (CRTP pattern)
  requires std::derived_from<std::decay_t<T>,
                             traits::sample_trait<std::decay_t<T>>>;
};

#endif

} // namespace cudaq
