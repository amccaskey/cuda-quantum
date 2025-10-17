/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include <concepts>

namespace cudaq::traits {
class remote {};
} // namespace cudaq::traits

namespace cudaq {
#ifndef CUDAQ_NO_STD20

template <typename T>
concept RemoteQPU = requires {
  // Type requirement: T must derive from sample_trait<T> (CRTP pattern)
  requires std::derived_from<std::decay_t<T>, traits::remote>;
};
#endif
template <typename T>
bool is_remote(T &&t) {
  return std::is_base_of_v<traits::remote, std::decay_t<T>>;
}

} // namespace cudaq
