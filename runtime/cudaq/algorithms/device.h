/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <functional>
#include <type_traits>
#include <utility>

#define __qpu_device__ __attribute__((annotate("quantum_device")))

namespace cudaq {

template <typename Result, typename DeviceCode, typename... Args>
  requires std::is_invocable_r_v<Result, DeviceCode, Args...>
Result device_call(DeviceCode &&deviceCode, Args &&...args) {
  return std::invoke(std::move(deviceCode), std::forward<Args>(args)...);
}

template <typename DeviceCode, typename... Args>
void device_call(DeviceCode &&deviceCode, Args &&...args) {
  return std::invoke(std::move(deviceCode), std::forward<Args>(args)...);
}

} // namespace cudaq
