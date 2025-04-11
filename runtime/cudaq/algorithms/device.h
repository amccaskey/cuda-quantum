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


#if 0
// This code to create automatic conversions of device_ptr arguments is only
// available in C++20.

namespace cudaq::details {
// Concepts to test for device_ptr arguments.
template <typename T>
concept DevicePtr =
    std::same_as<std::decay_t<std::remove_cvref_t<T>>, cudaq::driver::device_ptr>;
template <typename T>
concept NotDevicePtr = !DevicePtr<T>;

// Conversion functions.
template <NotDevicePtr T>
T convert(T &&t) {
  return std::forward<T>(t);
}
template <NotDevicePtr T>
T convert(const T &t) {
  return t;
}

template <DevicePtr T>
void *convert(T &&devicePtr) {
  // FIXME: add code to lookup the pointer from the devicePtr.
  return nullptr;
}
template <DevicePtr T>
void *convert(const T &devicePtr) {
  // FIXME: add code to lookup the pointer from the devicePtr.
  return nullptr;
}

template <typename Call, typename... Args>
void gpu_call_dispatcher(Call call, Args... args) {
  auto tup = std::make_tuple(details::convert<Args>(args)...);
  return std::apply(call, tup);
}
} // namespace cudaq::details

#define AUTOGENERATE_ARGUMENT_CONVERSION(FUN, ...)                             \
  cudaq::details::gpu_call_dispatcher(FUN, __VA_ARGS__)

/// Users can autogenerate glue code to call a device callback on a GPU using
/// the above template functions and the `AUTOGENERATE_ARGUMENT_CONVERSION`
/// macro. This can be done as in the following example.

//   // A CUDA function kernel.
//   __global__ void my_gpu_function(int value, void* buffer);
//
//   // Our device_call trampoline function.
//   void my_gpu_trampoline(int value, cudaq::device_ptr adaptor) {
//     AUTOGENERATE_ARGUMENT_CONVERSION(my_gpu_function, value, adaptor);
//   }
//
//   // In our CUDA-Q kernel, we would then use a device_call such as
//   __qpu__ void quantum_kernel() {
//     ...
//     cudaq::device_call<numBlocks, threadsPerBlock>(
//                        my_gpu_trampoline, 12, myDevPtr);
//     ...
//   }

#endif


} // namespace cudaq
