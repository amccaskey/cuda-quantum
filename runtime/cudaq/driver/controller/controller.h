/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cudaq/driver/controller/channel.h"
#include "cudaq/driver/controller/quake_compiler.h"
#include "cudaq/utils/extension_point.h"
#include <map>

/// The Controller Interface provides an extension point
/// for QPU device side control. By assumption, it sits in
/// a separate process / memory space from host-side code.
/// The controller receives requests from the host-side driver
/// to allocate / deallocate memory, and to load and launch
/// quantum kernels. The Controller has knowledge of the available
/// classical device communication channels and can perform automatic
/// data marshalling across devices.
///
/// The design here provides an extension point class that is
/// meant to be subtyped for the specific host-control communcation
/// type, e.g. rpc ethernet, pcie, etc.

namespace cudaq::driver {

/// @brief The controller provides an interface for concrete
/// QPU control (concrete here implies the communication protocol with
/// the host system). Controllers implement a control-side data marshalling
/// API (returns handles instead of device_ptrs). Controllers keep
/// track of available classical devices and how to communicate with them
/// via channels. Controllers are able to load (compile) CUDA-Q kernels and
/// perform analysis for understanding what device side callbacks are necessary.
/// Controllers ultimately launch quantum kernels which run on abstract
/// targets.
class controller : public extension_point<controller> {
protected:
  /// @brief Communication channels for available QPU-side classical devices
  std::vector<std::unique_ptr<device_channel>> communication_channels;

  /// @brief Abstract compiler for mapping Quake code to executable object code.
  std::unique_ptr<quake_compiler> compiler;

  /// @brief Map handles to concrete local data.
  std::map<intptr_t, void *> memory_pool;
  std::map<intptr_t, device_ptr> allocated_device_ptrs;

public:
  /// @brief Initialize this controller with command line input
  virtual void initialize(int argc, char **argv) = 0;

  /// @brief Return true if this controller server should exit.
  virtual bool should_stop() = 0;

  /// @brief Connect host-side clients to this controller server
  virtual void connect(const std::string &cfg);

  /// @brief Allocate memory on devId. Return a unique handle
  virtual handle malloc(std::size_t size, std::size_t devId);

  /// @brief Free allocated memory
  virtual void free(handle handle);

  /// @brief Copy host-side data to the device_ptr at the given handle
  virtual void memcpy_to(handle handle, std::vector<char> &data,
                         std::size_t size);

  /// @brief Copy controller-side data at the given handle back to the
  /// host-side.
  virtual std::vector<char> memcpy_from(handle handle, std::size_t size);

  /// @brief Launch the callback function of given name, with given argument
  /// handle, on specified device.
  virtual launch_result launch_callback(std::size_t deviceId,
                                        const std::string &funcName,
                                        handle argsHandle);

  /// @brief For the kernel at given handle, return the names of all
  /// callback functions it may invoke.
  virtual std::vector<std::string> get_callbacks(handle hdl);

  /// @brief Take host-side shared library paths and distribute them to
  /// classical devices over the communication channels.
  virtual void
  distribute_symbol_locations(const std::vector<std::string> &locs);

  /// @brief Load the kernel into controller memory and perform target-specific
  /// compilation.
  virtual handle load_kernel(const std::string &quake);

  /// @brief Launch and return the result data. This function will
  /// load and compile any required callbacks across the channels to
  /// the classical devices before executing the kernel.
  virtual std::vector<char> launch_kernel(handle kernelHandle,
                                          std::size_t argsHandle);
};

/// @brief Initialize this controller service (called from the controller
/// main()).
void initialize(const std::string &controllerType, int argc, char **argv);

/// @brief Return true if it is time to stop.
bool should_stop();

} // namespace cudaq::driver

#if CUDAQ_USE_STD20
// This code to create automatic conversions of device_ptr arguments is only
// available in C++20.

namespace cudaq::details {
// Concepts to test for device_ptr arguments.
template <typename T>
concept DevicePtr =
    std::same_as<std::decay_t<std::remove_cvref_t<T>>, cudaq::device_ptr>;
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
