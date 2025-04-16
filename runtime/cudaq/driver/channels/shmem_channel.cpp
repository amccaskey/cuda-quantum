/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/Logger.h"
#include "common/RuntimeMLIR.h"
#include "common/ThunkInterface.h"

#include "cudaq/Support/TargetConfig.h"
#include "cudaq/driver/channel.h"

#include "cudaq/utils/process.h"

namespace cudaq::driver {
using namespace shmem;

/// @brief Shared memory communication channel implementation.
/// @details Models a shared memory connection where the device shares a memory
/// space with the controller.
class shmem_channel : public channel {
protected:
  /// @brief Map of callback function names to their unmarshaller function
  /// pointers.
  std::map<std::string, KernelThunkResultType (*)(void *, bool)> unmarshallers;

  /// @brief Compiler instance for processing Quake code.
  std::unique_ptr<quake_compiler> compiler;

  /// @brief Map of compiled unmarshaller functions for callbacks.
  std::map<std::string, std::size_t> compiledUnmarshallers;

public:
  /// @brief Connect to the shared memory space where data resides.
  /// @param assignedID Device ID for this channel.
  /// @param config Target configuration details.
  void connect(std::size_t assignedID,
               const config::TargetConfig &config) override {
    device_id = assignedID;
    compiler = quake_compiler::get("default_compiler");
    compiler->initialize(config);
    symbol_locations =
        config.Devices[assignedID].Config.ExposedLibraries.value_or(
            std::vector<std::string>{});
  }

  /// @brief Return the raw pointer corresponding to the
  /// provided device_ptr.
  void *get_raw_pointer(device_ptr &devPtr) override {
    return reinterpret_cast<void *>(devPtr.handle);
  }

  /// @brief Allocate memory on the device and return a device pointer handle.
  /// @param size Size of memory to allocate in bytes.
  /// @return Device pointer handle representing allocated memory.
  device_ptr malloc(std::size_t size) override {
    return {to_handle(std::malloc(size)), size, device_id};
  }

  /// @brief Free allocated memory on the device.
  /// @param d Device pointer handle to free.
  void free(device_ptr &d) override { std::free(to_ptr(d)); }

  /// @brief Copy data from host to device memory.
  /// @param dest Destination device pointer on the device.
  /// @param src Source data in host memory.
  void send(device_ptr &dest, const void *src) override {
    std::memcpy(to_ptr(dest), src, dest.size);
  }

  /// @brief Copy data from device to host memory.
  /// @param dest Destination buffer in host memory.
  /// @param src Source device pointer on the device.
  void recv(void *dest, device_ptr &src) override {
    std::memcpy(dest, to_ptr(src), src.size);
  }

  /// @brief Load a callback function using MLIR FuncOp code representation.
  /// @param funcName Name of the callback function to load.
  /// @param unmarshallerCode MLIR FuncOp code for unmarshalling arguments.
  void load_callback(const std::string &funcName,
                     const std::string &unmarshallerCode) override {
    compiledUnmarshallers.insert(
        {funcName,
         compiler->compile_unmarshaler(unmarshallerCode, symbol_locations)});
  }

  /// @brief Load a callback function using an unmarshaller function pointer.
  /// @param funcName Name of the callback function to load.
  /// @param shmemUnmarshallerFunc Function pointer for shared memory
  /// unmarshalling logic.
  void load_callback(
      const std::string &funcName,
      KernelThunkResultType (*shmemUnmarshallerFunc)(void *, bool)) override {
    unmarshallers.insert({funcName, shmemUnmarshallerFunc});
  }

  /// @brief Launch a callback function with specified arguments on this
  /// channel.
  /// @param funcName Name of the callback function to launch.
  /// @param args Device pointer containing arguments for the callback
  /// execution.
  /// @return Result of callback execution (empty result if successful).
  launch_result launch_callback(
      const std::string &funcName, const device_ptr &args,
      cuda_launch_parameters params = cuda_launch_parameters()) override {
    auto iter = unmarshallers.find(funcName);
    if (iter != unmarshallers.end()) {
      unmarshallers.at(funcName)(to_ptr(args), false);
      return {};
    }

    auto hdl = compiledUnmarshallers.at(funcName);
    compiler->launch(hdl, to_ptr(args));
    return {};
  }

  CUDAQ_EXTENSION_CREATOR_FUNCTION(channel, shmem_channel);
};
} // namespace cudaq::driver
CUDAQ_REGISTER_EXTENSION_TYPE(cudaq::driver::shmem_channel)
