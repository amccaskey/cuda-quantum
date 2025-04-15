/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "driver.h"
#include "target.h"

#include "common/Logger.h"
#include "common/RuntimeMLIR.h"
#include "cudaq/Support/TargetConfig.h"

#include <dlfcn.h>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <stdarg.h>
#include <string.h>
#include <vector>

/// @file
/// @brief Implementation of CUDA Quantum driver functionality for managing
/// communication between host and QPU.
/// @details This file implements the driver API, which facilitates kernel
/// compilation, memory management,
///          and execution workflows for quantum devices. It supports shared
///          memory and separate process channels.

INSTANTIATE_REGISTRY_NO_ARGS(cudaq::driver::channel)
INSTANTIATE_REGISTRY_NO_ARGS(cudaq::driver::control_dispatcher)

namespace cudaq::driver {

/// @brief Special identifier for host-QPU communication channel.
static constexpr std::size_t host_qpu_channel_id =
    std::numeric_limits<std::size_t>::max();

extern void setTarget(target *);

namespace details {
/// @brief Collection of communication channels between host and devices.
std::vector<std::unique_ptr<channel>> communication_channels;

/// @brief Convert a shared memory device pointer handle to its raw pointer.
/// @param d Device pointer handle.
/// @return Raw pointer representation of the device memory.
void *to_ptr(const device_ptr &d) { return reinterpret_cast<void *>(d.handle); }

/// @brief Convert a raw pointer to its handle representation.
/// @param ptr Raw pointer.
/// @return Handle representation of the pointer.
std::size_t to_handle(void *ptr) { return reinterpret_cast<uintptr_t>(ptr); }

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
  launch_result launch_callback(const std::string &funcName,
                                const device_ptr &args) override {
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

CUDAQ_REGISTER_EXTENSION_TYPE(shmem_channel)
/// @brief Shared memory interface implementation for quantum control dispatcher
/// @details Manages communication between host and quantum processing units
/// (QPUs)
///          via shared memory channels. Handles device memory management,
///          kernel compilation/execution, and classical-quantum callback
///          coordination.
class shmem_iface : public control_dispatcher {
private:
  /// @brief Quantum compiler for kernel transformation
  std::unique_ptr<quake_compiler> compiler;

  /// @brief Target QPU execution backend
  std::unique_ptr<target> backend;

  /// @brief Library paths for symbol resolution
  std::vector<std::string> symbolLocations;

public:
  /// @brief Initialize shared memory interface with target configuration
  /// @param config Target configuration containing device parameters and
  /// settings
  /// @details Configures communication channels for each device, initializes
  ///          quantum compiler with callback support, and sets up target
  ///          backend.
  void connect(const config::TargetConfig &config) override {
    // Configure communication channels for each target device
    for (std::size_t id = 0; auto &device : config.Devices) {
      cudaq::info(
          "[shmem_iface] adding classical connected device with name {}.",
          device.Name);
      communication_channels.emplace_back(channel::get(device.Config.Channel));
      communication_channels.back()->connect(id++, config);

      // Collect exposed libraries for symbol resolution
      auto devLibs =
          device.Config.ExposedLibraries.value_or(std::vector<std::string>{});
      symbolLocations.insert(symbolLocations.end(), devLibs.begin(),
                             devLibs.end());
    }

    // Initialize quantum compiler with callback preservation
    compiler = quake_compiler::get("default_compiler");
    compiler->initialize(config, {{"remove_unmarshals", true}});

    // Configure target execution backend
    backend = target::get("default_target");
    backend->initialize(config);
    setTarget(backend.get());

    cudaq::info("Connecting to shared memory host-control interface. "
                "Controller manages {} devices.",
                communication_channels.size());
  }

  /// @brief Terminate all communication channels and clean up resources
  void disconnect() override {
    cudaq::info("Disconnecting quantum communication channels.");
    for (auto &c : details::communication_channels)
      c->disconnect();
  }

  /// @brief Allocate device memory with automatic host/device routing
  /// @param size Memory size in bytes
  /// @param devId Target device ID (host_qpu_channel_id for host memory)
  /// @return device_ptr handle to allocated memory
  device_ptr malloc(std::size_t size, std::size_t devId) override {
    return (devId == host_qpu_channel_id)
               ? device_ptr{to_handle(std::malloc(size)), size, devId}
               : communication_channels[devId]->malloc(size);
  }

  /// @brief Release memory resources based on device location
  /// @param d device_ptr handle to memory block
  void free(device_ptr &d) override {
    (d.deviceId == host_qpu_channel_id)
        ? std::free(to_ptr(d))
        : communication_channels[d.deviceId]->free(d);
  }

  /// @brief Transfer data to device memory with automatic routing
  /// @param dest Destination device pointer
  /// @param src Host memory source pointer
  void send(device_ptr &dest, const void *src) override {
    if (dest.deviceId == host_qpu_channel_id) {
      std::memcpy(to_ptr(dest), src, dest.size);
      return;
    }

    communication_channels[dest.deviceId]->send(dest, src);
  }

  /// @brief Retrieve data from device memory with automatic routing
  /// @param dest Host memory destination pointer
  /// @param src Source device pointer
  void recv(void *dest, device_ptr &src) override {
    if (src.deviceId == host_qpu_channel_id) {
      std::memcpy(dest, to_ptr(src), src.size);
      return;
    }

    communication_channels[src.deviceId]->recv(dest, src);
  }

  /// @brief Compile quantum kernel to target-specific representation
  /// @param quake Kernel code in MLIR/Quake format
  /// @return handle to compiled kernel executable
  handle load_kernel(const std::string &quake) const override {
    return compiler->compile(quake, symbolLocations);
  }

  /// @brief Execute quantum kernel with managed resource allocation
  /// @param kernelHandle Compiled kernel handle from load_kernel
  /// @param argsHandle Device pointer to kernel arguments
  /// @return launch_result containing execution results/errors
  launch_result launch_kernel(handle kernelHandle,
                              device_ptr &argsHandle) const override {
    // Distribute callback unmarshallers to all devices
    auto callbacks = compiler->get_callbacks(kernelHandle);
    for (auto &callback : callbacks)
      for (auto &channel : communication_channels)
        channel->load_callback(callback.callbackName,
                               callback.unmarshalFuncOpCode);

    // Dynamic qubit allocation for adaptive execution profiles

    if (auto maybeNumQubits = compiler->get_required_num_qubits(kernelHandle)) {
      backend->allocate(*maybeNumQubits); // Acquire qubits
      compiler->launch(kernelHandle, to_ptr(argsHandle));
      backend->deallocate(*maybeNumQubits); // Release qubits
      return {};
    }

    compiler->launch(kernelHandle, to_ptr(argsHandle));
    return {};
  }

  CUDAQ_EXTENSION_CREATOR_FUNCTION(control_dispatcher, shmem_iface)
};

/// @brief Register shared memory interface as control dispatcher implementation
CUDAQ_REGISTER_EXTENSION_TYPE(shmem_iface)

// The communication channel from host to QPU control.
// Has to be the driver_channel subtype
static std::unique_ptr<control_dispatcher> host_qpu_channel;

} // namespace details

void initialize(const config::TargetConfig &cfg) {
  // setup the host to qpu control communication channel
  // can swap out the host->control channel
  auto ctrlChannelTy = cfg.HostControlChannel.value_or("shmem_iface");
  cudaq::info("Creating the dispatcher - {}", ctrlChannelTy);
  details::host_qpu_channel = control_dispatcher::get(ctrlChannelTy);
  details::host_qpu_channel->connect(cfg);
}

device_ptr malloc(std::size_t size) {
  // If devId not equal to driver id, then this is a request to
  // the driver to allocate the memory on the correct device
  return details::host_qpu_channel->malloc(size, host_qpu_channel_id);
}

device_ptr malloc(std::size_t size, std::size_t devId) {
  // If devId not equal to driver id, then this is a request to
  // the driver to allocate the memory on the correct device
  return details::host_qpu_channel->malloc(size, devId);
}

void free(device_ptr &d) { return details::host_qpu_channel->free(d); }

// Copy the given src data into the data element.
void memcpy(device_ptr &arg, const void *src) {
  return details::host_qpu_channel->send(arg, src);
}

void memcpy(void *dest, device_ptr &src) {
  details::host_qpu_channel->recv(dest, src);
}

handle load_kernel(const std::string &quake) {
  return details::host_qpu_channel->load_kernel(quake);
}

launch_result launch_kernel(handle kernelHandle, device_ptr args) {
  return details::host_qpu_channel->launch_kernel(kernelHandle, args);
}

void shutdown() { details::host_qpu_channel->disconnect(); }

} // namespace cudaq::driver

extern "C" {

void *__nvqpp__device_extract_device_ptr(cudaq::device_ptr *devPtr) {
  using namespace cudaq::driver;

  cudaq::info("Extracting the device pointer for {}, {}", devPtr->handle,
              devPtr->deviceId);
  // Here we know we are in shared memory only
  if (devPtr->deviceId == host_qpu_channel_id)
    return reinterpret_cast<void *>(devPtr->handle);

  auto &channel = details::communication_channels[devPtr->deviceId];
  // Should this only be valid for CUDA and Shmem channels?
  if (channel->runs_on_separate_process())
    throw std::runtime_error("error extracting callback pointer argument - "
                             "channel is separate process.");

  return channel->get_raw_pointer(*devPtr);
}
cudaq::KernelThunkResultType
__nvqpp__device_callback_run(std::uint64_t deviceId, const char *funcName,
                             void *unmarshalFunc, void *argsBuffer,
                             std::uint64_t argsBufferSize,
                             std::uint64_t returnOffset) {
  using namespace cudaq::driver;
  cudaq::info("classical callback with shmem (host-side) func={} args_size={}",
              std::string(funcName), argsBufferSize);

  // Get the correct channel
  auto &channel = details::communication_channels[deviceId];

  // Could be that our host+controller+channel are all on shared memory
  // in which case we can just run the unmarshal function pointer
  // If not, then we need to send the data and call the JIT
  // compiled unmarshal function
  if (channel->runs_on_separate_process()) {
    // Separate process channel, allocate the unmarshal args
    auto argPtr = channel->malloc(argsBufferSize);
    // Send the args
    channel->send(argPtr, argsBuffer);
    // Launch the callback
    channel->launch_callback(funcName, argPtr);
    // Update the local args pointer
    channel->recv(argsBuffer, argPtr);
    // Free the data
    channel->free(argPtr);
    return {};
  }

  // This is a local shared memory callback, use the unmarshal function pointer.
  auto *castedFunc =
      reinterpret_cast<cudaq::KernelThunkResultType (*)(void *, bool)>(
          unmarshalFunc);

  // Load the callback (this is simple, just provide the channel with the
  // function pointer)
  channel->load_callback(funcName, castedFunc);

  // Launch the callback, result data stored to argsBuffer.
  channel->launch_callback(funcName,
                           {cudaq::driver::details::to_handle(argsBuffer),
                            argsBufferSize, deviceId});

  return {};
}
}
