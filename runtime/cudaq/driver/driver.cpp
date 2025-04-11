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

INSTANTIATE_REGISTRY_NO_ARGS(cudaq::driver::channel)
INSTANTIATE_REGISTRY_NO_ARGS(cudaq::driver::control_dispatcher)

// Driver API is host-side, but needs to potentially
// communicate actions in a separate process (the controller)

namespace cudaq::driver {
static constexpr std::size_t host_qpu_channel_id =
    std::numeric_limits<std::size_t>::max();

extern void setTarget(target *);

namespace details {
std::vector<std::unique_ptr<channel>> communication_channels;

// Convert a shared memory device_ptr handle to its pointer
void *to_ptr(const device_ptr &d) { return reinterpret_cast<void *>(d.handle); }
// Convert a shared memory pointer to its handle representation
std::size_t to_handle(void *ptr) { return reinterpret_cast<uintptr_t>(ptr); }

/// @brief The shmem_channel provides a device channel
/// implementation that models a shared memory connection,
/// i.e. the device shares a memory space with the controller.
class shmem_channel : public channel {
protected:
  /// @brief Map of callback function names to their provided
  /// unmarshaller function pointers (only valid for shared memory channels)
  std::map<std::string, KernelThunkResultType (*)(void *, bool)> unmarshallers;
  std::unique_ptr<quake_compiler> compiler;
  std::map<std::string, std::size_t> compiledUnmarshallers;

public:
  /// @brief Connect to the process / memory space where
  /// the data should live.
  void connect(std::size_t assignedID,
               const config::TargetConfig &config) override {
    device_id = assignedID;
    compiler = quake_compiler::get("default_compiler");
    compiler->initialize(config);
    symbol_locations =
        config.Devices[assignedID].Config.ExposedLibraries.value_or(
            std::vector<std::string>{});
  }

  /// @brief Allocate data of given size on given device and
  /// return the device_ptr handle.
  device_ptr malloc(std::size_t size) override {
    return {to_handle(std::malloc(size)), size, device_id};
  }

  /// @brief Free the data at the given device_ptr.
  void free(device_ptr &d) override { std::free(to_ptr(d)); }

  /// @brief Copy data to the given device_ptr.
  void send(device_ptr &dest, const void *src) override {
    std::memcpy(to_ptr(dest), src, dest.size);
  }

  /// @brief Copy data from the given device_ptr.
  void recv(void *dest, device_ptr &src) override {
    std::memcpy(dest, to_ptr(src), src.size);
  }

  /// @brief Load the callback of given name with the given MLIR FuncOp code.
  void load_callback(const std::string &funcName,
                     const std::string &unmarshallerCode) override {
    compiledUnmarshallers.insert(
        {funcName,
         compiler->compile_unmarshaler(unmarshallerCode, symbol_locations)});
  }

  void load_callback(
      const std::string &funcName,
      KernelThunkResultType (*shmemUnmarshallerFunc)(void *, bool)) override {
    unmarshallers.insert({funcName, shmemUnmarshallerFunc});
  }

  /// @brief Launch the callback with given thunk arguments.
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

class shmem_iface : public control_dispatcher {
private:
  std::unique_ptr<quake_compiler> compiler;
  std::unique_ptr<target> backend;
  std::vector<std::string> symbolLocations;

public:
  /// @brief Connect to the process / memory space where
  /// the data should live.
  void connect(const config::TargetConfig &config) override {
    // Configure the communication channels
    for (std::size_t id = 0; auto &device : config.Devices) {
      cudaq::info(
          "[shmem_iface] adding classical connected device with name {}.",
          device.Name);
      communication_channels.emplace_back(channel::get(device.Config.Channel));
      communication_channels.back()->connect(id++, config);

      auto devLibs =
          device.Config.ExposedLibraries.value_or(std::vector<std::string>{});
      for (auto &d : devLibs)
        symbolLocations.push_back(d);
    }

    compiler = quake_compiler::get("default_compiler");
    compiler->initialize(config, {{"remove_unmarshals", false}});

    backend = target::get("default_target");
    backend->initialize(config);

    setTarget(backend.get());

    cudaq::info("connecting to shmem host-control interface. controller has "
                "access to {} devices.",
                communication_channels.size());
  }

  void disconnect() override {
    cudaq::info("disconnecting channels.");
    for (auto &c : details::communication_channels)
      c->disconnect();
  }

  /// @brief Allocate data of given size on given device and
  /// return the device_ptr handle.
  device_ptr malloc(std::size_t size, std::size_t devId) override {
    if (devId == host_qpu_channel_id) {
      return {to_handle(std::malloc(size)), size, devId};
    }

    return communication_channels[devId]->malloc(size);
  }

  /// @brief Free the data at the given device_ptr.
  void free(device_ptr &d) override {
    if (d.deviceId == host_qpu_channel_id)
      return std::free(to_ptr(d));

    communication_channels[d.deviceId]->free(d);
  }

  /// @brief Copy data to the given device_ptr.
  void send(device_ptr &dest, const void *src) override {
    if (dest.deviceId == host_qpu_channel_id) {
      std::memcpy(to_ptr(dest), src, dest.size);
      return;
    }

    communication_channels[dest.deviceId]->send(dest, src);
  }

  /// @brief Copy data from the given device_ptr.
  void recv(void *dest, device_ptr &src) override {
    if (src.deviceId == host_qpu_channel_id) {
      std::memcpy(dest, to_ptr(src), src.size);
      return;
    }

    communication_channels[src.deviceId]->recv(dest, src);
  }

  /// @brief Load the provided quantum kernel (target-specific JIT compilation)
  handle load_kernel(const std::string &quake) const override {
    return compiler->compile(quake, symbolLocations);
  }

  /// @brief Launch the quantum kernel with given thunk args.
  launch_result launch_kernel(handle kernelHandle,
                              device_ptr &argsHandle) const override {
    // Make callback code available to devices
    auto callbacks = compiler->get_callbacks(kernelHandle);
    for (auto &callback : callbacks)
      for (auto &channel : communication_channels)
        channel->load_callback(callback.callbackName,
                               callback.unmarshalFuncOpCode);

    // We assume adaptive profile...
    auto maybeNumQubits = compiler->get_required_num_qubits(kernelHandle);
    if (maybeNumQubits)
      backend->allocate(*maybeNumQubits);

    compiler->launch(kernelHandle, to_ptr(argsHandle));

    if (maybeNumQubits)
      backend->deallocate(*maybeNumQubits);

    return {};
  }

  CUDAQ_EXTENSION_CREATOR_FUNCTION(control_dispatcher, shmem_iface)
};

CUDAQ_REGISTER_EXTENSION_TYPE(shmem_iface);

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
