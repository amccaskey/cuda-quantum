/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/Logger.h"
#include "common/RuntimeMLIR.h"
#include "cudaq/Support/TargetConfig.h"

#include "cudaq/driver/channel.h"
#include "cudaq/driver/target.h"
#include "cudaq/utils/cudaq_utils.h"

namespace cudaq::driver {

using namespace shmem;

extern void setTarget(target *);

/// @brief Shared memory interface implementation for quantum control dispatcher
/// @details Manages communication between host and quantum processing units
/// (QPUs)
///          via shared memory channels. Handles device memory management,
///          kernel compilation/execution, and classical-quantum callback
///          coordination.
class shmem_iface : public control_dispatcher {
private:
  /// @brief Quantum compiler for kernel transformation
  std::unique_ptr<mlir_compiler> compiler;

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
    compiler = mlir_compiler::get("default_qir_compiler");
    compiler->initialize(config);

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
    for (auto &c : communication_channels)
      c->disconnect();
  }

  /// @brief Allocate device memory with automatic host/device routing
  /// @param size Memory size in bytes
  /// @param devId Target device ID (host_qpu_channel_id for host memory)
  /// @return device_ptr handle to allocated memory
  device_ptr malloc(std::size_t size, std::size_t devId) override {
    return (devId == host_qpu_channel_id)
               ? device_ptr{shmem::to_handle(std::malloc(size)), size, devId}
               : communication_channels[devId]->malloc(size);
  }

  /// @brief Release memory resources based on device location
  /// @param d device_ptr handle to memory block
  void free(device_ptr &d) override {
    (d.deviceId == host_qpu_channel_id)
        ? std::free(shmem::to_ptr(d))
        : communication_channels[d.deviceId]->free(d);
  }

  /// @brief Transfer data to device memory with automatic routing
  /// @param dest Destination device pointer
  /// @param src Host memory source pointer
  void send(device_ptr &dest, const void *src) override {
    if (dest.deviceId == host_qpu_channel_id) {
      std::memcpy(shmem::to_ptr(dest), src, dest.size);
      return;
    }

    communication_channels[dest.deviceId]->send(dest, src);
  }

  /// @brief Retrieve data from device memory with automatic routing
  /// @param dest Host memory destination pointer
  /// @param src Source device pointer
  void recv(void *dest, device_ptr &src) override {
    if (src.deviceId == host_qpu_channel_id) {
      std::memcpy(dest, shmem::to_ptr(src), src.size);
      return;
    }

    communication_channels[src.deviceId]->recv(dest, src);
  }

  /// @brief Compile quantum kernel to target-specific representation
  /// @param quake Kernel code in MLIR/Quake format
  /// @return handle to compiled kernel executable
  handle load_kernel(const std::string &quake) const override {
    auto hdl = compiler->load(quake);

    // Do we need the unmarshal functions,
    // in a distributed system, we may need to remove them
    // since the symbols won't be local
    bool needUnmarshallers = true;
    for (auto &channel : communication_channels)
      needUnmarshallers &=
          communication_channels.back()->requires_unmarshaller();

    // Loop over the callbacks and upload
    // the unmarshal code
    auto callbacks = compiler->get_callbacks(hdl);
    for (auto &callback : callbacks) {
      for (auto &channel : communication_channels)
        channel->load_callback(callback.callbackName,
                               callback.unmarshalFuncOpCode);

      // Drop the unmarshal function if we don't need it
      if (!needUnmarshallers)
        compiler->remove_callback(hdl, callback.callbackName);
    }

    // Lower to LLVM
    compiler->compile(hdl);

    // Return the loaded module handle
    return hdl;
  }

  /// @brief Execute quantum kernel with managed resource allocation
  /// @param kernelHandle Compiled kernel handle from load_kernel
  /// @param argsHandle Device pointer to kernel arguments
  /// @return launch_result containing execution results/errors
  launch_result launch_kernel(handle kernelHandle,
                              device_ptr &argsHandle) const override {
    compiler->launch(kernelHandle, shmem::to_ptr(argsHandle));
    return {};
  }

  CUDAQ_EXTENSION_CREATOR_FUNCTION(control_dispatcher, shmem_iface)
};

/// @brief Register shared memory interface as control dispatcher
/// implementation
CUDAQ_REGISTER_EXTENSION_TYPE(shmem_iface)

} // namespace cudaq::driver
