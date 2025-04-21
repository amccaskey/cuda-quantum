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

#include "rpc/client.h"

#include <dlfcn.h>

namespace cudaq::driver {

/// @brief Ethernet-based device communication channel.
/// @details This class implements a communication channel between the host and
///          quantum processing units (QPUs) using Ethernet-based RPC (Remote
///          Procedure Call). It supports memory allocation, data transfer,
///          callback loading, and kernel execution.
class ethernet_device_channel : public channel {
  /// @brief RPC client for communicating with the remote QPU controller.
  std::unique_ptr<rpc::client> client;

  /// @brief Mapping of callback function names to their handles on the remote
  /// QPU.
  std::map<std::string, std::size_t> funcNamesToHandles;

public:
  using channel::channel;

  /// @brief Connect to the remote QPU controller via Ethernet.
  /// @param assignedID The ID assigned to this device channel.
  /// @param config Configuration details for the target QPU and communication
  /// parameters.
  /// @details Establishes an RPC connection to the controller using IP and port
  /// from the configuration.
  void connect(std::size_t assignedID,
               const config::TargetConfig &config) override {
    device_id = assignedID;
    std::string ip =
        config.Devices[assignedID].Config.ChannelIP.value_or("127.0.0.1");
    auto port = config.Devices[assignedID].Config.ChannelPort.value_or(8090);
    client = std::make_unique<rpc::client>(ip, port);

    // Serialize configuration for remote initialization
    std::string asStr;
    {
      llvm::raw_string_ostream os(asStr);
      llvm::yaml::Output yout(os);
      yout << const_cast<config::TargetConfig &>(config);
    }
    client->call("connect", assignedID, asStr);
    cudaq::info("Ethernet device channel connected to {}:{}.", ip, port);
  }

  bool runs_on_separate_process() override { return true; }
  bool requires_unmarshaller() override { return true; }

  /// @brief Return the raw pointer corresponding to the
  /// provided device_ptr.
  void *get_raw_pointer(device_ptr &devPtr) override {
    throw std::runtime_error("cannot return the raw pointer for a channel "
                             "with data from another process.");
  }

  /// @brief Disconnect from the remote QPU controller.
  /// @details Sends a stop signal to the remote server to terminate
  /// communication.
  void disconnect() override {
    cudaq::info("Disconnecting Ethernet channel.");
    client->call("stopServer");
  }

  /// @brief Allocate memory on the remote QPU device.
  /// @param size The size of memory to allocate in bytes.
  /// @return A device pointer handle representing the allocated memory on the
  /// QPU.
  device_ptr malloc(std::size_t size) override {
    auto result = client->call("malloc", size);
    return {result.as<std::size_t>(), size, device_id};
  }

  /// @brief Free allocated memory on the remote QPU device.
  /// @param d The device pointer handle representing the memory block to free.
  void free(device_ptr &d) override { client->call("free", d.handle); }

  /// @brief Transfer data from host memory to remote QPU memory.
  /// @param src The destination device pointer on the QPU.
  /// @param dst Pointer to the source data in host memory.
  void send(device_ptr &src, const void *dst) override {
    auto size = src.size;
    std::vector<char> buffer(size);
    std::memcpy(buffer.data(), dst, size);
    cudaq::info("Ethernet Channel calling send with {}", src.handle);
    client->call("send", src.handle, buffer, size);
  }

  /// @brief Transfer data from remote QPU memory to host memory.
  /// @param dst Pointer to the destination buffer in host memory.
  /// @param src The source device pointer on the QPU.
  void recv(void *dst, device_ptr &src) override {
    auto size = src.size;
    std::vector<char> result =
        client->call("recv", src.handle, size).as<std::vector<char>>();
    cudaq::info("Ethernet Channel calling recv with {}", src.handle);
    std::memcpy(dst, result.data(), size);
  }

  /// @brief Load a callback function onto the remote QPU using MLIR FuncOp code
  /// representation.
  /// @param funcName Name of the callback function being loaded.
  /// @param unmarshallerCode MLIR FuncOp code for unmarshalling arguments
  /// during execution.
  void load_callback(const std::string &funcName,
                     const std::string &unmarshallerCode) override {
    cudaq::info("Ethernet Channel loading callback with name {}", funcName);
    funcNamesToHandles.insert(
        {funcName, client->call("load_callback", funcName, unmarshallerCode)
                       .as<std::size_t>()});
    return;
  }

  /// @brief Load a callback function using an unmarshaller function pointer
  /// (unsupported for Ethernet channels).
  /// @param funcName Name of the callback function being loaded.
  /// @param shmemUnmarshallerFunc Function pointer for shared memory
  /// unmarshalling logic.
  /// @throws std::runtime_error Always throws as this operation is unsupported
  /// for Ethernet channels.
  void load_callback(
      const std::string &funcName,
      KernelThunkResultType (*shmemUnmarshallerFunc)(void *, bool)) override {
    throw std::runtime_error("load_callback(func_ptr) - unsupported operation "
                             "for Ethernet channels.");
  }

  /// @brief Launch a callback function on the remote QPU with specified
  /// arguments.
  /// @param funcName Name of the callback function being executed.
  /// @param args Device pointer containing arguments for callback execution on
  /// the QPU.
  /// @return Result of callback execution containing any returned data or
  /// errors.
  launch_result launch_callback(const std::string &funcName,
                                const device_ptr &args,
                                cuda_launch_parameters params) override {
    cudaq::info("Ethernet Channel launching callback - {}", funcName);
    auto handle = funcNamesToHandles.at(funcName);
    auto resultData = client->call("launch_callback", handle, args.handle)
                          .as<std::vector<char>>();
    return {resultData};
  }

  /// @brief Check if this channel operates in a separate process space from the
  /// host controller.
  /// @return Always returns true for Ethernet-based channels as they operate
  /// remotely via RPC.
  bool runs_on_separate_process() override { return true; }

  CUDAQ_EXTENSION_CREATOR_FUNCTION(channel, ethernet_device_channel);
};

/// @brief Register Ethernet-based device channel as a CUDA Quantum extension
/// type. Enables runtime selection through "ethernet_device_channel" target
/// specification.
CUDAQ_REGISTER_EXTENSION_TYPE(ethernet_device_channel)

} // namespace cudaq::driver
