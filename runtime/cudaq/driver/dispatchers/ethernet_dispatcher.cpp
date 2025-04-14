/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/Logger.h"
#include "cudaq/Support/TargetConfig.h"
#include "cudaq/driver/channel.h"
#include "cudaq/utils/cudaq_utils.h"
#include "cudaq/utils/process.h"
#include "rpc/client.h"

namespace cudaq::driver {

/// @brief Checks availability of quantum controller server
/// @return True if server responds to ping within 200ms, false otherwise
/// @details Used during controller auto-launch sequence to verify readiness
bool is_server_available() {
  try {
    rpc::client client("127.0.0.1", 8080);
    client.set_timeout(200); // 200ms timeout
    client.call("ping");
    return true;
  } catch (...) {
    return false;
  }
}

/// @brief Ethernet-based quantum control dispatcher for remote QPU management
/// @details Handles RPC communication with quantum controller service for:
///          - Controller process lifecycle management
///          - Remote memory allocation/execution
///          - Quantum kernel deployment and execution
class ethernet_dispatcher : public control_dispatcher {
private:
  /// @brief RPC client for controller communication
  std::unique_ptr<rpc::client> client;

  /// @brief Library paths for symbol resolution
  std::vector<std::string> symbolLocations;

public:
  /// @brief Establish connection to quantum controller service
  /// @param config Target configuration parameters
  /// @details Automatically launches local controller if none specified in
  /// config.
  ///          Implements retry logic with 500ms intervals for 2.5 seconds
  ///          total.
  /// @throws std::runtime_error If local controller fails to start
  void connect(const config::TargetConfig &config) override {
    std::string ip = "127.0.0.1";
    auto port = config.ControllerPort.value_or(8070);

    if (!config.ControllerIP) {
      // Launch local controller process
      auto cudaqControllerTool = cudaq::driver::getCUDAQControllerPath();
      auto [ret, msg] = launchProcess(cudaqControllerTool.string().c_str());
      if (ret == -1)
        throw std::runtime_error("Controller launch failed: " + msg);

      // Wait for controller readiness
      for (int i = 0; i < 5; ++i) {
        if (is_server_available())
          break;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
      }
    } else {
      ip = config.ControllerIP.value();
    }

    // Establish RPC connection
    client = std::make_unique<rpc::client>(ip, port);

    // Serialize and transmit configuration
    std::string asStr;
    llvm::raw_string_ostream os(asStr);
    llvm::yaml::Output yout(os);
    yout << const_cast<config::TargetConfig &>(config);
    client->call("connect", asStr);

    cudaq::info("Controller channel established: {}:{}", ip, port);
  }

  /// @brief Terminate connection to quantum controller
  void disconnect() override { client->call("stopServer"); }

  /// @brief Allocate device memory through remote controller
  /// @param size Memory size in bytes
  /// @param devId Target QPU device ID
  /// @return device_ptr handle to remote memory allocation
  device_ptr malloc(std::size_t size, std::size_t devId) override {
    return {client->call("malloc", size, devId).as<std::size_t>(), size, devId};
  }

  /// @brief Release device memory through remote controller
  /// @param d device_ptr handle to release
  void free(device_ptr &d) override { client->call("free", d.handle); }

  /// @brief Transfer data to remote QPU memory
  /// @param src Destination device pointer
  /// @param dst Host memory source buffer
  void send(device_ptr &src, const void *dst) override {
    std::vector<char> buffer(src.size);
    std::memcpy(buffer.data(), dst, src.size);
    client->call("send", src.handle, buffer, src.size);
  }

  /// @brief Retrieve data from remote QPU memory
  /// @param dst Host memory destination buffer
  /// @param src Source device pointer
  void recv(void *dst, device_ptr &src) override {
    auto result =
        client->call("recv", src.handle, src.size).as<std::vector<char>>();
    std::memcpy(dst, result.data(), src.size);
  }

  /// @brief Compile quantum kernel through remote controller
  /// @param quake Kernel code in MLIR/Quake format
  /// @return handle to compiled kernel executable
  handle load_kernel(const std::string &quake) const override {
    return client->call("load_kernel", quake).as<std::size_t>();
  }

  /// @brief Execute compiled quantum kernel on remote QPU
  /// @param kernelHandle Compiled kernel handle from load_kernel
  /// @param argsHandle Device pointer containing kernel arguments
  /// @return launch_result with execution data/errors
  launch_result launch_kernel(handle kernelHandle,
                              device_ptr &argsHandle) const override {
    return {client->call("launch_kernel", kernelHandle, argsHandle.handle)
                .as<std::vector<char>>()};
  }

  CUDAQ_EXTENSION_CREATOR_FUNCTION(control_dispatcher, ethernet_dispatcher)
};

/// @brief Register Ethernet dispatcher as control dispatcher implementation
/// Enables runtime selection via "ethernet_dispatcher" target specification
CUDAQ_REGISTER_EXTENSION_TYPE(ethernet_dispatcher)

} // namespace cudaq::driver
