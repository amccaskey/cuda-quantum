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

class ethernet_dispatcher : public control_dispatcher {
private:
  std::unique_ptr<rpc::client> client;
  std::vector<std::string> symbolLocations;

public:
  /// @brief Connect to the process / memory space where
  /// the data should live.
  void connect(const config::TargetConfig &config) override {
    // If there is no controller IP address in the config, start one
    // locally
    std::string ip = "127.0.0.1";
    auto port = config.ControllerPort.value_or(8070);

    if (!config.ControllerIP) {
      auto cudaqControllerTool = cudaq::driver::getCUDAQControllerPath();
      auto argString = cudaqControllerTool.string();

      auto [ret, msg] = launchProcess(argString.c_str());
      if (ret == -1)
        throw std::runtime_error("Could not launch controller");

      for (int i = 0; i < 5; ++i) {
        if (is_server_available())
          break;

        std::this_thread::sleep_for(std::chrono::milliseconds(500));
      }
    } else
      ip = config.ControllerIP.value();

    client = std::make_unique<rpc::client>(ip, port);
    // FIXME handle didn't connect
    std::string asStr;
    {
      llvm::raw_string_ostream os(asStr);
      llvm::yaml::Output yout(os);
      yout << const_cast<config::TargetConfig &>(config);
    }
    client->call("connect", asStr);
    cudaq::info("controller channel connected to {}:{}.", ip, port);
  }

  void disconnect() override { client->call("stopServer"); }

  /// @brief Allocate data of given size on given device and
  /// return the device_ptr handle.
  device_ptr malloc(std::size_t size, std::size_t devId) override {
    auto result = client->call("malloc", size, devId);
    // need to create some sort of mapping
    return {result.as<std::size_t>(), size, devId};
  }

  /// @brief Free the data at the given device_ptr.
  void free(device_ptr &d) override { client->call("free", d.handle); }

  /// @brief Copy data to the given device_ptr.
  void send(device_ptr &src, const void *dst) override {
    auto size = src.size;
    std::vector<char> buffer(size);
    std::memcpy(buffer.data(), dst, size);
    cudaq::info("RPC Channel calling send with {}", src.handle);
    client->call("send", src.handle, buffer, size);
  }

  /// @brief Copy data from the given device_ptr.
  void recv(void *dst, device_ptr &src) override {
    // Get remote handle for destination pointer
    auto size = src.size;
    std::vector<char> result =
        client->call("recv", src.handle, size).as<std::vector<char>>();
    std::memcpy(dst, result.data(), size);
  }

  /// @brief Load the provided quantum kernel (target-specific JIT compilation)
  handle load_kernel(const std::string &quake) const override {
    auto kernelHandle = client->call("load_kernel", quake);
    // FIXME may want to do something with this here
    return kernelHandle.as<std::size_t>();
  }

  /// @brief Launch the quantum kernel with given thunk args.
  launch_result launch_kernel(handle kernelHandle,
                              device_ptr &argsHandle) const override {
    auto size = argsHandle.size;
    auto resultData =
        client->call("launch_kernel", kernelHandle, argsHandle.handle)
            .as<std::vector<char>>();
    return {resultData};
  }

  CUDAQ_EXTENSION_CREATOR_FUNCTION(control_dispatcher, ethernet_dispatcher)
};

CUDAQ_REGISTER_EXTENSION_TYPE(ethernet_dispatcher);

} // namespace cudaq::driver
