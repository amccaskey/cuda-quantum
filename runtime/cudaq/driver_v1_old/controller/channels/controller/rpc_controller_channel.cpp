/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/Logger.h"
#include "cudaq/Support/TargetConfig.h"

#include "cudaq/driver/controller/channel.h"
#include "cudaq/driver/controller/utils/process.h"

#define CUDAQ_RTTI_DISABLED
#include "cudaq/utils/cudaq_utils.h"
#undef CUDAQ_RTTI_DISABLED

#include "rpc/client.h"

#include <filesystem>
#include <stdexcept>
#include <string>
#include <thread>

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

namespace cudaq::driver {

class rpc_controller_channel : public controller_channel {
protected:
  bool startedLocalServer = false;

public:
  using controller_channel::controller_channel;

  std::unique_ptr<rpc::client> client;

  void connect(std::size_t assignedID,
               const config::TargetConfig &config) override {

    // If there is no controller IP address in the config, start one
    // locally
    std::string ip = "127.0.0.1";
    auto port = config.ControllerPort.value_or(8080);

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

  device_ptr malloc(std::size_t size, std::size_t devId) override {
    auto result = client->call("malloc", size, devId);
    // need to create some sort of mapping
    return {result.as<std::size_t>(), size, devId};
  }

  void free(device_ptr &d) override { client->call("free", d.handle); }

  // copy to QPU
  void memcpy(device_ptr &src, const void *dst) override {
    auto size = src.size;
    std::vector<char> buffer(size);
    std::memcpy(buffer.data(), dst, size);
    cudaq::info("RPC Channel calling memcpy_to with {}", src.handle);
    client->call("memcpy_to", src.handle, buffer, size);
  }

  void memcpy(void *dst, device_ptr &src) override {
    // Get remote handle for destination pointer
    auto size = src.size;
    std::vector<char> result =
        client->call("memcpy_from", src.handle, size).as<std::vector<char>>();
    std::memcpy(dst, result.data(), size);
  }

  handle load_kernel(const std::string &quake) const {
    auto kernelHandle = client->call("load_kernel", quake);
    // FIXME may want to do something with this here
    return kernelHandle.as<std::size_t>();
  }

  std::vector<std::string> get_callbacks(handle kernelHandle) override {
    return client->call("get_callbacks", kernelHandle)
        .as<std::vector<std::string>>();
  }

  launch_result launch_kernel(handle kernelHandle,
                              device_ptr &argsHandle) const {
    auto size = argsHandle.size;
    auto resultData =
        client->call("launch_kernel", kernelHandle, argsHandle.handle)
            .as<std::vector<char>>();
    return {resultData};
  }

  ~rpc_controller_channel() {
    if (startedLocalServer) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1500));
      cudaq::info("shutdown controller channel.");
      client->call("stopServer");
    }
  }

  CUDAQ_EXTENSION_CREATOR_FUNCTION(controller_channel, rpc_controller_channel);
};

CUDAQ_REGISTER_EXTENSION_TYPE(rpc_controller_channel)

} // namespace cudaq::driver
