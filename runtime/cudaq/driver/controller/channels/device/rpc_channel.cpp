/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/ThunkInterface.h"
#include "cudaq/Support/TargetConfig.h"
#include "cudaq/driver/controller/channel.h"
#include "cudaq/driver/controller/quake_compiler.h"

#include "cudaq/driver/controller/utils/process.h"
#include "rpc/client.h"

#include <dlfcn.h>

namespace cudaq::driver {

class rpc_channel : public device_channel {

  std::unique_ptr<rpc::client> client;

  std::map<std::string, std::size_t> funcNamesToHandles;

public:
  using device_channel::device_channel;

  void connect(std::size_t assignedID,
               const config::TargetConfig &config) override {
    std::string ip =
        config.Devices[assignedID].Config.ChannelIP.value_or("127.0.0.1");
    auto port = config.Devices[assignedID].Config.ChannelPort.value_or(8090);
    client = std::make_unique<rpc::client>(ip, port);
    // FIXME handle didn't connect
    std::string asStr;
    {
      llvm::raw_string_ostream os(asStr);
      llvm::yaml::Output yout(os);
      yout << const_cast<config::TargetConfig &>(config);
    }
    client->call("connect", assignedID, asStr);
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

  void load_callback(const std::string &funcName,
                     const std::string &unmarshallerCode) override {
    funcNamesToHandles.insert(
        {funcName, client->call("load_callback", funcName, unmarshallerCode)
                       .as<std::size_t>()});
    return;
  }

  launch_result launch_callback(const std::string &funcName,
                                device_ptr &args) override {
    auto handle = funcNamesToHandles.at(funcName);
    auto resultData = client->call("launch_callback", handle, args.handle)
                          .as<std::vector<char>>();
    return {resultData};
  }

  CUDAQ_EXTENSION_CREATOR_FUNCTION(device_channel, rpc_channel);
};

CUDAQ_REGISTER_EXTENSION_TYPE(rpc_channel)

} // namespace cudaq::driver
