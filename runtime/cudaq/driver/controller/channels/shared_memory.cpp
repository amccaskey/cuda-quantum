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

#include <dlfcn.h>

INSTANTIATE_REGISTRY_NO_ARGS(cudaq::driver::channel)
INSTANTIATE_REGISTRY_NO_ARGS(cudaq::driver::controller_channel)

namespace cudaq::driver {

class shared_memory : public channel {
  std::unique_ptr<quake_compiler> unmarshalCompiler;
  std::map<std::string, std::size_t> handles;

public:
  using channel::channel;

  void connect(std::size_t assignedID,
               const config::TargetConfig &config) override {
    cudaq::info("shared_memory channel connected.");
    unmarshalCompiler = quake_compiler::get("default_compiler");
    unmarshalCompiler->initialize(config);
  }

  device_ptr malloc(std::size_t size, std::size_t devId) override {
    cudaq::info("shared memory channel allocating data {} {}", size, devId);
    return {std::malloc(size), size, devId};
  }

  void free(device_ptr &d) override { std::free(d.data); }
  void free(std::size_t argsHandle) override {}

  void memcpy(device_ptr &arg, const void *src) override {
    cudaq::info("shared memory channel memcpy data {}", arg.size);
    std::memcpy(arg.data, src, arg.size);
  }
  void memcpy(void *dst, device_ptr &src) override {
    std::memcpy(dst, src.data, src.size);
  }

  void load_callback(const std::string &funcName,
                     const std::string &unmarshallerCode) override {
    cudaq::info("shared memory load_callback called");
    auto handle = unmarshalCompiler->compile_unmarshaler(unmarshallerCode);
    handles.insert({funcName, handle});
  }

  launch_result launch_callback(const std::string &funcName,
                                device_ptr &args) override {
    // Here we are given the function name "add"
    // We want to get the "unmarshal.add" function and run it
    cudaq::info("HELLO WE ARE LAUNCHING {}", funcName);
    auto handle = handles[funcName];
    unmarshalCompiler->launch_callback(handle, args.data);
    // auto *handle = dlopen("/workspaces/cuda-quantum/build/a.out", RTLD_LAZY);
    // if (!handle)
    //   printf("BAD DLOPEN %s\n", dlerror());
    // std::string unmarshalName = "unmarshal." + funcName;
    // auto *symbol = dlsym(handle, unmarshalName.c_str());
    // const char *error = dlerror();
    // printf("Error here? %s\n", error);
    // auto fcn =
    //     reinterpret_cast<cudaq::CallbackResultType (*)(void *,
    //     bool)>(symbol);
    // auto res = fcn(args.data, false);
    return {args, 0, ""};
  }

  CUDAQ_EXTENSION_CREATOR_FUNCTION(channel, shared_memory);
};

CUDAQ_REGISTER_EXTENSION_TYPE(shared_memory)

} // namespace cudaq::driver
