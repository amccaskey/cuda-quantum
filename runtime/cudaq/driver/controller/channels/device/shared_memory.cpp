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

INSTANTIATE_REGISTRY_NO_ARGS(cudaq::driver::device_channel)
INSTANTIATE_REGISTRY_NO_ARGS(cudaq::driver::controller_channel)

namespace cudaq::driver {

class shared_memory : public device_channel {
  std::unique_ptr<quake_compiler> unmarshalCompiler;
  std::map<std::string, std::size_t> handles;
  std::vector<std::string> symbol_locations;

  std::map<handle, void *> local_memory_pool;

public:
  using device_channel::device_channel;

  void connect(std::size_t assignedID,
               const config::TargetConfig &config) override {
    cudaq::info("shared_memory channel connected.");
    unmarshalCompiler = quake_compiler::get("default_compiler");
    unmarshalCompiler->initialize(config);
  }

  void add_symbol_locations(const std::vector<std::string> &locs) {
    symbol_locations = locs;
  }

  device_ptr malloc(std::size_t size, std::size_t devId) override {
    cudaq::info("shared memory channel allocating data {} {}", size, devId);
    auto *raw = std::malloc(size);
    local_memory_pool.insert({reinterpret_cast<uintptr_t>(raw), raw});
    return {reinterpret_cast<uintptr_t>(raw), size, devId};
  }

  void free(device_ptr &d) override {
    std::free(local_memory_pool.at(d.handle));
  }

  void memcpy(device_ptr &arg, const void *src) override {
    cudaq::info("shared memory channel memcpy data {}", arg.size);
    std::memcpy(local_memory_pool.at(arg.handle), src, arg.size);
  }
  void memcpy(void *dst, device_ptr &src) override {
    std::memcpy(dst, local_memory_pool.at(src.handle), src.size);
  }

  void load_callback(const std::string &funcName,
                     const std::string &unmarshallerCode) override {
    cudaq::info("shared memory load_callback called - {}", funcName);
    auto handle = unmarshalCompiler->compile_unmarshaler(unmarshallerCode,
                                                         symbol_locations);
    handles.insert({funcName, handle});
  }

  launch_result launch_callback(const std::string &funcName,
                                device_ptr &args) override {
    // Here we are given the function name "add"
    // We want to get the "unmarshal.add" function and run it
    cudaq::info("shared_memory channel launching callback {}", funcName);
    auto handle = handles[funcName];
    unmarshalCompiler->launch(handle, local_memory_pool.at(args.handle));
    // FIXME I'm not sure this is necessary, the
    // channels are local to the controller, and handle any
    // remote aspects in an encapsulated way
    std::vector<char> resVec(args.size);
    std::memcpy(resVec.data(), local_memory_pool.at(args.handle), args.size);
    return {resVec};
  }

  CUDAQ_EXTENSION_CREATOR_FUNCTION(device_channel, shared_memory);
};

CUDAQ_REGISTER_EXTENSION_TYPE(shared_memory)

} // namespace cudaq::driver
