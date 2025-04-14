/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "rpc/server.h"

#include "common/Logger.h"

#include "cudaq/Support/TargetConfig.h"
#include "cudaq/driver/controller/quake_compiler.h"

#include "llvm/Support/CommandLine.h"
#include <thread>

static std::atomic<bool> _stopServer = false;
static llvm::cl::opt<int>
    port("rpc-port",
         llvm::cl::desc("TCP/IP port that the server will listen to."),
         llvm::cl::init(8090));

namespace cudaq::rpcd {

std::unique_ptr<driver::quake_compiler> compiler;
std::map<std::size_t, std::string> handleToFuncNames;
std::map<std::size_t, std::pair<void *, std::size_t>> memory_pool;
std::vector<std::string> symbolLocations;

void connect(std::size_t assignedID, const std::string &cfg) {
  llvm::yaml::Input yin(cfg.c_str());
  config::TargetConfig config;
  yin >> config;

  // FIXME add this to the config
  compiler = driver::quake_compiler::get("default_compiler");
  compiler->initialize(config);
  symbolLocations = config.Devices[assignedID].Config.ExposedLibraries.value_or(
      std::vector<std::string>{});
}

std::size_t malloc(std::size_t size, std::size_t devId) {
  auto *ptr = std::malloc(size);
  auto hdl = reinterpret_cast<uintptr_t>(ptr);
  memory_pool.insert({hdl, {ptr, size}});
  return hdl;
}

void free(std::size_t handle) {
  std::free(memory_pool.at(handle).first);
  memory_pool.erase(handle);
}

void memcpy_to(std::size_t handle, std::vector<char> &data, std::size_t size) {
  auto *ptr = memory_pool.at(handle).first;
  std::memcpy(ptr, data.data(), size);
}

std::vector<char> memcpy_from(std::size_t handle, std::size_t size) {
  std::vector<char> ret(size);
  std::memcpy(ret.data(), memory_pool.at(handle).first, size);
  return ret;
}

std::size_t load_callback(const std::string &funcName,
                          const std::string &unmarshalCode) {
  return compiler->compile_unmarshaler(unmarshalCode, symbolLocations);
}

std::vector<char> launch_callback(std::size_t handle, std::size_t argsHandle) {
  auto *thunkPtr = memory_pool.at(argsHandle).first;
  auto size = memory_pool.at(argsHandle).second;
  compiler->launch(handle, thunkPtr);
  std::vector<char> ret(size);
  std::memcpy(ret.data(), thunkPtr, size);
  return ret;
}

} // namespace cudaq::rpcd

#define RPCD_BIND_FUNCTION(NAME) srv->bind(#NAME, &cudaq::rpcd::NAME);

int main(int argc, char **argv) {
  cudaq::info("Initialize RPC Daemon on Port {}", port);
  auto srv = std::make_unique<rpc::server>(port);

  RPCD_BIND_FUNCTION(connect)
  RPCD_BIND_FUNCTION(malloc)
  RPCD_BIND_FUNCTION(free)
  RPCD_BIND_FUNCTION(memcpy_to)
  RPCD_BIND_FUNCTION(memcpy_from)
  RPCD_BIND_FUNCTION(load_callback)
  RPCD_BIND_FUNCTION(launch_callback)

  srv->bind("ping", []() { return true; });
  srv->bind("stopServer", [&]() { _stopServer = true; });
  srv->async_run();
  while (true) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (_stopServer)
      break;
  }
}
