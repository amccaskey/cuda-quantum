/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq/driver/controller/controller.h"
#include "rpc/server.h"
#include "llvm/Support/CommandLine.h"

static std::atomic<bool> _stopServer = false;
static llvm::cl::opt<int>
    port("port", llvm::cl::desc("TCP/IP port that the server will listen to."),
         llvm::cl::init(8080));

namespace cudaq::driver {
class rpc_controller : public controller {
protected:
  std::unique_ptr<rpc::server> srv;

public:
  void initialize(int argc, char **argv) override {
    cudaq::info("Initialize RPC Controller on Port {}", port);
    srv = std::make_unique<rpc::server>(port);
    srv->bind("ping", []() { return true; });
    srv->bind("connect", [&](const std::string &cfg) { this->connect(cfg); });
    srv->bind("malloc", [&](std::size_t size, std::size_t devId) {
      return this->malloc(size, devId);
    });
    srv->bind("free", [&](std::size_t handle) { this->free(handle); });
    srv->bind("memcpy_to",
              [&](std::size_t handle, std::vector<char> &data,
                  std::size_t size) { this->memcpy_to(handle, data, size); });
    srv->bind("memcpy_from", [&](std::size_t handle, std::size_t size) {
      return this->memcpy_from(handle, size);
    });
    srv->bind("load_kernel", [&](const std::string &quake) {
      return this->load_kernel(quake);
    });
    srv->bind("get_callbacks",
              [&](handle h) { return this->get_callbacks(h); });
    srv->bind("launch_kernel",
              [&](std::size_t kernelHandle, std::size_t argsHandle) {
                return this->launch_kernel(kernelHandle, argsHandle);
              });
    srv->bind("stopServer", [&]() { _stopServer = true; });
    srv->async_run();
  }
  bool should_stop() override { return _stopServer; }

  CUDAQ_EXTENSION_CREATOR_FUNCTION(controller, rpc_controller);
};

CUDAQ_REGISTER_EXTENSION_TYPE(rpc_controller)

} // namespace cudaq::driver
