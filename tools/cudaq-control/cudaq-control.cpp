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

#include <thread>

static std::atomic<bool> _stopServer = false;

static llvm::cl::opt<int>
    port("port", llvm::cl::desc("TCP/IP port that the server will listen to."),
         llvm::cl::init(8080));
int main(int argc, char **argv) {

  llvm::cl::ParseCommandLineOptions(argc, argv, "cudaq-controller server\n");
  rpc::server srv(port);
  srv.bind("ping", []() { return true; });
  srv.bind("connect", &cudaq::driver::connect);
  srv.bind("malloc", &cudaq::driver::malloc);
  srv.bind("free", &cudaq::driver::free);
  srv.bind("memcpy_to", &cudaq::driver::memcpy_to);
  srv.bind("memcpy_from", &cudaq::driver::memcpy_from);
  srv.bind("stopServer", [&]() {
    _stopServer = true;
  });
  srv.async_run();

  while (true) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (_stopServer)
      break;
  }

  return 0;
}